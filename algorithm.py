"""为一条已选路径选择共同信道和逐跳纤芯，不直接占用或释放资源。

由 main.ClassicalService 通过 ResourceAllocator 调用。输入资源和功率数组
为 [源节点, 目的节点, 芯, 信道]，资源状态 1 表示可用，2 表示经典占用，
3 表示量子保留；功率 W、距离 m、实际频率 Hz。返回 (各跳芯列表, 信道索引)，
无可用分配时为 (None, -1)。普通仿真全路径使用同一信道，中间节点可以换芯。
仅实验导出启用 bind_three，此时每跳返回固定方向组的三芯列表，三芯同时加载。

四种算法以 KeyConsumption_24node -7 core 的 Consumption_Dynamic.py
及 algorithm_MY/FF/SCWA.py 为来源；其中 my 即 CQLI。CQLI/CCA/SCWA 沿用参考
算法名称，参考源码未注明其英文全称。参考频率数组升序，故四者均
按实际频率从低到高搜索。保留本项目频点和量子预留，不迁移参考仿真配置。
FF（first-fit，遇到可用资源就选）与 CCA 双向共享经典芯，禁止反向同芯同频
同时占用；CQLI 按方向分芯。FF 同频时按芯编号升序选择，七芯为 [0,1,2,3,4,5]，
对应参考搜索函数 range(core_num)，而非参考初始化列表的书写顺序。
CCA/CQLI 保留方向列表顺序；默认 CCA 为 [1,2,3,4,5,6]。
SCWA 保留奇偶分芯偏好，但修复小网格的阈值和无回退问题：只统计经典可用
位置，原首选集合占用比例达到参考的 14/24 时交换偏好；当前首选集合不能
贯通全路径时放开另一集合。序号按全部实际频率升序从 0 编起，包括量子频点，
但量子/禁用位置不计入占用率。这是可变信道数适配版，不等同于参考原版。
三芯绑定是硬件实验扩展，使用固定方向分组，不等同于参考单芯 FF。
GREEDY_MIN_NOISE 直接比较量子接收端的拉曼与 FWM 总噪声增量；
不做 FWM 组合预筛选、奇偶分级或频差优先。噪声容差内优先选择同向同频的相邻芯占用数较少的候选。
"""
import numpy as np


ALGORITHMS = ('CQLI', 'CCA', 'FF', 'SCWA', 'GREEDY_MIN_NOISE')
GREEDY_NOISE_RTOL = 0.10


def normalize_algorithm(value):
    """统一大小写及连字符；first-fit/FF 统一为 FF，是否支持该名称由调用方检查。"""
    name = value.upper().replace('-', '_')
    return 'FF' if name == 'FIRST_FIT' else name


class GreedyMinNoise:
    """先限制量子端噪声增量，再优先选择同向同频的相邻芯占用数较少的分配。"""
    def __init__(self, forward_cores, backward_cores, scorer, *, noise_rtol=GREEDY_NOISE_RTOL):
        self.forward = tuple(forward_cores)
        self.backward = tuple(backward_cores)
        self.scorer = scorer
        self.frequencies = scorer.frequencies.copy()
        if not np.isfinite(noise_rtol) or not 0 <= noise_rtol <= 1:
            raise ValueError('greedy noise_rtol must be finite and in [0, 1]')
        self.noise_rtol = float(noise_rtol)

    def _candidate(self, a, b, core, wave, launch_power, resources, powers, distance):
        """返回本跳本芯加入 wave 的量子端总噪声增量 noise（W）及芯编号。

        分数为加入前后芯间自发拉曼散射（SpRS）与芯间四波混频（FWM）光功率之差，
        对各量子芯的所有量子频点求和。量子接收方向为小节点号到大节点号；
        a>b 表示经典光反向传播。QuantumLinkScorer 判定最近/次近邻和方向，
        并调用 noise_calculation 中的物理公式；远芯返回零是现有模型截断。
        不计算经典端 OSNR（光信噪比）、暗计数或同频线性串扰，不增加拉曼谱或频差
        的辅助排序。只修改功率副本，不占用资源；没有量子频点时增量为零。
        """
        i, j = sorted((a, b))
        active = np.where(resources[a, b, core] == 2, powers[a, b, core], 0.0)
        trial = active.copy()
        trial[wave] = launch_power
        total_delta = 0.0
        for qc in np.flatnonzero(np.any(resources[i, j] == 3, axis=1)):
            qi = tuple(int(q) for q in np.flatnonzero(resources[i, j, qc] == 3))
            before_r, before_f = self.scorer.core_components(qc, core, a>b, active, qi, distance)
            after_r, after_f = self.scorer.core_components(qc, core, a>b, trial, qi, distance)
            total_delta += float(np.sum(after_r-before_r) + np.sum(after_f-before_f))
        return dict(core=core, noise=total_delta)

    def _same_direction_count(self, a, b, core, wave, resources):
        """数本跳最近邻芯上同向、同频的经典占用数，每个占用计1。

        相邻芯使用布局的 first 邻接表；只统计 resources[a,b,邻芯,wave]==2。
        不计反向、次近邻和量子保留，不按功率或耦合系数加权，不计算串扰噪声。
        """
        return sum(int(resources[a, b, neighbor, wave] == 2)
                   for neighbor in self.scorer.first[core])

    def allocate(self, path, launch_power, resources, powers, distances):
        """在给定路径上按总噪声增量选择共同频点和逐跳纤芯。

        对每个全路径可用频点，逐跳求最小总噪声增量并相加；再在所有频点之间
        求 Nmin。不设 FWM 安全等级、奇偶优先或频差优先，FWM 仅通过物理功率计分。
        候选须满足 N <= Nmin+rtol*abs(Nmin)，默认 rtol=10%；Nmin=0 时无绝对容差。
        每跳仅在该跳相同相对容差内选择同向同频邻芯占用数较少的芯，并复核全路径总和不超上述上限；
        超限则恢复该频点的逐跳最低噪声芯。最终按邻芯占用计数之和、量子噪声增量、频点索引、
        芯编号排序。rtol=0 时严格最小化本路径的总增量，精确同分仍优先减少同向同频邻芯占用。
        正容差下是逐跳贪心，不穷举芯组合；容差不约束后续仿真的整体 SKR 损失。

        路由顺序由 main 决定，本函数不跨候选路径比较分数；全路径同频，可逐跳换芯。
        无可用资源或不足两节点返回 (None,-1)，不因噪声分数增加拒绝条件。
        重复节点路径、非正或非有限功率报错；仅检查本方向空闲，不追加反向互斥。
        三芯绑定另由 ResourceAllocator._bound_three 直接最小化三芯总增量，
        不启用邻芯占用计数排序和相对容差。
        """
        if len(path) < 2:
            return None, -1
        if len(set(path)) != len(path):
            raise ValueError('greedy_min_noise requires a simple path')
        if not np.isfinite(launch_power) or launch_power <= 0:
            raise ValueError('Launch power must be finite and positive')
        options = []
        for wave in range(resources.shape[-1]):
            per_hop = []
            for a, b in zip(path, path[1:]):
                candidates = [self._candidate(a,b,c,wave,launch_power,resources,powers,distances[a,b])
                              for c in (self.forward if a<b else self.backward)
                              if resources[a,b,c,wave] == 1]
                for candidate in candidates:
                    candidate['neighbors'] = self._same_direction_count(
                        a, b, candidate['core'], wave, resources)
                if not candidates:
                    break
                per_hop.append(candidates)
            if len(per_hop) != len(path)-1:
                continue
            chosen = [min(hop, key=lambda c: (c['noise'], c['neighbors'], c['core'])) for hop in per_hop]
            options.append(dict(wave=wave, chosen=chosen, per_hop=per_hop))
        if not options:
            return None, -1
        minimum = min(sum(c['noise'] for c in option['chosen']) for option in options)
        limit = minimum + self.noise_rtol * abs(minimum)
        ranked = []
        for option in options:
            original = option['chosen']
            if sum(c['noise'] for c in original) > limit:
                continue
            wave = option['wave']
            chosen = []
            for hop in option['per_hop']:
                local_min = min(c['noise'] for c in hop)
                local_limit = local_min + self.noise_rtol * abs(local_min)
                near = [c for c in hop if c['noise'] <= local_limit]
                chosen.append(min(near, key=lambda c: (
                    c['neighbors'], c['noise'], c['core'])))
            if sum(c['noise'] for c in chosen) > limit:
                chosen = original
            neighbor_count = sum(c['neighbors'] for c in chosen)
            cores = [c['core'] for c in chosen]
            key = (neighbor_count, sum(c['noise'] for c in chosen), wave, tuple(cores))
            ranked.append((key, cores, wave))
        _, cores, wave = min(ranked, key=lambda candidate: candidate[0])
        return cores, wave


class ResourceAllocator:
    """构造时传入算法、方向纤芯和评分器；每次分配显式传入实时状态。

    不持有仿真对象或资源快照，不修改传入数组。
    """
    def __init__(self, algorithm, forward_cores, backward_cores, frequencies, quantum_scorer=None,
                 *, bind_three=False, noise_rtol=GREEDY_NOISE_RTOL):
        self.algorithm = normalize_algorithm(algorithm)
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.frequencies = np.asarray(frequencies, dtype=float)
        if (self.frequencies.ndim != 1 or not len(self.frequencies)
                or not np.all(np.isfinite(self.frequencies)) or np.any(self.frequencies <= 0)):
            raise ValueError('frequencies must contain positive finite Hz values')
        self.core_f = tuple(forward_cores)
        self.core_b = tuple(backward_cores)
        # 仅实验导出开启；普通仿真和负载扫描仍逐跳选单芯。
        self.bind_three = bind_three
        self.reverse_exclusive = self.algorithm in ('FF', 'CCA') and not bind_three
        if bind_three and (self.algorithm not in ('FF', 'CCA', 'GREEDY_MIN_NOISE')
                           or len(self.core_f) != 3 or len(self.core_b) != 3
                           or len(set(self.core_f + self.core_b)) != 6):
            raise ValueError('Three-core experiments require FF/CCA/greedy and two disjoint three-core groups')
        if self.algorithm == "GREEDY_MIN_NOISE" and quantum_scorer is None:
            raise ValueError(f"{self.algorithm} requires a quantum receiver scorer")

        self.noise_policy = (GreedyMinNoise(self.core_f, self.core_b, quantum_scorer, noise_rtol=noise_rtol)
                             if self.algorithm == "GREEDY_MIN_NOISE" else None)

    def allocate(self, path, launch_power, *, resources, powers, distances):
        """返回 (各跳芯列表, 共同信道索引)，失败为 (None, -1)。
        
        默认每跳返回单芯编号；仅 bind_three=True 时每跳返回三个芯的列表。
        launch_power/powers 为 W，distances 为 m；不修改输入数组，实际占用/释放由 main 处理。
        """
        if self.bind_three:
            return self._bound_three(path, launch_power, resources, powers, distances)
        if self.noise_policy is not None:
            return self.noise_policy.allocate(path, launch_power, resources, powers, distances)
        if self.algorithm == "SCWA":
            return self._scwa(path, resources)
        return self._first_fit(path, resources)

    def _bound_three(self, path, launch_power, resources, powers, distances):
        """实验专用：返回 (逐跳三芯成员列表, 共同信道索引)，失败为 (None, -1)。

        每跳按节点号选择方向组，三芯全部空闲才可接入，反向资源独立。
        FF/CCA 按实际频率从低到高选第一个可行波长；CCA 使用实验专用方向三芯组。greedy 直接最小化全路径三芯的
        拉曼与 FWM 总增量，完全同分时选较小信道索引；保留实际邻接耦合差异。
        现有物理模型按经典芯相加，因此这等于三芯同时加载的增量，不是单芯乘三。
        每芯每信道均加载 launch_power W；仅评分副本，main 的事件负责实际占用。
        三芯强制同频，本模式不应用邻芯占用计数排序或近似噪声容差。
        """
        policy = self.noise_policy
        if len(path) < 2:
            return None, -1
        if len(set(path)) != len(path):
            raise ValueError('Three-core experiments require a simple path')
        if not np.isfinite(launch_power) or launch_power <= 0:
            raise ValueError('Launch power must be finite and positive')
        groups = [list(self.core_f if a < b else self.core_b) for a, b in zip(path, path[1:])]
        best_key, best = None, (None, -1)
        for wave in sorted(range(resources.shape[-1]), key=lambda w: (self.frequencies[w], w)):
            if not all(np.all(resources[a, b, cores, wave] == 1)
                       for a, b, cores in zip(path, path[1:], groups)):
                continue
            if policy is None:
                return groups, wave
            candidates = [policy._candidate(a, b, c, wave, launch_power, resources, powers, distances[a, b])
                          for a, b, cores in zip(path, path[1:], groups) for c in cores]
            key = (sum(c['noise'] for c in candidates), wave)
            if best_key is None or key < best_key:
                best_key, best = key, (groups, wave)
        return best

    def _first_fit(self, path, resources):
        """FF/CCA/CQLI 先按实际频率从低到高，再选择本跳第一个可用芯。

        三者复用 first-fit 搜索；每跳按节点号选择前向或后向列表。
        FF 按芯编号升序搜索，七芯两方向均为 [0,1,2,3,4,5]；即使传入列表乱序也排序。
        CCA/CQLI 保留列表顺序，默认 CCA 两方向均为 [1,2,3,4,5,6]。
        FF/CCA 还要求反向同芯同频未被经典业务占用；CQLI 只检查本方向。
        同频按原索引破同分；没有共同可用频点或不足两节点时返回 (None,-1)。
        """
        if len(path) < 2:
            return None, -1
        for wave in sorted(range(resources.shape[-1]), key=lambda w: (self.frequencies[w], w)):
            cores = []
            for a, b in zip(path, path[1:]):
                group = self.core_f if a < b else self.core_b
                if self.algorithm == "FF":
                    group = sorted(group)
                core = next((c for c in group if resources[a, b, c, wave] == 1
                             and (not self.reverse_exclusive or resources[b, a, c, wave] != 2)), None)
                if core is None:
                    break
                cores.append(int(core))
            if len(cores) == len(path) - 1:
                return cores, wave
        return None, -1

    def _scwa(self, path, resources):
        """SCWA 可变信道适配：奇偶偏好、按占用比例切换、全路径失败后回退。

        默认前向奇数芯 [4]、偶数芯 [3,5]；后向奇数芯 [6]、偶数芯 [0,2]。
        奇偶序号按全部实际频率升序从 0 编起；同频用原索引破同分。
        对每跳的原首选奇偶集合，仅状态 1/2 算有效经典位置、仅状态 2 算占用。
        占用比例达到 7/12（参考 16 信道名义阈值 14/24）时交换本跳偏好，
        每次到达重新判断；没有有效首选位置时直接偏好互补集合。
        默认 8 个总频点、1 个量子频点时，首选有效容量为 11，占用 7 个才切换，
        不再因量子预留位置使第一条业务后就切换。比例推广并非原版固定余量 10。

        先只在各跳当前首选集合中按低频优先找共同频点；整条路径失败后，重新
        按低频优先搜索全部方向芯。回退时每跳仍先尝试其当前首选芯，再尝试
        互补芯，因此允许一条路径的不同跳采用不同集合，不因奇偶偏好单独阻塞。
        同频同集合按方向芯列表顺序选择。不计算噪声，不修改资源；不足两节点
        或所有方向芯都没有全路径共同空闲频点时返回 (None,-1)。显式覆盖布局
        时仍将各方向第一芯作为奇数芯，其余作为偶数芯，属于消融。
        """
        if len(path) < 2:
            return None, -1
        waves = sorted(range(resources.shape[-1]), key=lambda w: (self.frequencies[w], w))
        parity = {wave: rank % 2 for rank, wave in enumerate(waves)}
        hops = []
        for a, b in zip(path, path[1:]):
            group = self.core_f if a < b else self.core_b
            preferred = [resources[a, b, core, wave]
                         for index, core in enumerate(group) for wave in waves
                         if parity[wave] == (1 if index == 0 else 0)]
            capacity = sum(state in (1, 2) for state in preferred)
            occupied = sum(state == 2 for state in preferred)
            swapped = capacity == 0 or occupied * 12 >= capacity * 7
            hops.append((a, b, group, swapped))
        for allow_fallback in (False, True):
            for wave in waves:
                cores = []
                for a, b, group, swapped in hops:
                    core = next((c for index, c in enumerate(group)
                                 if parity[wave] == ((1 if index == 0 else 0) ^ swapped)
                                 and resources[a, b, c, wave] == 1), None)
                    if core is None and allow_fallback:
                        core = next((c for c in group if resources[a, b, c, wave] == 1), None)
                    if core is None:
                        break
                    cores.append(core)
                if len(cores) == len(path) - 1:
                    return cores, wave
        return None, -1
