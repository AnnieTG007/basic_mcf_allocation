"""供调用方使用 ResourceAllocator(...).allocate(...)，从资源与功率状态选择共同信道和逐跳芯组。
仅支持固定七芯配置，返回分配结果而不修改资源。"""
import numpy as np

from core_layout import SEVEN_CORE_LAYOUTS
from skr_calculation import calculate_quantum_noise_components


# ALGORITHMS 为唯一算法名称集合，例如命令行选择 'QCNM'。
ALGORITHMS = ('CQLI', 'CCA', 'FF', 'SCWA', 'QCNM')


class ResourceAllocator:
    """用五种算法在既定候选组中选择资源；组内所有芯必须在同一信道一起分配。"""

    def __init__(
            self,
            algorithm,             # 算法名称，例如 'QCNM'，只接受 ALGORITHMS 中的名称。
            forward_groups,        # 前向候选组，例如 [(2,), (4,)] 或 [(2, 4, 6)]。
            backward_groups,       # 后向候选组，例如 [(3,), (5,)] 或 [(3, 5, 7)]。
            frequencies,           # 信道索引对应的频率（Hz），例如 [193.5e12, 193.4e12]。
            *,
            allow_bidirectional=False,  # 是否允许同一纤芯、同一信道内双向同频数据信号同时传输；默认 False（不允许）。
            # 芯是否能承载某方向由前后向候选列表决定；双向共有芯仍可波长 1 正向、波长 2 反向，本开关只约束同频。
    ):
        """前向指节点号小到大，后向相反；[(2,), (4,)] 是两个备选组，[(2, 4)] 是一个两芯组。"""
        self.algorithm = algorithm
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"未知算法：{algorithm}")
        self.frequencies = np.asarray(frequencies, dtype=float)
        if (self.frequencies.ndim != 1 or not len(self.frequencies)
                or not np.all(np.isfinite(self.frequencies)) or np.any(self.frequencies <= 0)):
            raise ValueError('信道频率必须为正的有限赫兹值')
        # group 是必须一起分配的芯编号元组，例如 (2, 4, 6)；保存为元组，后续按固定表排序。
        self.forward_groups = tuple(tuple(group) for group in forward_groups)
        self.backward_groups = tuple(tuple(group) for group in backward_groups)
        if any(not group for group in self.forward_groups + self.backward_groups):
            raise ValueError('候选芯组不能为空')
        # layout 为本算法的固定七芯配置，例如 SEVEN_CORE_LAYOUTS['SCWA']。
        layout = SEVEN_CORE_LAYOUTS[self.algorithm]
        # groups/allowed 为方向候选及固定允许芯，例如 ((6,), (5,), (7,)) / (6,5,7)。
        for groups, allowed in ((self.forward_groups, layout['classical_forward']),
                                (self.backward_groups, layout['classical_backward'])):
            if any(core not in allowed for group in groups for core in group):
                raise ValueError('候选芯组必须落在本算法的固定七芯配置内')
        # 按固定表顺序排列候选；绑定约束不改变算法的芯搜索次序。
        self.forward_groups = tuple(sorted(self.forward_groups, key=lambda group: tuple(
            layout['classical_forward'].index(core) for core in group)))
        self.backward_groups = tuple(sorted(self.backward_groups, key=lambda group: tuple(
            layout['classical_backward'].index(core) for core in group)))
        self.allow_bidirectional = allow_bidirectional

    def allocate(
            self,
            path,           # 简单路径的节点序列，例如 [0, 1, 2]；不跨候选路径比较分数。
            launch_power,   # 每个新增芯信道的发射功率，例如 0.01 W。
            *,
            resources,      # [源节点, 目的节点, 芯, 信道] 状态数组，例如 resources[0,1,2,3]=1。
            powers,         # 同维度已占用功率数组，例如 powers[0,1,2,3]=0.01 W。
            distances,      # 节点距离矩阵，例如 distances[0,1]=1000 m。
            **algorithm_options,  # QCNM 专用参数，如 noise_rtol=0.1 及物理模型、邻芯表。
    ):
        """返回 (逐跳芯组列表, 共同信道索引)，例如 ([[2], [4]], 3)；无可用分配返回 (None, -1)。"""
        # QCNM 必须显式传入专用参数；缺参或向其他算法传入专用参数，由对应方法签名报 TypeError。
        # resources 的 0/1/2/3 表示不可用/空闲经典/占用经典/量子保留；芯轴第 0 项留空，芯号直接作为下标。
        if resources.ndim != 4 or resources.shape[2] != 8:
            raise ValueError('资源分配算法只支持固定七芯布局')
        if len(path) < 2:
            return None, -1
        if len(set(path)) != len(path):
            raise ValueError('分配要求路径为简单路径，节点不重复')
        if not np.isfinite(launch_power) or launch_power <= 0:
            raise ValueError('发射功率必须为有限正数')
        # methods 将算法名称映射到对应方法，例如 methods['FF'] 为 self._ff。
        methods = {
            'CQLI': self._cqli, 'CCA': self._cca, 'FF': self._ff,
            'SCWA': self._scwa, 'QCNM': self._qcnm,
        }
        # chosen_groups/wave 为逐跳所选芯组及原信道索引，例如 ([(2,), (4,)], 3)。
        # 专用参数原样转交对应算法，公共分配器不解释其含义。
        chosen_groups, wave = methods[self.algorithm](
            path, launch_power, resources, powers, distances, **algorithm_options)
        if chosen_groups is None:
            return None, -1
        return [list(group) for group in chosen_groups], wave

    # 以下五个方法共用 allocate 的路径、功率、资源和距离输入；专用参数只在对应方法声明。
    # 算法缩写：CQLI 为 Classical and quantum signal layered interleaved（经典量子信号分层交错资源分配），
    # CCA 为 Conventional channel allocation（传统信道分配方案），
    # FF 为 first-fit（首次适配），SCWA 为 Synergistic core and wavelength allocation（协同纤芯波长分配方案）；
    # 芯分组全部取固定七芯表。
    def _cqli(self, path, launch_power, resources, powers, distances):
        """CQLI（经典量子信号分层交错资源分配）按低频优先搜索，同频按固定芯表顺序选择候选，例如先 (2,) 后 (4,)。"""
        return self._first_fit(path, resources, sort_groups=False)

    def _cca(self, path, launch_power, resources, powers, distances):
        """CCA（Conventional channel allocation，传统信道分配方案）按低频优先搜索，同频按固定芯表顺序选择候选。"""
        return self._first_fit(path, resources, sort_groups=False)

    def _ff(self, path, launch_power, resources, powers, distances):
        """FF（first-fit，首次适配）按低频、组内芯编号排序搜索，例如 (2,) 先于 (4,)。"""
        return self._first_fit(path, resources, sort_groups=True)

    def _scwa(self, path, launch_power, resources, powers, distances):
        """SCWA（协同纤芯波长分配方案）按固定七芯表和信道索引奇偶分配。

        本方向名义容量减去已占用数（即剩余可用位置）达到 10 时交换奇偶组，恰好等于 10 也交换；
        "不做回退"指奇偶交换本身不按比例化阈值或互补集合修正，而某个信道选不出整条路径的芯组时仍会尝试下一个信道。"""
        # odd_waves/even_waves 为原信道索引的奇偶列表，例如 11 个信道时为 [1,3,5,7,9] / [0,2,4,6,8,10]。
        odd_waves = tuple(range(1, resources.shape[-1], 2))
        even_waves = tuple(range(0, resources.shape[-1], 2))
        # hops 保存各跳的候选和允许奇偶组，例如 (0,1,((6,),(5,),(7,)),(6,),(5,7),False)。
        hops = []
        # a/b 为本跳起止节点，例如 0/1；direction 是固定表的方向键，例如 'forward'。
        for a, b in zip(path, path[1:]):
            direction = 'forward' if a < b else 'backward'
            # odd_cores/even_cores 为固定芯号，例如前向 (6,) / (5,7)，不根据候选列表首项推导。
            odd_cores = SEVEN_CORE_LAYOUTS['SCWA'][direction + '_odd']
            even_cores = SEVEN_CORE_LAYOUTS['SCWA'][direction + '_even']
            # groups 为本跳绑定候选，例如 ((6,), (5,), (7,))；core/wave 为芯号/信道索引，例如 6/1。
            groups = self.forward_groups if a < b else self.backward_groups
            # unavailable 为原奇偶分配中所有非空闲位置数，例如 14；状态 0、2、3 都计入。
            unavailable = sum(resources[a, b, core, wave] != 1 for core in odd_cores for wave in odd_waves)
            unavailable += sum(resources[a, b, core, wave] != 1 for core in even_cores for wave in even_waves)
            # capacity 为本方向名义容量，例如 16 信道时为 24；swapped 为是否交换，14 >= 24-10 时为 True。
            capacity = len(odd_cores) * len(odd_waves) + len(even_cores) * len(even_waves)
            swapped = unavailable >= capacity - 10
            hops.append((a, b, groups, odd_cores, even_cores, swapped))
        # 按原信道索引 0、1、2……搜索，不改用实际频率排序，这是 SCWA 与其余算法的区别。
        for wave in range(resources.shape[-1]):
            # chosen_groups 为已选的逐跳芯组，例如 [(6,), (1,)]。
            chosen_groups = []
            for a, b, groups, odd_cores, even_cores, swapped in hops:
                # allowed_cores 为本跳此信道允许的固定芯组，例如奇数信道未切换时为 (6,)。
                allowed_cores = odd_cores if (wave % 2 == 1) != swapped else even_cores
                # chosen 为第一个全组空闲且所有成员都在允许表中的候选，例如 (6,)，不存在为 None。
                chosen = next((group for group in groups
                               if all(core in allowed_cores for core in group)
                               and self._available(a, b, group, wave, resources)), None)
                if chosen is None:
                    break
                chosen_groups.append(chosen)
            if len(chosen_groups) == len(path) - 1:
                return chosen_groups, wave
        return None, -1

    def _qcnm(
            self, path, launch_power, resources, powers, distances,
            *,
            noise_rtol,      # 相对噪声容忍系数，有限非负且可大于1；例如 2 允许噪声增量高于最小值 200%。
            first_fiber, secondary_fiber,  # 最近/次近邻光纤，用于计算 Raman 与 FWM 功率（W）。
            first_neighbors, secondary_neighbors,  # 最近/次近邻芯编号表，如 {2: [1, 3, 7], ...}。
    ):
        """QCNM（Quantum channel noise mitigation，量子信道噪声抑制）在噪声增量容差内优先减少同向同频邻芯占用。"""
        if first_fiber is None or secondary_fiber is None:
            raise ValueError('QCNM 需要最近邻与次近邻光纤参数')
        if not np.isfinite(noise_rtol) or noise_rtol < 0:
            raise ValueError('QCNM 容忍系数必须为有限非负数')
        noise_rtol = float(noise_rtol)
        # options 保存各可贯通信道，例如 {'wave': 3, 'chosen': [...], 'per_hop': [[...]]}。
        options = []
        # wave 为原信道索引，例如 3；QCNM 最终按原索引破同分，不按实际频率重排。
        for wave in range(resources.shape[-1]):
            # per_hop 按路径顺序保存每跳候选评分，例如 [[{'cores': (2,), 'noise': 1e-12, 'neighbors': 0}]]。
            per_hop = []
            # a/b 为当前跳节点，例如 0/1；group 为一整个备选芯组，例如 (2, 3, 4)。
            for a, b in zip(path, path[1:]):
                # candidates 为本跳本频点可用组的评分列表；元素 candidate 为其中一个评分字典。
                candidates = [self._candidate(
                                  a, b, group, wave, launch_power, resources, powers, distances[a, b],
                                  first_fiber, secondary_fiber,
                                  first_neighbors, secondary_neighbors)
                              for group in (self.forward_groups if a < b else self.backward_groups)
                              if self._available(a, b, group, wave, resources)]
                if not candidates:
                    break
                per_hop.append(candidates)
            if len(per_hop) != len(path) - 1:
                continue
            # chosen 为逐跳最低噪声候选列表；hop 为一跳的候选列表，例如 per_hop[0]。
            chosen = [min(hop, key=lambda candidate: (
                candidate['noise'], candidate['neighbors'], candidate['cores'])) for hop in per_hop]
            options.append(dict(wave=wave, chosen=chosen, per_hop=per_hop))
        if not options:
            return None, -1

        # option 为一个可贯通信道的记录，例如 options[0]。
        # minimum/limit 为全路径最低增量及容许上限，例如 1e-12/1.1e-12 W；minimum=0 时无绝对容差。
        minimum = min(sum(candidate['noise'] for candidate in option['chosen']) for option in options)
        limit = minimum + noise_rtol * abs(minimum)
        # ranked 保存最终排序键和分配，例如 [((0, 1e-12, 3, ((2,),)), [(2,)], 3)]。
        ranked = []
        for option in options:
            # original 是此信道的逐跳最低噪声选择，例如 option['chosen']，用于超限回退。
            original = option['chosen']
            if sum(candidate['noise'] for candidate in original) > limit:
                continue
            wave = option['wave']
            chosen = []
            for hop in option['per_hop']:
                # local_min/local_limit 为当前跳的最低增量及容许上限，例如 5e-13/5.5e-13 W。
                local_min = min(candidate['noise'] for candidate in hop)
                local_limit = local_min + noise_rtol * abs(local_min)
                # near 为局部容差内候选，例如 [hop[0], hop[2]]，优先选邻芯占用更少的组。
                near = [candidate for candidate in hop if candidate['noise'] <= local_limit]
                chosen.append(min(near, key=lambda candidate: (
                    candidate['neighbors'], candidate['noise'], candidate['cores'])))
            # 局部贪心若超出全路径上限则回退；不穷举组组合，也不约束整体秘密密钥率损失。
            if sum(candidate['noise'] for candidate in chosen) > limit:
                chosen = original
            # neighbor_count 为各跳邻芯占用总数，例如 2；chosen_groups 为逐跳芯组，例如 [(2,), (1,)]。
            neighbor_count = sum(candidate['neighbors'] for candidate in chosen)
            chosen_groups = [candidate['cores'] for candidate in chosen]
            # key 按邻芯数、噪声、原信道索引、芯编号排序，例如 (2, 1e-12, 3, ((2,), (1,)))。
            key = (neighbor_count, sum(candidate['noise'] for candidate in chosen), wave, tuple(chosen_groups))
            ranked.append((key, chosen_groups, wave))
        # item 是 ranked 中的一条记录，例如 (key, chosen_groups, wave)；丢弃返回记录中的排序键。
        _, chosen_groups, wave = min(ranked, key=lambda item: item[0])
        return chosen_groups, wave

    def _frequency_order(self, resources):
        """resources 为状态数组（同 allocate）；返回按实际频率升序排列的原索引，例如 [3, 2, 1, 0]。"""
        # wave 为原信道索引，例如 3；同频时按该索引升序破同分。
        return sorted(range(resources.shape[-1]), key=lambda wave: (self.frequencies[wave], wave))

    def _first_fit(self, path, resources, *, sort_groups):
        """按低频优先寻找全路径首个可用共同信道，返回逐跳芯组；失败返回 (None, -1)。"""
        # path/resources 同 allocate；sort_groups=True 按芯编号排序候选组。
        # wave 为原信道索引，例如 3；chosen_groups 为当前已选芯组，例如 [(2,), (4,)]。
        for wave in self._frequency_order(resources):
            chosen_groups = []
            # a/b 为当前跳起止节点，例如 0/1；groups 为该方向候选，例如 ((2,), (4,))。
            for a, b in zip(path, path[1:]):
                groups = self.forward_groups if a < b else self.backward_groups
                if sort_groups:
                    # group 是一组芯，例如 (4, 2)；按组内排序后的编号 (2, 4) 比较，输出保留原成员顺序。
                    groups = sorted(groups, key=lambda group: tuple(sorted(group)))
                # chosen 是首个全组可用的候选，例如 (2,)，不存在时为 None。
                chosen = next((group for group in groups
                               if self._available(a, b, group, wave, resources)), None)
                if chosen is None:
                    break
                chosen_groups.append(chosen)
            if len(chosen_groups) == len(path) - 1:
                return chosen_groups, wave
        return None, -1

    def _available(self, a, b, group, wave, resources):
        """判断本跳芯组在指定信道是否全部空闲，并检查反向同芯同频占用限制。"""
        # a/b 如 0/1，group 如 (2,4)，wave 为信道索引；resources 同 allocate。
        # core 为组内芯编号，例如 2；需要组内所有芯空闲，并按通用开关决定是否允许反向同芯同频同时占用。
        return all(resources[a, b, core, wave] == 1
                   and (self.allow_bidirectional or resources[b, a, core, wave] != 2)
                   for core in group)

    def _candidate(self, a, b, group, wave, launch_power, resources, powers, distance,
                   first_fiber, secondary_fiber, first_neighbors, secondary_neighbors):
        """计算候选芯组新增量子噪声与同向同频邻芯占用数，返回评分字典。"""
        # 输入同 allocate/_available；distance 单位 m，模型与邻芯表同 _qcnm。
        # deltas 为组内各芯新增噪声（W），例如 [1e-12, 2e-12, 0]；core 是成员编号，例如 2。
        deltas = [self._core_noise(
            a, b, core, wave, launch_power, resources, powers, distance,
            first_fiber, secondary_fiber, first_neighbors, secondary_neighbors)
            for core in group]
        # neighbors 是同向同频最近邻占用总数，例如 2；同一邻芯若邻接多个成员，会分别计数。
        neighbors = sum(
            self._same_direction_count(a, b, core, wave, resources, first_neighbors)
            for core in group)
        # 物理模型按经典芯相加；每芯均加载 launch_power，而不是用某一芯噪声乘组大小。
        return dict(cores=group, noise=sum(deltas), neighbors=neighbors)

    def _core_noise(self, a, b, core, wave, launch_power, resources, powers, distance,
                    first_fiber, secondary_fiber, first_neighbors, secondary_neighbors):
        """a/b/core/wave 是节点、单芯和信道编号（如 0/1/2/3）；其余参数同 _candidate，返回单芯新增量子噪声（W）。"""
        # i/j 为从小到大排序的链路端点，例如反向 a/b=1/0 时 i/j=0/1，用于读取量子信道。
        i, j = sorted((a, b))
        # active 为该芯已占用信道的功率向量，例如 [0, 0.01, 0] W；trial 为加入业务后的功率副本。
        active = np.where(resources[a, b, core] == 2, powers[a, b, core], 0.0)
        trial = active.copy()
        trial[wave] = launch_power
        # total_delta 汇总全部量子芯和频点的噪声增量，例如 1e-12 W；没有量子频点时为零。
        total_delta = 0.0
        # qc 是量子芯编号，例如 1；qi 是该芯量子信道索引元组，例如 (0,)，index 为其中的原索引，如 0。
        for qc in np.flatnonzero(np.any(resources[i, j] == 3, axis=1)):
            qi = tuple(int(index) for index in np.flatnonzero(resources[i, j, qc] == 3))
            # 这里传入的 backward 表示经典光相对量子光的传播方向，与上方 forward_groups 的 a<b 判定不是同一个量：
            # 业务为前向（a<b）时经典光与量子信号同向，取 False；业务为后向（a>b）时两者反向，取 True。
            # before_r/before_f 为新增前各量子频点的拉曼/四波混频功率数组，如 [1e-12]/[2e-13] W。
            before_r, before_f = calculate_quantum_noise_components(
                qc, core, a > b, active, qi, distance, self.frequencies,
                first_neighbors, secondary_neighbors, first_fiber, secondary_fiber)
            # after_r/after_f 为新增后对应数组，如 [1.2e-12]/[3e-13] W，方向参数同上。
            after_r, after_f = calculate_quantum_noise_components(
                qc, core, a > b, trial, qi, distance, self.frequencies,
                first_neighbors, secondary_neighbors, first_fiber, secondary_fiber)
            total_delta += float(np.sum(after_r - before_r) + np.sum(after_f - before_f))
        return total_delta

    def _same_direction_count(self, a, b, core, wave, resources, first_neighbors):
        """a/b/core/wave 是节点、芯和信道编号（如 0/1/2/3），resources 同 allocate，模型与邻芯表同 _qcnm；返回邻芯占用数，例如 2。"""
        # first_neighbors[core] 是最近邻芯编号列表，例如 [1,3,7]；neighbor 为其中一个编号，例如 1。
        return sum(int(resources[a, b, neighbor, wave] == 2)
                   for neighbor in first_neighbors[core])
