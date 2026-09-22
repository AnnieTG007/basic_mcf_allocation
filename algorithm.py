"""为一条已选路径选择共同信道和逐跳纤芯，不直接占用或释放资源。

由 main.ClassicalService 通过 ResourceAllocator 调用。输入资源和功率数组
为 [源节点, 目的节点, 芯, 信道]，资源状态 1 表示可用，2 表示经典占用，
3 表示量子保留；功率 W、距离 m、实际频率 Hz。返回 (各跳芯列表, 信道索引)，
无可用分配时为 (None, -1)。全路径使用同一信道，中间节点可以换芯。

FF（first-fit，遇到可用资源就选）和 CQLI 共用搜索，但默认芯方向分组不同。
SCWA 比较经典芯内的四波混频（FWM）噪声增量；它沿用 Kong 2022 的贪心
思想做动态适配，不是完整静态复现，不调整频率网格或量子频率。现有材料
未给出完整论文条目，此处名称不能代替可核查的文献引用。
GREEDY_MIN_NOISE 优先避免新增 FWM 命中量子频率，再比较拉曼噪声；
“安全”只表示不新增命中组合，不保证原有 FWM 消失或 SKR（秘密密钥率）为正。
"""
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np


ALGORITHMS = ('CQLI', 'FF', 'SCWA', 'GREEDY_MIN_NOISE')


def normalize_algorithm(value):
    """统一大小写及连字符；first-fit/FF 统一为 FF，是否支持该名称由调用方检查。"""
    name = value.upper().replace('-', '_')
    return 'FF' if name == 'FIRST_FIT' else name


class GreedyMinNoise:
    """按新增频率碰撞和量子接收端噪声选择资源；评分器提供实际频率与噪声公式。"""
    def __init__(self, forward_cores, backward_cores, scorer):
        self.forward = tuple(forward_cores)
        self.backward = tuple(backward_cores)
        self.scorer = scorer
        self.frequencies = scorer.frequencies.copy()
        self.last_tier = None  # 最终接入等级：0 优先安全，1 其他安全，2 噪声回退。
        self._safe = lru_cache(maxsize=32768)(self._no_new_hit)

    def _no_new_hit(self, active, wave, quantum_indices):
        """检查新增 wave 是否参与会命中量子频点的同芯三频组合。
        
        active/wave/quantum_indices 均为信道索引；允许 i=j，排除 k=i 或 k=j
        的平凡组合，比较实际 fi+fj-fk 与 fq，差值小于 100 kHz 视为命中。
        只检查包含新增信道的组合，不重新拒绝原有噪声。跳过 C33 后不能用下标算频差。
        """
        trial = tuple(sorted((*active, wave)))
        targets = self.frequencies[list(quantum_indices)]
        for i, j in combinations_with_replacement(trial, 2):
            for k in trial:
                if k in (i, j) or wave not in (i, j, k):
                    continue
                generated = self.frequencies[i] + self.frequencies[j] - self.frequencies[k]
                if np.any(np.abs(targets - generated) < 1e5):
                    return False
        return True

    def preferred_parity(self, wave, quantum_indices):
        # 在 100 GHz 网格上，C35 为奇数，与它奇偶相反的是偶数泵浦。
        """是否处于 100 GHz 网格，且与所有量子频点的网格编号奇偶性相反。
        
        默认 C35 是奇数，优先选偶数经典信道；已有奇数经典信道时仍需逐组合检查安全性。
        """
        values = self.frequencies[[wave, *quantum_indices]] / 1e11
        grid = np.rint(values).astype(int)
        return bool(np.all(np.abs(values-grid) < 1e-6)
                    and np.all((grid[0]-grid[1:]) % 2 == 1))

    def _candidate(self, a, b, core, wave, launch_power, resources, powers, distance):
        """比较本跳本芯加入 wave 前后的量子端噪声，返回用于排序的数值。
        
        量子接收方向统一取小节点号到大节点号；raman/noise 为拉曼/总噪声增量 W，
        spectral 为拉曼谱系数的同分比较值，separation 为离最近量子频点的频差 Hz。
        只修改功率副本；噪声模型忽略的远芯可能为零增量，谱系数用于稳定区分这些候选。
        """
        i, j = sorted((a, b))
        active = np.where(resources[a, b, core] == 2, powers[a, b, core], 0.0)
        trial = active.copy()
        trial[wave] = launch_power
        quantum_cores = np.flatnonzero(np.any(resources[i, j] == 3, axis=1))
        safe, preferred = True, True
        raman_delta, total_delta, spectral_tie = 0.0, 0.0, 0.0
        separation = float('inf')
        for qc in quantum_cores:
            qi = tuple(int(q) for q in np.flatnonzero(resources[i, j, qc] == 3))
            safe &= self._safe(tuple(int(w) for w in np.flatnonzero(active)), wave, qi)
            preferred &= self.preferred_parity(wave, qi)
            separation = min(separation, float(np.min(np.abs(self.frequencies[wave]-self.frequencies[list(qi)]))))
            before_r, before_f = self.scorer.core_components(qc, core, a>b, active, qi, distance)
            after_r, after_f = self.scorer.core_components(qc, core, a>b, trial, qi, distance)
            raman_delta += float(np.sum(after_r-before_r))
            total_delta += float(np.sum(after_r-before_r) + np.sum(after_f-before_f))
            # 远芯被噪声模型忽略时，用拉曼谱系数稳定区分同分候选。
            fiber = self.scorer.model.first_fiber
            spectrum = self.scorer.model.raman
            for q in qi:
                spectral_tie += fiber.get_raman_eta(self.frequencies[wave], self.frequencies[q],
                    spectrum.coefficients, spectrum.index_center, spectrum.frequency_step_hz)
        return dict(core=core, safe=safe, preferred=preferred, raman=raman_delta,
                    noise=total_delta, spectral=spectral_tie, separation=separation)

    def allocate(self, path, launch_power, resources, powers, distances):
        """依次比较三个等级；只有所有跳都有空闲位置的共同信道才参与比较。
        
        对每个信道，各跳从安全芯中按拉曼增量、谱系数、芯编号升序选一个芯。
        所选各跳都满足优先奇偶条件为等级 0，其余安全分配为等级 1。等级相同时
        按全路径拉曼增量之和、谱系数之和、最小量子频差的降序、信道索引升序选择。
        若该信道任一跳没有安全芯，进入等级 2：各跳按总噪声增量、拉曼增量、芯编号
        选芯，跨信道按最小量子频差降序、总噪声之和、拉曼之和、信道索引升序选择。
        任何等级 0/1 都胜过等级 2；没有安全共同信道时仍可接入，不为避免 FWM 额外阻塞。
        last_tier 记录最终等级，失败为 None；它用于统计回退接入比例。
        """
        self.last_tier = None
        if len(path) < 2:
            return None, -1
        if len(set(path)) != len(path):
            raise ValueError('greedy_min_noise requires a simple path')
        if not np.isfinite(launch_power) or launch_power <= 0:
            raise ValueError('Launch power must be finite and positive')
        best_key, best = None, (None, -1)
        for wave in range(resources.shape[-1]):
            per_hop = []
            for a, b in zip(path, path[1:]):
                candidates = [self._candidate(a,b,c,wave,launch_power,resources,powers,distances[a,b])
                              for c in (self.forward if a<b else self.backward)
                              if resources[a,b,c,wave] == 1]
                if not candidates:
                    break
                per_hop.append(candidates)
            if len(per_hop) != len(path)-1:
                continue
            safe_path = all(any(c['safe'] for c in hop) for hop in per_hop)
            if safe_path:
                chosen = [min((c for c in hop if c['safe']),
                              key=lambda c: (c['raman'],c['spectral'],c['core'])) for hop in per_hop]
                tier = 0 if all(c['preferred'] for c in chosen) else 1
                key = (tier, sum(c['raman'] for c in chosen), sum(c['spectral'] for c in chosen),
                       -min(c['separation'] for c in chosen), wave)
            else:
                # 当前信道无法逐跳安全选芯，进入回退等级，优先比较离量子频率的距离。
                chosen = [min(hop,key=lambda c:(c['noise'],c['raman'],c['core'])) for hop in per_hop]
                key = (2, -min(c['separation'] for c in chosen), sum(c['noise'] for c in chosen),
                       sum(c['raman'] for c in chosen), wave)
            if best_key is None or key < best_key:
                best_key, best = key, ([c['core'] for c in chosen], wave)
        self.last_tier = None if best_key is None else best_key[0]
        return best


class ResourceAllocator:
    """构造时传入算法、方向纤芯和评分器；每次分配显式传入实时状态。

    不持有仿真对象或资源快照，不修改传入数组。
    """
    def __init__(self, algorithm, forward_cores, backward_cores, scorer, quantum_scorer=None):
        self.algorithm = normalize_algorithm(algorithm)
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if self.algorithm == "SCWA" and scorer is None:
            raise ValueError("SCWA requires a supplied FWM scorer")
        self.core_f = tuple(forward_cores)
        self.core_b = tuple(backward_cores)
        self.scorer = scorer
        if self.algorithm == "GREEDY_MIN_NOISE" and quantum_scorer is None:
            raise ValueError(f"{self.algorithm} requires a quantum receiver scorer")

        self.noise_policy = (GreedyMinNoise(self.core_f, self.core_b, quantum_scorer)
                             if self.algorithm == "GREEDY_MIN_NOISE" else None)

    def allocate(self, path, launch_power, *, resources, powers, distances):
        """返回 (各跳芯列表, 共同信道索引)，失败为 (None, -1)。
        
        launch_power/powers 为 W，distances 为 m；不修改输入数组，实际占用/释放由 main 处理。
        """
        if self.noise_policy is not None:
            return self.noise_policy.allocate(path, launch_power, resources, powers, distances)
        if self.algorithm == "SCWA":
            return self._scwa(path, launch_power, resources, powers, distances)
        return self._first_fit(path, resources)

    def _first_fit(self, path, resources):
        """FF/CQLI 按候选频率顺序、仿真芯编号升序搜索。

        两者复用 first-fit 搜索，以实例资源矩阵中的方向/量子芯布局区分。
        """
        if len(path) < 2:
            return None, -1
        for wave in range(resources.shape[-1]):
            cores = []
            for a, b in zip(path, path[1:]):
                free = np.flatnonzero(resources[a, b, :, wave] == 1)
                if not len(free):
                    break
                cores.append(int(free[0]))
            if len(cores) == len(path) - 1:
                return cores, wave
        return None, -1

    def _scwa(self, path, launch_power, resources, powers, distances):
        """最小化此次分配引入的经典芯内 FWM 增量之和（单位 W）。
    
        接收频点包含所有经典候选，不只包含占用频点。每一跳的未修改芯
        对所有候选相同，因此减去该芯分配前评分后求和，等价于比较整个
        路径分配后的总经典芯内 FWM。无芯间经典评分时，各跳可独立选芯。
        同分沿用波长倒序和传入芯顺序。路径优先级由调用方决定。
        只在副本上试分配，不改变已有业务、资源矩阵或实际发射功率。
        """
        core_f, core_b = self.core_f, self.core_b
        scorer = self.scorer
        if len(path) < 2:
            return None, -1
        if not np.isfinite(launch_power) or launch_power <= 0:
            raise ValueError("Launch power must be finite and positive")
        baselines = {}
        for a, b in zip(path, path[1:]):
            for core in (core_f if a < b else core_b):
                active = np.where(resources[a, b, core] == 2, powers[a, b, core], 0.0)
                baselines[a, b, core] = (active, scorer(active, distances[a, b]))
        best_cores, best_wave, best_score = None, -1, float("inf")
        for wave in range(resources.shape[-1] - 1, -1, -1):
            cores, path_score = [], 0.0
            for a, b in zip(path, path[1:]):
                selected, increment = None, float("inf")
                for core in (core_f if a < b else core_b):
                    if resources[a, b, core, wave] != 1:
                        continue
                    active, baseline = baselines[a, b, core]
                    trial = active.copy()
                    trial[wave] = launch_power
                    delta = scorer(trial, distances[a, b]) - baseline
                    if delta < increment:
                        selected, increment = core, delta
                if selected is None:
                    break
                cores.append(selected)
                path_score += increment
            if len(cores) == len(path) - 1 and path_score < best_score:
                best_cores, best_wave, best_score = cores, wave, path_score
        return best_cores, best_wave
