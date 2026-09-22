"""将量子信道的噪声功率换成探测计数，计算 BB84 秘密密钥率（SKR）。

由 main 的单次运行、traffic_scan 的统计和 algorithm 的候选评分调用，
不读取文件、不选择资源。距离 m、频率 Hz、功率 W；噪声计数是每探测门
的平均计数，SKR 为 bit/s，量子比特误码率 QBER 为 0..1 的比例。
资源/功率数组为 [源节点, 目的节点, 芯, 信道]，状态 0/1/2/3 为
不可用/空闲经典/占用经典/量子保留。

BB84_SKR 和 calculate_SKR_all 保留可为负的公式原值；扫描使用
QuantumLinkScorer.metrics，将每量子信道 SKR 各自截为非负再求和。
只评估节点号小到大的量子接收方向，经典噪声同时包括两个传播方向。
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from noise_calculation import calculate_noise_core, noise_power_to_counts


@dataclass(frozen=True)
class DetectorParameters:
    """探测参数：efficiency 是探测效率比例，gate_time 为门宽 s，
    insertion_loss_db 为接收端插入损耗 dB，rate_hz 为每秒发射/探测门数。
    """
    efficiency: float
    gate_time: float
    insertion_loss_db: float
    rate_hz: float

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.efficiency, self.gate_time, self.insertion_loss_db, self.rate_hz))
                or not 0 < self.efficiency <= 1 or self.gate_time <= 0
                or self.insertion_loss_db < 0 or self.rate_hz <= 0):
            raise ValueError("Invalid detector parameters")


@dataclass(frozen=True)
class BB84Parameters:
    """BB84 模型参数：loss_per_m 为指数衰减系数 m^-1（不是 dB/m），
    dark_count 为每门暗计数项，photon_launch 为每脉冲平均光子数，
    error_opt 为光学误码比例，sifting_efficiency 为基筛选保留比例，
    correct_error_eff 为相对理想纠错开销的倍率（至少为 1）。
    """
    loss_per_m: float
    dark_count: float
    photon_launch: float
    error_opt: float
    sifting_efficiency: float
    correct_error_eff: float

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
                or self.loss_per_m < 0 or self.dark_count < 0 or self.photon_launch <= 0
                or not 0 <= self.error_opt <= 1 or not 0 < self.sifting_efficiency <= 1
                or self.correct_error_eff < 1):
            raise ValueError("Invalid BB84 parameters")


def H2(x):
    """二元熵 -x*log2(x)-(1-x)*log2(1-x)，输入概率 x，端点 0 和 1 的熵均为零。"""
    if x == 0 or x == 1:
        return 0.0
    y = -x * math.log2(x) - (1 - x) * math.log2(1 - x)# 调用方传入 0<x<1 的概率
    return y
def BB84_SKR(distance, noise, params, detector):
    """返回 (原始秘密密钥率 bit/s, QBER 比例)，不将负密钥率截为零。
    
    distance 为 m；noise 为已经计入探测效率和插入损耗的每门噪声计数，
    不含暗计数。eta 为信号从发射到探测的总效率，Y0 为背景计数项，
    Y1 为单光子产额近似，Q1 为单光子增益，Q_ave 为平均检测增益；
    e1 和 e_ave 分别为单光子及总体误码率。最后乘 rate_hz 换成每秒比特数。
    这里保留已有渐近 BB84 公式，没有有限密钥长度修正。
    """
    if not math.isfinite(distance) or distance < 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError("Distance and noise must be finite and nonnegative")
    # noise 是单光子探测器探测后的噪声计数，已计入探测效率及插入损耗，不含暗计数。
    eta = detector.efficiency * math.exp(-params.loss_per_m * distance)* 10 ** (-0.1 * detector.insertion_loss_db)
    e0 = 1 / 2
    Y0 = params.dark_count + noise
    Y1 = Y0 + eta
    Q1 = Y1 * params.photon_launch * math.exp(-params.photon_launch)
    Q_ave = Y0 + 1 - math.exp(-eta * params.photon_launch)
    e1 = (e0 * Y0 + params.error_opt * eta) / Y1
    e_ave = (e0 * Y0 + params.error_opt * (1 - math.exp(-eta * params.photon_launch))) / Q_ave
    skr = params.sifting_efficiency * (-Q_ave * params.correct_error_eff * H2(e_ave) + Q1 * (1 - H2(e1)))
    qber = e_ave
    return skr*detector.rate_hz, qber


def _validate_inputs(resource_map, powers, distances_m, frequencies_hz,
                     first_neighbors, secondary_neighbors):
    """检查数组维度、资源状态、物理量取值和邻芯索引；缺失链路长度用 inf 表示。"""
    shape = resource_map.shape
    if len(shape) != 4 or shape[0] != shape[1]:
        raise ValueError("resource_map must have shape (nodes, nodes, cores, channels)")
    if powers.shape != shape or distances_m.shape != shape[:2]:
        raise ValueError("Power/distance shapes do not match resource_map")
    if np.shape(frequencies_hz) != (shape[-1],):
        raise ValueError("Frequency count does not match channels")
    if np.any(~np.isfinite(frequencies_hz)) or np.any(np.asarray(frequencies_hz) <= 0):
        raise ValueError("Frequencies must be finite and positive")
    if np.any(~np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("Powers must be finite and nonnegative")
    if not np.all(np.isin(resource_map, (0, 1, 2, 3))):
        raise ValueError("Unknown resource state")
    if np.any(np.isnan(distances_m)) or np.any(distances_m < 0):
        raise ValueError("Distances must be nonnegative, using inf for absent edges")
    if not np.array_equal(distances_m, distances_m.T):
        raise ValueError("Physical link distances must be symmetric")
    for neighbors in (first_neighbors, secondary_neighbors):
        if set(neighbors) != set(range(shape[2])):
            raise ValueError("Neighbor maps must cover every core")
        for core, adjacent in neighbors.items():
            if any(not isinstance(n, (int, np.integer)) or n < 0 or n >= shape[2] or n == core
                   for n in adjacent):
                raise ValueError("Invalid neighbor core index")


def calculate_SKR_core(i, j, c, *, resource_map, powers, distances_m,
                       frequencies_hz, first_neighbors, secondary_neighbors,
                       noise_model, detector_params, bb84_params):
    """指定链路/纤芯上所有量子信道的 SKR 之和，单位 bit/s。"""
    indices = np.flatnonzero(resource_map[i, j, c] == 3)
    if not len(indices):
        return 0.0
    quantum_frequencies = np.asarray(frequencies_hz)[indices]
    noise = calculate_noise_core(
        i, j, c, resource_map, powers, distances_m, frequencies_hz,
        first_neighbors, secondary_neighbors, noise_model,
    )
    counts = noise_power_to_counts(noise, quantum_frequencies, detector_params)
    return sum(BB84_SKR(distances_m[i, j], count, bb84_params, detector_params)[0]
               for count in counts)


def calculate_SKR_all(*, resource_map, powers, distances_m, frequencies_hz,
                      first_neighbors, secondary_neighbors, noise_model,
                      detector_params, bb84_params):
    """返回链路 SKR 上三角矩阵，双向物理链路只统计一次。"""
    _validate_inputs(resource_map, powers, distances_m, frequencies_hz,
                     first_neighbors, secondary_neighbors)
    node_count = resource_map.shape[0]
    result = np.zeros((node_count, node_count), dtype=float)
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if not np.isfinite(distances_m[i, j]):
                continue
            cores = np.flatnonzero(np.any(resource_map[i, j] == 3, axis=1))
            for c in cores:
                result[i, j] += calculate_SKR_core(
                    i, j, c, resource_map=resource_map, powers=powers,
                    distances_m=distances_m, frequencies_hz=frequencies_hz,
                    first_neighbors=first_neighbors, secondary_neighbors=secondary_neighbors,
                    noise_model=noise_model, detector_params=detector_params,
                    bb84_params=bb84_params,
                )
    return result


class QuantumLinkScorer:
    """量子接收端的物理评估器，不负责选择或占用资源。

    core_components() 为候选分配提供分芯的拉曼/FWM 噪声，并缓存相同输入；
    metrics() 汇总整条链路的噪声、原始/非负 SKR 和零 SKR 信道数。
    仿真实例通过 quantum_scorer 持有它，分配策略和统计模块共同复用。
    """

    def __init__(self, frequencies, first_neighbors, secondary_neighbors,
                 noise_model, detector_params, bb84_params):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.first = first_neighbors
        self.secondary = secondary_neighbors
        self.model = noise_model
        self.detector = detector_params
        self.bb84 = bb84_params
        self._cached = lru_cache(maxsize=32768)(self._component)

    def _component(self, rank, backward, powers, quantum_indices, distance):
        """计算一个邻芯的拉曼与 FWM（四波混频）功率数组 W；rank=1/2 指最近/次近邻芯。"""
        fiber = self.model.first_fiber if rank == 1 else self.model.secondary_fiber
        raman = self.model.raman
        frequencies = self.frequencies[list(quantum_indices)]
        z = np.asarray([distance])
        powers = np.asarray(powers)
        raman_fn = (fiber.get_inter_backward_raman_scatter if backward
                    else fiber.get_inter_forward_raman_scatter)
        fwm_fn = (fiber.get_backward_intercore_four_wave_mixing if backward
                  else fiber.get_intercore_four_wave_mixing)
        ram = fiber.get_raman_power_all2(
            self.frequencies, powers, frequencies, raman_fn, z,
            np.asarray(raman.coefficients), raman.index_center, raman.frequency_step_hz)
        fwm = fiber.get_fwm_power_all3(
            self.frequencies, powers, frequencies, fwm_fn, z)[1]
        return ram[:, 0], fwm[:, 0]

    def core_components(self, quantum_core, classical_core, backward, powers,
                        quantum_indices, distance):
        """返回指定经典芯对各量子频点的 (拉曼功率数组, FWM 功率数组)，单位 W。
        
        powers 为该芯所有信道的有效发射功率，空闲信道应置零；backward 表示
        经典光反向传播。只计最近及次近邻，远芯返回零是模型截断，不是器件无耦合。
        相同输入复用缓存；更改频率或物理模型时应重新创建评分器。
        """
        rank = (1 if classical_core in self.first[quantum_core] else
                2 if classical_core in self.secondary[quantum_core] else 0)
        if rank == 0:
            return np.zeros(len(quantum_indices)), np.zeros(len(quantum_indices))
        return self._cached(rank, backward, tuple(float(p) for p in powers),
                            tuple(int(q) for q in quantum_indices), float(distance))

    def metrics(self, forward_resources, backward_resources,
                forward_powers, backward_powers, distance):
        """评估一条无向链路，输入两方向的 [芯, 信道] 资源和功率，以及长度 m。
        
        forward 对应小节点到大节点的量子接收方向。返回 skr（逐信道非负后求和）、
        raw_skr（公式原值之和）、拉曼/FWM 功率 W、每门噪声计数之和及量子信道计数。
        no_fwm_skr 仅在同一资源状态中去掉 FWM；zero_noise_skr 去掉外加噪声但保留
        暗计数，二者都是 bit/s 的假设计算，不是其他算法实际分配的结果。
        """
        result = dict(skr=0.0, raw_skr=0.0, no_fwm_skr=0.0, zero_noise_skr=0.0,
                      raman_w=0.0, fwm_w=0.0, noise_counts=0.0,
                      quantum_channels=0, zero_skr_channels=0)
        for qc in np.flatnonzero(np.any(forward_resources == 3, axis=1)):
            qi = np.flatnonzero(forward_resources[qc] == 3)
            ram, fwm = np.zeros(len(qi)), np.zeros(len(qi))
            for backward, resources, powers in (
                    (False, forward_resources, forward_powers),
                    (True, backward_resources, backward_powers)):
                for cc in self.first[qc] + self.secondary[qc]:
                    active = np.where(resources[cc] == 2, powers[cc], 0.0)
                    r, f = self.core_components(qc, cc, backward, active, qi, distance)
                    ram += r
                    fwm += f
            counts = noise_power_to_counts(ram + fwm, self.frequencies[qi], self.detector)
            rcounts = noise_power_to_counts(ram, self.frequencies[qi], self.detector)
            for count, rcount in zip(counts, rcounts):
                raw = BB84_SKR(distance, float(count), self.bb84, self.detector)[0]
                result['quantum_channels'] += 1
                result['zero_skr_channels'] += int(raw <= 0)
                result['raw_skr'] += raw
                result['skr'] += max(0.0, raw)
                result['no_fwm_skr'] += max(0.0, BB84_SKR(
                    distance, float(rcount), self.bb84, self.detector)[0])
                result['zero_noise_skr'] += max(0.0, BB84_SKR(
                    distance, 0.0, self.bb84, self.detector)[0])
            result['raman_w'] += float(ram.sum())
            result['fwm_w'] += float(fwm.sum())
            result['noise_counts'] += float(counts.sum())
        return result
