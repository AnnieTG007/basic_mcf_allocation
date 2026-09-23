"""将量子信道的噪声功率换成探测计数，计算 BB84 秘密密钥率（SKR）。

由 main 的单次运行、traffic_scan 的统计和 algorithm 的候选评分调用，
不读取文件、不选择资源。距离 m、频率 Hz、功率 W；噪声计数是每探测门
的平均计数，SKR 为 bit/s，量子比特误码率 QBER 为 0..1 的比例。
资源/功率数组为 [源节点, 目的节点, 芯, 信道]，状态 0/1/2/3 为
不可用/空闲经典/占用经典/量子保留。

正式 SKR 采用用户提供的 SKR_new.py 中 BB84_SKR_finite 双诱骗态估算，
逐量子信道截零后再求链路和时间平均；raw_skr 仅保留有效界下截零前的差值。
每个时隙末资源状态按固定脉冲块估算，不把变化的业务轨迹当作一个已采集密钥块。
只评估节点号小到大的量子接收方向，经典噪声同时包括两个传播方向。
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np

from noise_calculation import (calculate_noise_core, noise_power_to_counts,
                               forward_P_XT, XT_PARAMS)


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
    """双诱骗态 BB84 的物理和有限样本参数。

    loss_per_m 为指数衰减系数 m^-1；dark_count 是每个探测器每门暗计数概率。
    signal/decoy_intensity 为信号态/弱诱骗态的平均光子数；真空态强度为0，
    概率为1-signal_probability-decoy_probability。pulse_count 是每个量子信道
    所有强度态的合计发射脉冲数，不是筛选后密钥长度，也不是跨信道共享预算。
    fluctuation_gamma 为标准差倍数，
    不直接等于可组合安全参数。沿用参考的均匀选基，筛选效率固定为1/2。
    """
    loss_per_m: float
    dark_count: float
    error_opt: float
    sifting_efficiency: float
    correct_error_eff: float
    signal_intensity: float = 0.6
    decoy_intensity: float = 0.2
    signal_probability: float = 14 / 16
    decoy_probability: float = 1 / 16
    pulse_count: float = 1e10
    fluctuation_gamma: float = 5.3

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
                or self.loss_per_m < 0 or not 0 <= self.dark_count < 1
                or not 0 <= self.error_opt <= 0.5 or self.sifting_efficiency != 0.5
                or self.correct_error_eff < 1
                or not 0 < self.decoy_intensity < self.signal_intensity < 1
                or not 0 < self.signal_probability < 1 or not 0 < self.decoy_probability < 1
                or self.signal_probability + self.decoy_probability >= 1
                or self.pulse_count < 1 or not float(self.pulse_count).is_integer()
                or self.fluctuation_gamma < 0):
            raise ValueError("Invalid finite-key BB84 parameters")


SKR_DEFINITIONS = dict(
    skr_model_version='two_decoy_gaussian_finite_v1',
    skr_reference='User-supplied SKR_new.py: BB84_SKR_finite; adapted with existing detector and loss parameters',
    skr_definition='Time mean of per-channel nonnegative finite-size two-decoy BB84 estimates on the selected link; bit/pulse multiplied by pulse rate once',
    raw_skr_definition='Pre-clipping finite-size expression when bounds are admissible; zero when bounds fail; diagnostic only, not usable key rate',
    skr_block_definition='Each sampled resource state is held stationary for the configured pulse block; block duration=N/rate_hz; independent of simulation slots and holding time; not key extraction from a time-varying collected block',
    skr_security_scope='Gaussian fluctuation estimate on weak-decoy gain/error counts; signal gain and vacuum yield use model expectations; no composable epsilon security claim or phase-error finite-size proof',
    skr_noise_definition='Existing detector-adjusted per-gate noise used as per-detector background probability, following supplied reference; Y0=1-(1-dark_count-noise)^2; no repeated efficiency/loss conversion',
)


def skr_model_config(params, detector):
    """返回可复现的模型配置；实际秒数仅指假设静态资源状态下的脉冲块。"""
    return dict(**SKR_DEFINITIONS, bb84=asdict(params), detector=asdict(detector),
                vacuum_probability=1-params.signal_probability-params.decoy_probability,
                block_duration_s=params.pulse_count/detector.rate_hz)


def H2(x):
    """二元熵；概率端点0和1的熵为零。调用方保证物理域0<=x<=1。"""
    if x == 0 or x == 1:
        return 0.0
    return -x * math.log2(x) - (1-x) * math.log2(1-x)


def BB84_SKR(distance, noise, params, detector, *, clip=True):
    """返回有限样本估算 (SKR bit/s, 信号态QBER)，默认密钥率非负。

    迁移 SKR_new.py 的 BB84_SKR_finite，不使用其中的 simple 近似分支。
    distance 为m；noise 沿用项目每门探测后计数的低计数概率近似，已计探测效率
    和插损、不含暗计数。按参考作为每个探测器背景概率输入两探测器Y0公式。
    探测效率和插损只在信号eta中另算，不对传入noise重复乘系数。

    Qv下界和EvQv上界用弱诱骗态筛选后样本数 p_decoy*N/2 的高斯波动估算，
    再求单光子产额下界和误码上界。参考未给信号态/真空态有限置信界、相位误码
    抽样界及可组合安全扣除项；这是有限样本性能估算，不是完整有限密钥安全证明。
    理论背景/点击概率超域、单光子下界非正或误码上界>=1/2时返回0。
    clip=False只供raw_skr诊断有效界下的负差值，界失效时仍返回0。
    参考函数输出bit/pulse；这里只乘一次rate_hz，供全项目统一使用bit/s。
    """
    if not math.isfinite(distance) or distance < 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError("Distance and noise must be finite and nonnegative")
    eta = detector.efficiency * math.exp(-params.loss_per_m * distance) * 10 ** (-0.1 * detector.insertion_loss_db)
    background = params.dark_count + noise
    if background >= 1:
        return 0.0, 0.5
    y0 = 1 - (1-background) ** 2
    u, v = params.signal_intensity, params.decoy_intensity
    signal_clicks = -math.expm1(-eta*u)
    decoy_clicks = -math.expm1(-eta*v)
    qu, qv = y0 + signal_clicks, y0 + decoy_clicks
    if not (0 < qu <= 1 and 0 < qv <= 1):
        return 0.0, 0.5
    eu = (0.5*y0 + params.error_opt*signal_clicks) / qu
    evqv = 0.5*y0 + params.error_opt*decoy_clicks
    n_decoy_basis = params.decoy_probability * params.pulse_count / 2
    qv_lower = max(qv - params.fluctuation_gamma * math.sqrt(qv/n_decoy_basis), 0.0)
    evqv_upper = evqv + params.fluctuation_gamma * math.sqrt(evqv/n_decoy_basis)
    y1_lower = u / (u*v - v*v) * (
        qv_lower*math.exp(v) - (v*v/(u*u))*qu*math.exp(u)
        - ((u*u-v*v)/(u*u))*y0)
    if y1_lower <= 0:
        return 0.0, eu
    e1_upper = (evqv_upper*math.exp(v) - 0.5*y0) / (v*y1_lower)
    if not 0 <= e1_upper < 0.5:
        return 0.0, eu
    q1_lower = y1_lower*u*math.exp(-u)
    per_pulse = params.signal_probability * params.sifting_efficiency * (
        -qu*params.correct_error_eff*H2(eu) + q1_lower*(1-H2(e1_upper)))
    return (max(0.0, per_pulse) if clip else per_pulse)*detector.rate_hz, eu


def synergy_skr_bounds(distance, classical_core_count, launch_power, quantum_frequencies,
                       params, detector):
    """返回协同度的 SKR 标尺及串扰条件；距离 m、功率 W、频率 Hz、SKR bit/s。

    上限为零外加噪声 SKR，仍保留暗计数和有限样本惩罚。下限假设每个量子信道
    受到 N_classical-1 个同功率正向串扰源，每源耦合系数固定为 1e-6 km^-1。
    N_classical 由调用方按前后向经典芯集合的并集计数，不重复计算双向共享芯。
    quantum_frequencies 按实际量子芯/信道逐项传入；逐项截零后求平均。
    串扰功率通过正式 noise_power_to_counts 转为探测后每门计数，不使用历史 SDM
    接口的额外 1/2 因子。这是假设归一化标尺，不向实际量子噪声模型加入串扰。
    经典芯不足1或量子频率列表为空时抛错；单经典芯或零功率会使上下限重合。
    """
    source_count = classical_core_count - 1
    if source_count < 0 or len(quantum_frequencies) == 0:
        raise ValueError('SKR bounds require classical cores and quantum channels')
    xt_power = source_count * forward_P_XT(1e-6, distance, launch_power, XT_PARAMS)
    counts = noise_power_to_counts(xt_power, quantum_frequencies, detector)
    lower = float(np.mean([BB84_SKR(distance, float(noise), params, detector)[0]
                           for noise in counts]))
    upper = BB84_SKR(distance, 0.0, params, detector)[0]
    return dict(synergy_skr_lower=lower, synergy_skr_upper=upper,
                synergy_classical_core_count=classical_core_count,
                synergy_xt_source_count=source_count, synergy_xt_power_w=xt_power,
                synergy_reference_launch_power_w=launch_power)


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
    metrics() 汇总整条链路的噪声、有限样本截零前/非负 SKR 和零 SKR 信道数。
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
        raw_skr（有效有限样本界下截零前差值之和，界失效记0）、拉曼/FWM 功率 W、每门噪声计数之和及量子信道计数。
        no_fwm_skr 仅在同一资源状态中去掉 FWM；zero_noise_skr 去掉外加噪声但保留
        暗计数；这两项与正式skr均使用同一有限样本模型并逐信道截零。
        二者都是 bit/s 的假设计算，不是其他算法实际分配的结果。
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
                raw = BB84_SKR(distance, float(count), self.bb84, self.detector, clip=False)[0]
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
