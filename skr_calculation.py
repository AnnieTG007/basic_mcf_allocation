"""供 main、traffic_scan 和 algorithm 导入调用，将资源状态及噪声功率换算为 BB84 协议的秘密密钥率（SKR，bit/s）。
提供有限样本估算、链路汇总及候选噪声评分，不读取文件或分配资源。"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np

from noise_calculation import calculate_noise_core, noise_power_to_counts


@dataclass(frozen=True)
class DetectorParameters:
    """保存探测器参数，并检查效率、门宽、损耗和发射速率的有效范围。"""
    efficiency: float  # 探测效率比例，例如 0.1。
    gate_time: float  # 单次探测门宽，s，例如 1e-9。
    insertion_loss_db: float  # 接收端插入损耗，dB，例如 1.0。
    rate_hz: float  # 每秒发射脉冲/探测门数，Hz，例如 1e9。

    def __post_init__(self):
        """拒绝非有限数及超出物理范围的探测参数。"""
        if (not all(math.isfinite(v) for v in (self.efficiency, self.gate_time, self.insertion_loss_db, self.rate_hz))
                or not 0 < self.efficiency <= 1 or self.gate_time <= 0
                or self.insertion_loss_db < 0 or self.rate_hz <= 0):
            raise ValueError("Invalid detector parameters")


@dataclass(frozen=True)
class BB84Parameters:
    """保存 BB84 双诱骗态估算的物理参数与有限样本参数。"""
    loss_per_m: float  # 指数衰减系数，m^-1，例如 4.6e-5。
    dark_count: float  # 每个探测器每门的暗计数概率，例如 1e-6。
    error_opt: float  # 光学误码比例，例如 0.015。
    sifting_efficiency: float  # 均匀选基后的保留比例，固定为 0.5。
    correct_error_eff: float  # 纠错开销相对理论极限的倍数，例如 1.16。
    signal_intensity: float = 0.6  # 信号态平均光子数。
    decoy_intensity: float = 0.2  # 弱诱骗态平均光子数；真空态为 0。
    signal_probability: float = 14 / 16  # 信号态发送概率。
    decoy_probability: float = 1 / 16  # 弱诱骗态概率；剩余概率发送真空态。
    pulse_count: float = 1e10  # 每个量子信道所有强度态的总脉冲数，不是筛选后密钥长度。
    fluctuation_gamma: float = 5.3  # 高斯波动的标准差倍数，不等同于可组合安全参数。

    def __post_init__(self):
        """检查强度、概率和样本数是否满足双诱骗态公式的前提。"""
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


# 随结果导出的模型口径；每次采样假设资源在整个脉冲块内静止，块时长与仿真时隙无换算关系。
# skr_definition 描述下游最终均值；本模块的链路汇总仅返回单次状态的信道总和。
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
    """将 BB84Parameters 和 DetectorParameters 实例转成可导出的模型配置。
    块时长为总脉冲数除以发射速率，例如 1e10 / 1e9 = 10 s。"""
    return dict(**SKR_DEFINITIONS, bb84=asdict(params), detector=asdict(detector),
                vacuum_probability=1-params.signal_probability-params.decoy_probability,
                block_duration_s=params.pulse_count/detector.rate_hz)


def H2(x):
    """计算概率 x（例如 0.01）的二元熵，端点 0 和 1 返回零。
    调用方保证 0 <= x <= 1。"""
    if x == 0 or x == 1:
        return 0.0
    return -x * math.log2(x) - (1-x) * math.log2(1-x)


def BB84_SKR(distance, noise, params, detector, *, clip=True):
    """按用户提供的 SKR_new.py 中 BB84_SKR_finite 估算 (SKR bit/s, 信号态量子比特误码率 QBER)。
    默认密钥率截零；clip=False 仅供诊断有效界下的负差值，界失效时仍返回零密钥率。"""
    # distance 为长度 m（如 1000）；noise 为每门探测后噪声计数（如 1e-6），不含暗计数。
    # params、detector 分别为上述 BB84 和探测参数；noise 已计效率及插损，按低计数概率近似使用。
    if not math.isfinite(distance) or distance < 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError("Distance and noise must be finite and nonnegative")
    # eta 为信号光子的总探测效率；不对传入的 noise 重复施加损耗。
    eta = detector.efficiency * math.exp(-params.loss_per_m * distance) * 10 ** (-0.1 * detector.insertion_loss_db)
    background = params.dark_count + noise
    if background >= 1:
        return 0.0, 0.5
    # background 是单探测器背景概率；y0 是两个探测器至少一个背景点击的概率。
    y0 = 1 - (1-background) ** 2
    u, v = params.signal_intensity, params.decoy_intensity
    signal_clicks = -math.expm1(-eta*u)
    decoy_clicks = -math.expm1(-eta*v)
    # u/v 为信号态/弱诱骗态强度，qu/qv 为各自总点击概率；超出概率域时不产密钥。
    qu, qv = y0 + signal_clicks, y0 + decoy_clicks
    if not (0 < qu <= 1 and 0 < qv <= 1):
        return 0.0, 0.5
    # 背景点击按一半出错：eu 为信号态 QBER，evqv 为弱诱骗态错误点击概率。
    eu = (0.5*y0 + params.error_opt*signal_clicks) / qu
    evqv = 0.5*y0 + params.error_opt*decoy_clicks
    # 用筛选后的弱诱骗态样本数求点击概率下界和错误点击概率上界。
    # 仅作高斯有限样本估算，未补信号/真空置信界、相位误码抽样界或可组合安全扣除项。
    n_decoy_basis = params.decoy_probability * params.pulse_count / 2
    qv_lower = max(qv - params.fluctuation_gamma * math.sqrt(qv/n_decoy_basis), 0.0)
    evqv_upper = evqv + params.fluctuation_gamma * math.sqrt(evqv/n_decoy_basis)
    # y1_lower 为单光子产额下界，e1_upper 为其误码上界；不可用的界返回零密钥率。
    y1_lower = u / (u*v - v*v) * (
        qv_lower*math.exp(v) - (v*v/(u*u))*qu*math.exp(u)
        - ((u*u-v*v)/(u*u))*y0)
    if y1_lower <= 0:
        return 0.0, eu
    e1_upper = (evqv_upper*math.exp(v) - 0.5*y0) / (v*y1_lower)
    if not 0 <= e1_upper < 0.5:
        return 0.0, eu
    # q1_lower 为信号态单光子点击概率下界；隐私放大收益扣除纠错开销后得到 bit/pulse。
    q1_lower = y1_lower*u*math.exp(-u)
    per_pulse = params.signal_probability * params.sifting_efficiency * (
        -qu*params.correct_error_eff*H2(eu) + q1_lower*(1-H2(e1_upper)))
    # 发射速率只乘一次，将每脉冲密钥量换成 bit/s。
    return (max(0.0, per_pulse) if clip else per_pulse)*detector.rate_hz, eu


def synergy_skr_bounds(distance, classical_core_count, launch_power, quantum_frequencies,
                       params, detector, fiber):
    """返回协同度归一化所用的 SKR 上下限（bit/s）及串扰条件。
    标尺是假设计算，不向实际量子噪声模型加入串扰；无经典芯或量子频率时抛错。"""
    # distance/launch_power 为链路长度 m / 每源功率 W（如 1000 / 0.01），params、detector 同 BB84_SKR。
    # classical_core_count 按双向经典芯并集计数（如 6）；quantum_frequencies 按量子芯/信道列频率 Hz（如 [193.1e12]）。
    # fiber 为最近邻光纤实例，复用芯间串扰（ICXT）参数 hmn，默认 1e-9 m^-1。
    source_count = classical_core_count - 1
    if source_count < 0 or len(quantum_frequencies) == 0:
        raise ValueError('SKR bounds require classical cores and quantum channels')
    # reference_xt_power 是单源正向串扰功率 W，也供经典侧标尺使用；下限叠加经典芯数减一份。
    reference_xt_power = fiber.get_forward_icxt_power(distance, launch_power)
    xt_power = source_count * reference_xt_power
    # counts 为逐量子信道每门噪声计数，不额外乘 1/2；逐信道截零后平均得到下限。
    counts = noise_power_to_counts(xt_power, quantum_frequencies, detector)
    lower = float(np.mean([BB84_SKR(distance, float(noise), params, detector)[0]
                           for noise in counts]))
    # 上限去掉外加噪声，仍保留暗计数和有限样本惩罚；单经典芯或零功率时上下限重合。
    upper = BB84_SKR(distance, 0.0, params, detector)[0]
    return dict(synergy_skr_lower=lower, synergy_skr_upper=upper,
                synergy_classical_core_count=classical_core_count,
                synergy_xt_source_count=source_count, synergy_xt_power_w=xt_power,
                synergy_reference_launch_power_w=launch_power,
                synergy_reference_xt_power_w=reference_xt_power)


def _validate_inputs(resource_map, powers, distances_m, frequencies_hz,
                     first_neighbors, secondary_neighbors):
    """检查网络输入的维度、物理量范围与邻芯编号，不合要求时抛出 ValueError。"""
    # resource_map/powers 为 [源节点, 目的节点, 芯, 信道] 数组，例如形状 (2, 2, 7, 16)。
    # 状态 0/1/2/3 为不可用/空闲经典/占用经典/量子保留；powers 单位 W，例如 0.01。
    # distances_m 为对称距离矩阵，单位 m（如 1000），缺失边为 inf；frequencies_hz 为信道频率列表（如 193.1e12 Hz）。
    # first_neighbors/secondary_neighbors 分别列出最近/次近邻芯，如 {0: [1, 2], ...}，必须覆盖所有芯。
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
    """返回指定链路和纤芯上逐量子信道截零后的 SKR 之和（bit/s），无量子信道时返回零。"""
    # i/j/c 为源节点、目的节点和量子芯编号（如 0/1/6）；网络数组沿用 _validate_inputs 的格式。
    # noise_model 为物理噪声模型实例，detector_params/bb84_params 为本模块的两类参数实例。
    indices = np.flatnonzero(resource_map[i, j, c] == 3)
    if not len(indices):
        return 0.0
    quantum_frequencies = np.asarray(frequencies_hz)[indices]
    # 只取状态 3 的量子频点，将双向经典光产生的噪声功率 W 换成每门计数。
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
    """返回各链路 SKR 总和的上三角矩阵（bit/s），缺失链路及其余位置为零。
    输入数组和模型参数沿用 calculate_SKR_core 的约定。"""
    _validate_inputs(resource_map, powers, distances_m, frequencies_hz,
                     first_neighbors, secondary_neighbors)
    node_count = resource_map.shape[0]
    result = np.zeros((node_count, node_count), dtype=float)
    # 每条物理链路只评估小节点到大节点的量子接收方向，避免双计；经典噪声仍含双向。
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
    """缓存候选经典芯产生的拉曼与四波混频（FWM）噪声，并汇总链路 SKR。
    由仿真实例的 quantum_scorer 持有，供分配与统计调用，不选择或占用资源。"""

    def __init__(self, frequencies, first_neighbors, secondary_neighbors,
                 noise_model, detector_params, bb84_params):
        """接收信道频率、两级邻芯表、噪声模型及探测/BB84 参数，约定同 calculate_SKR_core。
        frequencies 单位 Hz（如 [193.1e12]）；频率或物理模型改变后应重建评分器。"""
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.first = first_neighbors
        self.secondary = secondary_neighbors
        self.model = noise_model
        self.detector = detector_params
        self.bb84 = bb84_params
        self._cached = lru_cache(maxsize=32768)(self._component)

    def _component(self, rank, backward, powers, quantum_indices, distance):
        """计算一个邻芯对各量子频点的拉曼与 FWM 功率数组（W）。"""
        # rank=1/2 表示最近/次近邻；其余参数同 core_components。
        fiber = self.model.first_fiber if rank == 1 else self.model.secondary_fiber
        frequencies = self.frequencies[list(quantum_indices)]
        z = np.asarray([distance])  # 底层接口接收距离数组，此处只算一个距离。
        powers = np.asarray(powers)
        raman_fn = (fiber.get_inter_backward_raman_scatter if backward
                    else fiber.get_inter_forward_raman_scatter)
        fwm_fn = (fiber.get_backward_intercore_four_wave_mixing if backward
                  else fiber.get_intercore_four_wave_mixing)
        ram = fiber.get_raman_power(
            self.frequencies, powers, frequencies, raman_fn, z)
        fwm = fiber.get_fwm_power(
            self.frequencies, powers, frequencies, fwm_fn, z)[1]
        return ram[:, 0], fwm[:, 0]  # 取唯一距离列，保留量子频点顺序。

    def core_components(self, quantum_core, classical_core, backward, powers,
                        quantum_indices, distance):
        """返回指定经典芯对各量子频点的 (拉曼功率数组, FWM 功率数组)，单位 W。
        相同输入复用缓存，只计最近邻和次近邻耦合。"""
        # quantum_core/classical_core 为量子/经典芯编号（如 6/0）；backward=True 表示经典光反向传播。
        # powers 为各信道有效功率 W（如 [0.01, 0]），空闲信道置零；quantum_indices 为量子信道索引（如 [0, 2]）。
        # distance 为长度 m（如 1000）；rank=0 表示超出模型邻芯范围，返回零不代表器件无耦合。
        rank = (1 if classical_core in self.first[quantum_core] else
                2 if classical_core in self.secondary[quantum_core] else 0)
        if rank == 0:
            return np.zeros(len(quantum_indices)), np.zeros(len(quantum_indices))
        # 用不可变的数值元组作缓存键，避免数组不可哈希。
        return self._cached(rank, backward, tuple(float(p) for p in powers),
                            tuple(int(q) for q in quantum_indices), float(distance))

    def metrics(self, forward_resources, backward_resources,
                forward_powers, backward_powers, distance):
        """汇总一条链路当前状态的 SKR（bit/s）、噪声功率（W）、每门噪声计数及量子信道数量。
        SKR、功率和噪声计数均跨量子信道求和；正式 skr 逐信道截零，raw_skr 仅用于诊断。"""
        # 两方向 resources/powers 均为 [芯, 信道] 数组（如 (7, 16)），状态和 W 单位同 _validate_inputs。
        # forward 为小节点到大节点的量子接收方向，backward 为反向经典光；distance 单位 m（如 1000）。
        result = dict(skr=0.0, raw_skr=0.0, no_fwm_skr=0.0, zero_noise_skr=0.0,
                      raman_w=0.0, fwm_w=0.0, noise_counts=0.0,
                      quantum_channels=0, zero_skr_channels=0)
        # qc/qi 为量子芯/信道索引；ram/fwm 按量子频点累加两个传播方向的邻芯噪声功率。
        for qc in np.flatnonzero(np.any(forward_resources == 3, axis=1)):
            qi = np.flatnonzero(forward_resources[qc] == 3)
            ram, fwm = np.zeros(len(qi)), np.zeros(len(qi))
            for backward, resources, powers in (
                    (False, forward_resources, forward_powers),
                    (True, backward_resources, backward_powers)):
                for cc in self.first[qc] + self.secondary[qc]:
                    # cc 为经典芯编号，active 只保留状态 2 的占用信道功率。
                    active = np.where(resources[cc] == 2, powers[cc], 0.0)
                    r, f = self.core_components(qc, cc, backward, active, qi, distance)
                    ram += r
                    fwm += f
            # counts/rcounts 分别是总噪声/仅拉曼噪声的每门计数，效率及插损在此统一换算。
            counts = noise_power_to_counts(ram + fwm, self.frequencies[qi], self.detector)
            rcounts = noise_power_to_counts(ram, self.frequencies[qi], self.detector)
            for count, rcount in zip(counts, rcounts):
                # raw 保留有效界下的负差值，界失效仍为零；正式 SKR 禁止正负信道相互抵消。
                raw = BB84_SKR(distance, float(count), self.bb84, self.detector, clip=False)[0]
                result['quantum_channels'] += 1
                result['zero_skr_channels'] += int(raw <= 0)
                result['raw_skr'] += raw
                result['skr'] += max(0.0, raw)
                # 两项假设指标只去掉 FWM 或全部外加噪声，保留暗计数和有限样本模型，不代表其他算法。
                result['no_fwm_skr'] += max(0.0, BB84_SKR(
                    distance, float(rcount), self.bb84, self.detector)[0])
                result['zero_noise_skr'] += max(0.0, BB84_SKR(
                    distance, 0.0, self.bb84, self.detector)[0])
            result['raman_w'] += float(ram.sum())
            result['fwm_w'] += float(fwm.sum())
            result['noise_counts'] += float(counts.sum())
        return result
