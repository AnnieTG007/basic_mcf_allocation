"""供 main、traffic_scan 和 algorithm 导入调用，将资源状态及噪声功率换算为 BB84 协议的秘密密钥率（SKR，bit/s）。
提供有限样本估算、链路汇总及候选噪声评分，不读取文件或分配资源。"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np

from noise_calculation import calculate_intercore_noise_components, noise_power_to_counts


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


# 当前实现为信号、弱诱骗、真空三种强度的诱骗态协议；本模型为高斯统计波动近似，不给出可组合安全性证明。
SKR_MODEL_VERSION = 'three_intensity_decoy_gaussian_finite'
# 随结果导出的说明：仅保留模型版本一项，指标含义与数据格式写在代码注释里，不写进结果文件。
SKR_MODEL = dict(model=SKR_MODEL_VERSION)


def skr_model_config(params, detector):
    """将 BB84Parameters 和 DetectorParameters 实例转成可导出的模型配置。
    块时长为总脉冲数除以发射速率，例如 1e10 / 1e9 = 10 s。"""
    return dict(model=SKR_MODEL_VERSION, bb84=asdict(params), detector=asdict(detector),
                block_duration_s=params.pulse_count/detector.rate_hz)


def H2(x):
    """计算概率 x（例如 0.01）的二元熵，端点 0 和 1 返回零。
    调用方保证 0 <= x <= 1。"""
    if x == 0 or x == 1:
        return 0.0
    return -x * math.log2(x) - (1-x) * math.log2(1-x)


def BB84_SKR(distance, noise, params, detector, *, clip=True):
    """按信号、弱诱骗、真空三种强度的诱骗态估算 (SKR bit/s, 信号态量子比特误码率 QBER)。
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


@lru_cache(maxsize=32768)
def _cached_quantum_noise(fiber, frequencies, powers, quantum_indices, distance, backward):
    """缓存单个经典芯的 Raman（拉曼）和 FWM 功率；完整光纤参数与频率进入缓存键。
    最多保留 32768 项，配置不同不会误用旧结果；返回数组只供读取。"""
    # fiber 为不可变 MulticoreFiber；频率 Hz、功率 W、目标索引使用元组，距离 m。
    frequencies = np.asarray(frequencies, dtype=float)
    raman, fwm = calculate_intercore_noise_components(
        fiber, frequencies, np.asarray(powers), frequencies[list(quantum_indices)],
        np.asarray([distance]), backward=backward)
    return raman[:, 0], fwm[:, 0]


def calculate_quantum_noise_components(quantum_core, classical_core, backward, powers,
                                       quantum_indices, distance, frequencies,
                                       first_neighbors, secondary_neighbors,
                                       first_fiber, secondary_fiber):
    """返回一个经典芯贡献的 (Raman 功率数组, FWM 功率数组)，单位 W。
    只计两级邻芯；缓存数组不可原地修改，范围外返回零仅表示模型未计入。"""
    # 两芯编号如 1/2；backward=True 为经典光相对量子光反向；powers 为有效功率谱 W。
    # quantum_indices 为目标信道索引，distance 为 m，frequencies 为 Hz；邻芯表和模型同 calculate_SKR_core。
    if classical_core in first_neighbors[quantum_core]:
        fiber = first_fiber
    elif classical_core in secondary_neighbors[quantum_core]:
        fiber = secondary_fiber
    else:
        return np.zeros(len(quantum_indices)), np.zeros(len(quantum_indices))
    return _cached_quantum_noise(
        fiber, tuple(float(f) for f in frequencies), tuple(float(p) for p in powers),
        tuple(int(q) for q in quantum_indices), float(distance), bool(backward))


def calculate_quantum_metrics(forward_resources, backward_resources,
                              forward_powers, backward_powers, distance, frequencies,
                              first_neighbors, secondary_neighbors,
                              first_fiber, secondary_fiber,
                              detector_params, bb84_params):
    """汇总一条链路当前状态的 SKR（bit/s）、噪声功率（W）、每门噪声计数及量子信道数量。
    SKR、功率和噪声计数均跨量子信道求和；正式 skr 逐信道截零，raw_skr 仅用于诊断。"""
    # 频率、邻芯表、物理模型及探测/BB84 参数同 calculate_SKR_core。
    # 两方向 resources/powers 均为 [芯, 信道] 数组（如 (8, 16)，芯轴第 0 项留空），状态 0/1/2/3 为不可用/空闲经典/占用经典/量子保留，功率单位 W。
    # forward 为小节点到大节点的量子接收方向，backward 为反向经典光；distance 单位 m（如 1000）。
    frequencies = np.asarray(frequencies, dtype=float)
    result = dict(skr=0.0, raw_skr=0.0, no_fwm_skr=0.0, zero_noise_skr=0.0,
                  raman_w=0.0, fwm_w=0.0, noise_counts=0.0,
                  quantum_channels=0, zero_skr_channels=0)
    # qc/qi 为量子芯/信道索引；按量子频点分别累加两个传播方向的 Raman（拉曼）与 FWM 噪声功率。
    for qc in np.flatnonzero(np.any(forward_resources == 3, axis=1)):
        qi = np.flatnonzero(forward_resources[qc] == 3)
        raman, fwm = np.zeros(len(qi)), np.zeros(len(qi))
        for backward, resources, powers in (
                (False, forward_resources, forward_powers),
                (True, backward_resources, backward_powers)):
            for cc in first_neighbors[qc] + secondary_neighbors[qc]:
                # cc 为经典芯编号，active 只保留状态 2 的占用信道功率。
                active = np.where(resources[cc] == 2, powers[cc], 0.0)
                r, f = calculate_quantum_noise_components(
                    qc, cc, backward, active, qi, distance, frequencies,
                    first_neighbors, secondary_neighbors, first_fiber, secondary_fiber)
                raman += r
                fwm += f
        # counts/rcounts 分别是总噪声/仅拉曼噪声的每门计数，效率及插损在此统一换算。
        counts = noise_power_to_counts(raman + fwm, frequencies[qi], detector_params)
        rcounts = noise_power_to_counts(raman, frequencies[qi], detector_params)
        for count, rcount in zip(counts, rcounts):
            # raw 保留有效界下的负差值，界失效仍为零；正式 SKR 禁止正负信道相互抵消。
            raw = BB84_SKR(distance, float(count), bb84_params, detector_params, clip=False)[0]
            result['quantum_channels'] += 1
            result['zero_skr_channels'] += int(raw <= 0)
            result['raw_skr'] += raw
            result['skr'] += max(0.0, raw)
            # 两项假设指标只去掉 FWM 或全部外加噪声，保留暗计数和有限样本模型，不代表其他算法。
            result['no_fwm_skr'] += max(0.0, BB84_SKR(
                distance, float(rcount), bb84_params, detector_params)[0])
            result['zero_noise_skr'] += max(0.0, BB84_SKR(
                distance, 0.0, bb84_params, detector_params)[0])
        result['raman_w'] += float(raman.sum())
        result['fwm_w'] += float(fwm.sum())
        result['noise_counts'] += float(counts.sum())
    return result
