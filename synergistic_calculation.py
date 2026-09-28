"""供 main/traffic_scan 调用，将同工况算法与 FF 基准的光信噪比（OSNR）和秘密密钥率（SKR）换算为有符号协同度。
输入为线性 OSNR 与 bit/s 密钥率，返回无量纲指标或未定义值 None。"""
import math

from noise_calculation import (CLASSICAL_FIBER_LOSS_DB_PER_KM, CLASSICAL_NOISE_FLOOR_W,
                               CLASSICAL_XT_SOURCE_COUNT)

# 协同度模型版本，随结果导出，便于判断历史结果的公式口径；基准固定为 FF。
SYNERGY_MODEL_VERSION = 'shared_fiber_icxt_bounds_signed_v5'


def osnr_db(linear):
    """线性功率比转 dB；空值或非正值返回 None。"""
    # linear 为线性光信噪功率比，例如 100 对应 20 dB。
    return 10 * math.log10(linear) if linear is not None and linear > 0 else None


def calculate_synergistic(
    distance,  # 观测链路长度，m，例如 1000。
    osnr1,  # 待评价算法的线性 OSNR（不是 dB），例如 100。
    skr1,  # 待评价算法的平均 SKR，bit/s，例如 1e4。
    osnr2,  # 同工况 FF 基准的线性 OSNR，例如 80。
    skr2,  # 同工况 FF 基准的平均 SKR，bit/s，例如 8e3。
    *,
    power,  # 每芯每经典信道的发射功率，W，例如 1e-3。
    skr_lower,  # 本次距离、功率及探测参数下，多芯串扰对应的 SKR 下限，bit/s。
    skr_upper,  # 同一距离及探测参数下，零外加噪声对应的 SKR 上限，bit/s。
    reference_xt_power,  # 单个正向邻芯串扰功率，W；由调用方保证有限正值，例如 1e-9。
):
    """按共同物理标尺计算两算法的有符号协同度；指标缺失、非有限或量子标尺跨度非正时返回 None。"""
    if any(v is None or not math.isfinite(v) for v in (osnr1, skr1, osnr2, skr2, skr_lower, skr_upper)):
        return None
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('协同度距离必须为有限正数')
    if not math.isfinite(power) or power <= 0:
        raise ValueError('协同度功率必须为有限正数')
    if skr_upper <= skr_lower:
        return None
    # 固定衰减、固定噪声底与串扰源数取自 noise_calculation 的共用常数，保证与经典 OSNR 同口径。
    received = power * math.pow(10, -(distance * CLASSICAL_FIBER_LOSS_DB_PER_KM * 1e-4))
    # 经典标尺按实际单跳长度与功率构造：下限为全部正向串扰源，上限为零串扰，两者都保留固定噪声底。
    c_alpha = received / (CLASSICAL_XT_SOURCE_COUNT * reference_xt_power + CLASSICAL_NOISE_FLOOR_W)
    c_beta = received / CLASSICAL_NOISE_FLOOR_W
    # uc/uq 分别为经典/量子归一化指标，不截断到 [0,1]；量子上下限由调用方按实际距离计算。
    uc1, uc2 = (osnr1-c_alpha)/(c_beta-c_alpha), (osnr2-c_alpha)/(c_beta-c_alpha)
    span = skr_upper - skr_lower  # 量子归一化标尺的跨度，bit/s。
    uq1, uq2 = (skr1-skr_lower)/span, (skr2-skr_lower)/span
    # 幅值取两侧归一化差绝对值的几何平均；任一指标相等则幅值为零。
    magnitude = math.sqrt(abs(uc1-uc2) * abs(uq1-uq2))
    if magnitude == 0:
        return 0.0
    # 仅两指标相对基准都严格提高时取正；其余非零幅值取负，不限幅。
    return magnitude if osnr1 > osnr2 and skr1 > skr2 else -magnitude


def add_paired_synergy(
    runs,  # 运行结果字典列表，含算法、工况、种子及指标；skr_mean 须为数值。
    keys,  # 配对字段名列表，例如 ['scenario', 'offered_load_erlang', 'seed']。
    *,
    baseline='FF',  # 基准在结果中的算法名称，可使用 FF 的导出别名。
):
    """按 keys 指定的同工况条件配对基准，原地添加协同度及 SKR/OSNR 差值。"""
    # keys 应包含 seed，逐种子计算后由调用方跨种子平均；每组须唯一基准，重复会取最后一条。
    references = {tuple(row[key] for key in keys): row for row in runs if row['algorithm'] == baseline}
    for row in runs:
        ref = references.get(tuple(row[key] for key in keys))
        if ref is not None and (row['observed_link'] != ref['observed_link'] or row['observed_length_m'] != ref['observed_length_m']):
            raise ValueError('配对算法必须观测同一条链路且长度一致')
        if ref is not None and any(row[key] != ref[key] for key in
                                   ('synergy_skr_lower', 'synergy_skr_upper', 'synergy_reference_launch_power_w', 'synergy_reference_xt_power_w')):
            raise ValueError('配对算法必须使用同一 SKR 上下限与发射功率')
        row['synergy_vs_FF'] = None if ref is None else calculate_synergistic(
            row['observed_length_m'], row['osnr_linear_mean'], row['skr_mean'],
            ref['osnr_linear_mean'], ref['skr_mean'],
            power=row['synergy_reference_launch_power_w'],
            skr_lower=row['synergy_skr_lower'], skr_upper=row['synergy_skr_upper'],
            reference_xt_power=row['synergy_reference_xt_power_w'])
        # OSNR 缺失时仍保留可计算的 SKR 差值，避免协同度为空掩盖单侧变化。
        valid = ref is not None and row['osnr_linear_mean'] is not None and ref['osnr_linear_mean'] is not None
        row['delta_osnr_linear_vs_FF'] = row['osnr_linear_mean']-ref['osnr_linear_mean'] if valid else None
        row['delta_skr_vs_FF'] = row['skr_mean']-ref['skr_mean'] if ref is not None else None
