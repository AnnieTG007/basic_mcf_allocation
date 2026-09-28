"""供 main/traffic_scan 调用，将同工况算法与 FF 基准的光信噪比（OSNR）和秘密密钥率（SKR）换算为有符号协同度。
输入为线性 OSNR 与 bit/s 密钥率，返回无量纲指标或未定义值 None。"""
import math
from skr_calculation import SKR_DEFINITIONS

METRIC_DEFINITIONS = dict(
    metric_reference='Classical OSNR and synergy magnitude adapted from KeyConsumption_24node -7 core; distance-dependent SKR bounds use supplied SKR_new.py finite-size estimator',
    **SKR_DEFINITIONS,
    metric_scope='One selected undirected link; classical traffic routed and allocated across the full network',
    osnr_definition='Single-link adaptation of reference business entry: mean received signal / (mean XT + 3.21e-9 W), one hop; linear time mean over nonempty slot ends, then dB',
    osnr_db_definition='10*log10 of seed mean linear OSNR; across seeds report mean and sample SD of seed dB values',
    classical_noise_model='ICXT only: shared fiber loss_c, loss_Rayleigh and hmn in m^-1; nearest/secondary defaults 1e-9/1e-10 m^-1; retain reference victim-power gating for backward XT; fixed floor 3.21e-9 W per occupied channel',
    classical_noise_floor_w=3.21e-9,
    osnr_model_version='reference_business_shared_fiber_icxt_v2',
    osnr_idle_definition='Empty slots excluded; all-idle window is null; occupied zero-XT slots remain finite due to reference floor',
    synergy_definition='M=sqrt(abs(Uc_A-Uc_FF)*abs(Uq_A-Uq_FF)); S=+M if OSNR_A>OSNR_FF and SKR_A>SKR_FF else -M; zero magnitude returns 0; Uc=(OSNR-C_alpha)/(C_beta-C_alpha); Uq=(SKR-Q_lower)/(Q_upper-Q_lower)',
    synergy_reference='C_alpha=receive_power(power,L)/(6*first_fiber.get_forward_icxt_power(L,power)+3.21e-9); C_beta=receive_power(power,L)/3.21e-9; power is actual launch power per core/channel in W; one hop',
    synergy_skr_bounds='Q_upper=BB84_SKR(L,0); Q_lower=mean per-channel BB84_SKR(L,noise_power_to_counts((N_classical-1)*first_fiber.get_forward_icxt_power(L,P_launch),f_q,detector)); same finite-key and detector parameters as actual SKR',
    synergy_classical_core_definition='Count unique cores in union of forward/backward classical core sets',
    synergy_skr_bound_units='SKR bit/s; distance m; launch and XT power W; shared nearest-fiber coupling hmn in m^-1; noise counts per gate; no historical SDM half factor',
    synergy_undefined='Missing OSNR or nonpositive SKR normalization span yields null',
    synergy_version='shared_fiber_icxt_bounds_signed_v5', synergy_baseline='FF',
    synergy_interpretation='Positive only for strict improvement in both OSNR and SKR; otherwise negative when magnitude is nonzero; equality in either metric gives zero; no clipping, no guaranteed absolute bound of 1; not a percentage gain; sign assigned per paired seed before averaging',
)


def osnr_db(linear):
    """线性功率比转 dB；空值或非正值返回 None。"""
    return 10 * math.log10(linear) if linear is not None and linear > 0 else None


def calculate_synergistic(distance, osnr1, skr1, osnr2, skr2, *, power, skr_lower, skr_upper, reference_xt_power):
    """按共同物理标尺计算两算法的有符号协同度；指标缺失、非有限或量子标尺跨度非正时返回 None。"""
    # distance 为 m、power 为每芯每信道 W、skr* 为 bit/s；第一组是待评价算法，第二组是基准。
    if any(v is None or not math.isfinite(v) for v in (osnr1, skr1, osnr2, skr2, skr_lower, skr_upper)):
        return None
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Synergy distance must be finite and positive')
    if not math.isfinite(power) or power <= 0:
        raise ValueError('Synergy power must be finite and positive')
    if skr_upper <= skr_lower:
        return None
    # reference_xt_power 是单个正向邻芯串扰功率（W）；调用方须保证有限正值，以免经典标尺退化。
    # 固定衰减为 0.2 dB/km，distance 从 m 换 km 后再由 dB 换为线性功率比。
    received = power * math.pow(10, -(distance * 0.2 * 1e-4))
    # 经典标尺下限含六个串扰源，上限为零串扰；两者均保留 3.21e-9 W 固定噪声底。
    c_alpha = received / (6 * reference_xt_power + 3.21e-9)
    c_beta = received / 3.21e-9
    # uc/uq 分别为经典/量子归一化指标，不截断到 [0,1]；量子上下限由调用方按实际距离计算。
    uc1, uc2 = (osnr1-c_alpha)/(c_beta-c_alpha), (osnr2-c_alpha)/(c_beta-c_alpha)
    span = skr_upper - skr_lower
    uq1, uq2 = (skr1-skr_lower)/span, (skr2-skr_lower)/span
    # 沿用参考幅值：两侧归一化差绝对值的几何平均；任一指标相等则幅值为零。
    magnitude = math.sqrt(abs(uc1-uc2) * abs(uq1-uq2))
    if magnitude == 0:
        return 0.0
    # 仅两指标相对基准都严格提高时取正；其余非零幅值取负，不限幅。
    return magnitude if osnr1 > osnr2 and skr1 > skr2 else -magnitude


def add_paired_synergy(runs, keys, *, baseline='FF'):
    """按 keys 指定的同工况条件配对基准，原地添加协同度及 SKR/OSNR 差值。"""
    # keys 如场景、负载、功率与种子，每键应仅有一条基准；baseline 可指定 FF 的导出别名。
    references = {tuple(row[key] for key in keys): row for row in runs if row['algorithm'] == baseline}
    for row in runs:
        ref = references.get(tuple(row[key] for key in keys))
        if ref is not None and (row['observed_link'] != ref['observed_link'] or row['observed_length_m'] != ref['observed_length_m']):
            raise ValueError('Paired algorithms must observe the same link and length')
        if ref is not None and any(row[key] != ref[key] for key in
                                   ('synergy_skr_lower', 'synergy_skr_upper', 'synergy_reference_launch_power_w', 'synergy_reference_xt_power_w')):
            raise ValueError('Paired algorithms must use the same SKR bounds and launch power')
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
