"""基于 KeyConsumption_24node -7 core 幅值扩展的有符号协同度，由 main/traffic_scan 调用。

输入为同链路同种子的时间平均线性 OSNR 和每量子信道非负有限样本 SKR（bit/s）。
量子侧标尺由 skr_calculation 按本次距离与参数计算：上限为零外加噪声 SKR，
下限为受到（经典纤芯数-1）个正向串扰源时的 SKR，单位 bit/s。
经典侧使用本次每芯每信道发射功率、六个正向串扰源和固定噪声底的单跳 OSNR 标尺。
幅值 M=sqrt(abs(Uc1-Uc2)*abs(Uq1-Uq2))；相对基准 OSNR 和 SKR 均严格提高时 S=+M，
否则 S=-M。任一指标相等时幅值为零，返回 0；不截断，不保证绝对值 <=1。
另存原始差值以区分单项退化和双项退化；标尺和指标有效时 FF 对自身为零。
空闲窗口的 OSNR 未定义或 SKR 上下限相等时，协同度为 None。历史结果不改写。
"""
import math
from noise_calculation import forward_P_XT, XT_PARAMS
from skr_calculation import SKR_DEFINITIONS

METRIC_DEFINITIONS = dict(
    metric_reference='Classical OSNR and synergy magnitude adapted from KeyConsumption_24node -7 core; distance-dependent SKR bounds use supplied SKR_new.py finite-size estimator',
    **SKR_DEFINITIONS,
    metric_scope='One selected undirected link; classical traffic routed and allocated across the full network',
    osnr_definition='Single-link adaptation of reference business entry: mean received signal / (mean XT + 3.21e-9 W), one hop; linear time mean over nonempty slot ends, then dB',
    osnr_db_definition='10*log10 of seed mean linear OSNR; across seeds report mean and sample SD of seed dB values',
    classical_noise_model='Reference XT only: nearest 1e-6 and second-nearest 1e-7 km^-1; retain reference victim-power gating for backward XT; fixed floor 3.21e-9 W per occupied channel',
    classical_noise_floor_w=3.21e-9,
    osnr_model_version='reference_business_single_link_v1',
    osnr_idle_definition='Empty slots excluded; all-idle window is null; occupied zero-XT slots remain finite due to reference floor',
    synergy_definition='M=sqrt(abs(Uc_A-Uc_FF)*abs(Uq_A-Uq_FF)); S=+M if OSNR_A>OSNR_FF and SKR_A>SKR_FF else -M; zero magnitude returns 0; Uc=(OSNR-C_alpha)/(C_beta-C_alpha); Uq=(SKR-Q_lower)/(Q_upper-Q_lower)',
    synergy_reference='C_alpha=receive_power(power,L)/(6*forward_P_XT(1e-6,L,power)+3.21e-9); C_beta=receive_power(power,L)/3.21e-9; power is actual launch power per core/channel in W; one hop',
    synergy_skr_bounds='Q_upper=BB84_SKR(L,0); Q_lower=mean per-channel BB84_SKR(L,noise_power_to_counts((N_classical-1)*forward_P_XT(1e-6,L,P_launch),f_q,detector)); same finite-key and detector parameters as actual SKR',
    synergy_classical_core_definition='Count unique cores in union of forward/backward classical core sets',
    synergy_skr_bound_units='SKR bit/s; distance m; launch and XT power W; coupling 1e-6 km^-1; noise counts per gate; no historical SDM half factor',
    synergy_undefined='Missing OSNR or nonpositive SKR normalization span yields null',
    synergy_version='actual_power_distance_bounds_signed_v4', synergy_baseline='FF',
    synergy_interpretation='Positive only for strict improvement in both OSNR and SKR; otherwise negative when magnitude is nonzero; equality in either metric gives zero; no clipping, no guaranteed absolute bound of 1; not a percentage gain; sign assigned per paired seed before averaging',
)


def osnr_db(linear):
    """线性功率比转 dB；空值或非正值返回 None。"""
    return 10 * math.log10(linear) if linear is not None and linear > 0 else None


def calculate_synergistic(distance, osnr1, skr1, osnr2, skr2, *, power, skr_lower, skr_upper):
    """距离 m、实际每芯每信道功率 power（W）、线性 OSNR、SKR 及上下限 bit/s。

    返回无量纲协同度或 None。指标/界缺失、非有限或量子侧跨度非正时返回 None；
    有效指标下距离和功率必须为有限正数，否则抛错。两算法共用本次参数的标尺。
    第一组为待评价算法，第二组为基准；仅 OSNR、SKR 都严格高于基准时取正幅值，
    否则取负幅值。任一指标相等时返回 0；不对归一化值或最终协同度限幅。
    """
    if any(v is None or not math.isfinite(v) for v in (osnr1, skr1, osnr2, skr2, skr_lower, skr_upper)):
        return None
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Synergy distance must be finite and positive')
    if not math.isfinite(power) or power <= 0:
        raise ValueError('Synergy power must be finite and positive')
    if skr_upper <= skr_lower:
        return None
    received = power * math.pow(10, -(distance * 0.2 * 1e-4))
    c_alpha = received / (6 * forward_P_XT(1e-6, distance, power, XT_PARAMS) + 3.21e-9)
    c_beta = received / 3.21e-9
    uc1, uc2 = (osnr1-c_alpha)/(c_beta-c_alpha), (osnr2-c_alpha)/(c_beta-c_alpha)
    span = skr_upper - skr_lower
    uq1, uq2 = (skr1-skr_lower)/span, (skr2-skr_lower)/span
    magnitude = math.sqrt(abs(uc1-uc2) * abs(uq1-uq2))
    if magnitude == 0:
        return 0.0
    return magnitude if osnr1 > osnr2 and skr1 > skr2 else -magnitude


def add_paired_synergy(runs, keys, *, baseline='FF'):
    """原地添加协同度及 SKR/OSNR 差值；缺基准/OSNR 或量子标尺重合时协同度留空。

    keys 由调用方覆盖场景、负载、功率、种子等配对条件；每键应仅有一条基准记录。
    另核对观测链路、距离、发射功率和上下限相同。baseline 允许 FF 的导出别名 first-fit，
    结果字段仍统一为 *_vs_FF；OSNR 缺失时仍保留可计算的 SKR 差值。
    """
    references = {tuple(row[key] for key in keys): row for row in runs if row['algorithm'] == baseline}
    for row in runs:
        ref = references.get(tuple(row[key] for key in keys))
        if ref is not None and (row['observed_link'] != ref['observed_link'] or row['observed_length_m'] != ref['observed_length_m']):
            raise ValueError('Paired algorithms must observe the same link and length')
        if ref is not None and any(row[key] != ref[key] for key in
                                   ('synergy_skr_lower', 'synergy_skr_upper', 'synergy_reference_launch_power_w')):
            raise ValueError('Paired algorithms must use the same SKR bounds and launch power')
        row['synergy_vs_FF'] = None if ref is None else calculate_synergistic(
            row['observed_length_m'], row['osnr_linear_mean'], row['skr_mean'],
            ref['osnr_linear_mean'], ref['skr_mean'],
            power=row['synergy_reference_launch_power_w'],
            skr_lower=row['synergy_skr_lower'], skr_upper=row['synergy_skr_upper'])
        valid = ref is not None and row['osnr_linear_mean'] is not None and ref['osnr_linear_mean'] is not None
        row['delta_osnr_linear_vs_FF'] = row['osnr_linear_mean']-ref['osnr_linear_mean'] if valid else None
        row['delta_skr_vs_FF'] = row['skr_mean']-ref['skr_mean'] if ref is not None else None
