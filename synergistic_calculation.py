"""参考 KeyConsumption_24node -7 core 的协同度，由 main/traffic_scan 调用。

输入为同链路同种子的时间平均线性 OSNR 和每量子信道原始 SKR（bit/s）。
严格保留参考 calculate_synergistic 的固定标尺：1 mW、40/71 kbit/s，
距离为 40 km 时 leap=3，其他距离 leap=4。这是参考归一化标尺，
不是观测链路的实际跳数；链路 OSNR 自身采用一跳。
S=sqrt(abs(Uc1-Uc2)*abs(Uq1-Uq2))，非负、无截断、不保证 <=1，
也不能表示提升方向。另存原始差值以区分提升和退化；FF 对自身为零。
空闲窗口的 OSNR 未定义，协同度也为 None。历史结果不改写。
"""
import math
from noise_calculation import forward_P_XT, XT_PARAMS

METRIC_DEFINITIONS = dict(
    metric_reference='KeyConsumption_24node -7 core: Consumption_Dynamic.py, synergistic_calculation.py, SKR_calculate.py, BB84_SKR.py, XT.py',
    metric_scope='One selected undirected link; classical traffic routed and allocated across the full network',
    skr_definition='Time mean of raw BB84 SKR per reserved quantum channel on the selected link; no clipping; quantum direction smaller to larger node',
    osnr_definition='Single-link adaptation of reference business entry: mean received signal / (mean XT + 3.21e-9 W), one hop; linear time mean over nonempty slot ends, then dB',
    osnr_db_definition='10*log10 of seed mean linear OSNR; across seeds report mean and sample SD of seed dB values',
    classical_noise_model='Reference XT only: nearest 1e-6 and second-nearest 1e-7 km^-1; retain reference victim-power gating for backward XT; fixed floor 3.21e-9 W per occupied channel',
    classical_noise_floor_w=3.21e-9,
    osnr_model_version='reference_business_single_link_v1',
    osnr_idle_definition='Empty slots excluded; all-idle window is null; occupied zero-XT slots remain finite due to reference floor',
    synergy_definition='sqrt(abs(Uc_A-Uc_FF)*abs(Uq_A-Uq_FF)); Uc=(OSNR-C_alpha)/(C_beta-C_alpha); Uq=(SKR-40000)/(71000-40000)',
    synergy_reference='C_alpha=receive_power(1e-3,L)/(leap*(6*forward_P_XT(1e-6,L,1e-3)+3.21e-9)); C_beta=receive_power(1e-3,L)/(leap*3.21e-9); leap=3 at L=40000 m, otherwise 4',
    synergy_version='reference_unsigned_v1', synergy_baseline='FF',
    synergy_interpretation='Nonnegative difference magnitude; no sign, no clipping, no guaranteed upper bound of 1; not a percentage gain',
)


def osnr_db(linear):
    """线性功率比转 dB；空值或非正值返回 None。"""
    return 10 * math.log10(linear) if linear is not None and linear > 0 else None


def calculate_synergistic(distance, osnr1, skr1, osnr2, skr2):
    """与参考五参数函数相同：距离 m、线性 OSNR、每信道 SKR bit/s。"""
    if any(v is None or not math.isfinite(v) for v in (osnr1, skr1, osnr2, skr2)):
        return None
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Synergy distance must be finite and positive')
    leap = 3 if distance == 40e3 else 4
    received = 1e-3 * math.pow(10, -(distance * 0.2 * 1e-4))
    c_alpha = received / (leap * (6 * forward_P_XT(1e-6, distance, 1e-3, XT_PARAMS) + 3.21e-9))
    c_beta = received / (leap * 3.21e-9)
    uc1, uc2 = (osnr1-c_alpha)/(c_beta-c_alpha), (osnr2-c_alpha)/(c_beta-c_alpha)
    uq1, uq2 = (skr1-40000)/31000, (skr2-40000)/31000
    return math.sqrt(abs(uc1-uc2) * abs(uq1-uq2))


def add_paired_synergy(runs, keys, *, baseline='FF'):
    """同工况同种子配对，不跨链路比较；缺 FF 或 OSNR 时协同度留空。"""
    references = {tuple(row[key] for key in keys): row for row in runs if row['algorithm'] == baseline}
    for row in runs:
        ref = references.get(tuple(row[key] for key in keys))
        if ref is not None and (row['observed_link'] != ref['observed_link'] or row['observed_length_m'] != ref['observed_length_m']):
            raise ValueError('Paired algorithms must observe the same link and length')
        row['synergy_vs_FF'] = None if ref is None else calculate_synergistic(
            row['observed_length_m'], row['osnr_linear_mean'], row['skr_mean'],
            ref['osnr_linear_mean'], ref['skr_mean'])
        valid = ref is not None and row['osnr_linear_mean'] is not None and ref['osnr_linear_mean'] is not None
        row['delta_osnr_linear_vs_FF'] = row['osnr_linear_mean']-ref['osnr_linear_mean'] if valid else None
        row['delta_skr_vs_FF'] = row['skr_mean']-ref['skr_mean'] if ref is not None else None
