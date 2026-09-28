"""由 main 或 traffic_scan 调用，将仿真状态及统计结果导出为回放 JSON、Excel 和图表。
输出目录由调用方指定，本模块不生成业务、执行分配或重算统计。"""
from dataclasses import asdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from noise_calculation import raman_model_config
from skr_calculation import skr_model_config


TRACE_SCHEMA_VERSION = 3
# 七芯专用映射：仿真 0..5 是外圈物理 2..7，仿真 6 是物理中心芯 1。
CORE_TO_PHYSICAL = [2, 3, 4, 5, 6, 7, 1]


class TrafficRecorder:
    """观察已完成事件并记录所选链路的资源归属，生成三芯业务回放。"""

    def __init__(self, sim, *, seed, warmup):
        """接收三芯实例 sim，记录随机种子 seed（如 53）和预热时长 warmup（仿真时间单位）。"""
        if not sim.bind_three:
            raise ValueError('Business trace requires a three-core experiment')
        self.sim = sim
        a, b = sim.observed_link
        self.links = [(a, b), (b, a)]
        self.link_index = {tuple(link): i for i, link in enumerate(self.links)}
        # occupancy[有向链路][仿真芯][经典信道]：-2 不可用、-1 空闲、非负整数为业务编号。
        self.occupancy = [
            [[-1 if sim.m_resourceMap[a, b, c, w] == 1 else -2
              for w in range(sim.quantum_wave_num, sim.WaveNumber)]
             for c in range(sim.core_num)] for a, b in self.links]
        # 量子资源只存配置；经典索引从零开始，频段缺口应读取频率表，不能按索引连续推算。
        self.config = dict(
            allow_bidirectional=sim.allow_bidirectional,
            algorithm=sim.algorithm,
            seed=int(seed), duration=float(sim.Ts), warmup=float(warmup),
            observed_link=list(sim.observed_link), allocation_scope='full_network',
            network_node_count=len(sim.graph), network_edge_count=sim.graph.number_of_edges(),
            network_length_scaling=deepcopy(sim.graph.graph['length_scaling']),
            network_edges_m=[(int(a), int(b), float(sim.a_m[a, b])) for a, b in sim.graph.edges],
            mean_holding_time=float(sim.m_rou1), arrival_rate=float(sim.lambda1),
            offered_load_erlang=float(sim.lambda1 * sim.m_rou1),
            power_dbm=float(10 * math.log10(sim.launch_power / 1e-3)),
            power_reference='per_core_per_classical_channel_at_fiber_input',
            allocation_mode='three_core_bound', traffic_unit='three_core_business_group',
            load_unit='Erlang of three-core business groups',
            arrival_rate_unit='three-core groups per simulation unit',
            direction_definition='business direction uses end-to-end source/destination; each hop direction uses its link endpoints (forward u<v, backward u>v); replay must use hop direction',
            time_unit='simulation_unit', direction_mode='bidirectional',
            directed_links=[list(link) for link in self.links],
            link_lengths_m=[float(sim.a_m[a, b]) for a, b in self.links],
            core_to_physical=CORE_TO_PHYSICAL,
            forward_cores=list(sim.classical_forward_cores),
            backward_cores=list(sim.classical_backward_cores),
            quantum_cores=list(sim.quantum_cores),
            quantum_frequencies_hz=[int(f) for f in sim.available_channel[:sim.quantum_wave_num]],
            classical_frequencies_hz=[int(f) for f in sim.available_channel[sim.quantum_wave_num:]],
            channel_spacing_hz=int(sim.wave_interval),
            end_policy='release_at_horizon',
            blocked_records_scope='all_network_arrivals',
            event_order='states array order is authoritative, including equal timestamps',
            skr_model=skr_model_config(sim.bb84_params, sim.detector_params),
            raman_model=raman_model_config(),
            first_fiber=asdict(sim.noise_model.first_fiber),
            secondary_fiber=asdict(sim.noise_model.secondary_fiber),
        )
        # 保存预热前初态，states 依事件原顺序记录，包含预热及同刻事件。
        self.initial = deepcopy(self.occupancy)
        self.active = {}
        self.states = []
        self.business_groups = []

    def record(self, event, *, blocked=False):
        """记录所选链路的已接入组与全网阻塞到达，并核对回放是否与实际资源一致。"""
        bid = int(event.m_id)
        arrival = bool(event.m_eventType['Arrival'])
        # 阻塞不一定有工作路径，先保留记录；成功业务只导出观测链路部分。
        if arrival:
            if not blocked and not any((a, b) in self.link_index for a, b in zip(event.m_workPath, event.m_workPath[1:])):
                return
        elif bid not in self.active:
            return
        if arrival:
            # 端到端方向只描述业务；实验回放须使用 allocation.hops 中逐跳方向。
            service = dict(group_id=bid, source=int(event.m_sourceNode), destination=int(event.m_destNode),
                direction='forward' if event.m_sourceNode < event.m_destNode else 'backward',
                arrival_time=float(event.m_time), holding_time=float(event.m_holdTime),
                scheduled_end_time=float(event.m_time + event.m_holdTime),
                status='blocked' if blocked else 'accepted', allocation=None,
                release_time=None, release_reason=None)
            self.business_groups.append(service)
            if not blocked:
                path = [int(n) for n in event.m_workPath]
                frequency = float(self.sim.available_channel[event.m_ocuppiedwave])
                # path 保留完整路由，hops 仅含观测链路；物理芯号不代表硬件端口。
                service['allocation'] = dict(path=path,
                    hops=[dict(link=[a, b], direction='forward' if a < b else 'backward',
                               cores=[int(c) for c in cores],
                               physical_cores=[CORE_TO_PHYSICAL[c] for c in cores])
                          for a, b, cores in zip(path, path[1:], event.m_ocuppiedcore)
                          if (a, b) in self.link_index],
                    channel_index=int(event.m_ocuppiedwave - self.sim.quantum_wave_num),
                    frequency_hz=frequency, wavelength_nm=299792458 / frequency * 1e9)
                self.active[bid] = service
        else:
            service = self.active.pop(bid)
            service.update(release_time=float(event.m_time), release_reason='natural')
        if not blocked:
            kind = 'arrival' if arrival else 'leave'
            self._update(service, kind)
            self.states.append(dict(time=float(event.m_time), event=kind,
                                    group_id=bid, occupancy=deepcopy(self.occupancy)))
        # 阻塞保留到达记录，但分配和释放字段为空，占用状态不变。
        if blocked:
            self.states.append(dict(time=float(event.m_time), event='blocked',
                                    group_id=bid, occupancy=deepcopy(self.occupancy)))
        self._check_state()

    def _update(self, service, kind):
        """回放三芯归属一起更新；到达须空闲，离去须属于同一组，否则报错。"""
        allocation = service['allocation']
        wave = allocation['channel_index']
        for hop in allocation['hops']:
            a, b = hop['link']
            for core in hop['cores']:
                row = self.occupancy[self.link_index[a, b]][core]
                expected = -1 if kind == 'arrival' else service['group_id']
                if row[wave] != expected:
                    raise ValueError('Trace ownership disagrees with simulation lifecycle')
                row[wave] = service['group_id'] if kind == 'arrival' else -1

    def _check_state(self):
        """只读核查两方向的三芯一致性及回放占用；不会修复或改变仿真状态。"""
        sim = self.sim
        for index, (a, b) in enumerate(self.links):
            resources = sim.m_resourceMap[a, b, :, sim.quantum_wave_num:]
            powers = sim.P_link[a, b, :, sim.quantum_wave_num:]
            recorded = np.asarray(self.occupancy[index])
            actual = np.where(recorded >= 0, 2, np.where(recorded == -1, 1, 0))
            cores = sim.classical_forward_cores if a < b else sim.classical_backward_cores
            if (not np.array_equal(resources, actual)
                    or not np.all(recorded[cores] == recorded[cores[0]])
                    or not np.allclose(powers, np.where(resources == 2, sim.launch_power, 0),
                                       rtol=1e-6, atol=0)):
                raise ValueError('Three-core trace disagrees with actual resource or per-core power')

    def finish(self, metrics, source_hashes):
        """生成含终点释放记录的回放字典，保留原仿真时间单位，不修改仿真资源或指标。"""
        self._check_state()
        # 终点强制释放只结束回放；计划离去时间可超过终点，不能解释成自然离去。
        for bid in sorted(self.active):
            self.active[bid].update(release_time=float(self.sim.Ts), release_reason='horizon')
            self._update(self.active[bid], 'leave')
        if self.active:
            self.states.append(dict(time=float(self.sim.Ts), event='horizon_release',
                                    group_ids=sorted(self.active), occupancy=deepcopy(self.occupancy)))
        self.active.clear()
        return dict(schema_version=TRACE_SCHEMA_VERSION, kind='qkd_resource_timeline',
                    config=self.config, source_sha256=source_hashes,
                    traffic_sha256=metrics['traffic_sha256'], metrics=metrics,
                    business_groups=self.business_groups,
                    initial_occupancy=self.initial, states=self.states)



def export_replay_timing(data, duration_seconds=None):
    """将 data 回放升级为版本 3 的独立副本，按 duration_seconds 换算总时长。
传入 None 时保留现有时间单位；输入记录、事件顺序和统计值不变。"""
    if data.get('kind') != 'qkd_resource_timeline' or data.get('schema_version') not in (2, 3):
        raise ValueError('仅支持 qkd_resource_timeline 版本 2/3')
    result = deepcopy(data)
    config = result['config']
    current_duration = config['duration']
    if not math.isfinite(current_duration) or current_duration <= 0:
        raise ValueError('原回放 duration 必须为有限正数')
    if config['time_unit'] not in ('simulation_unit', 's'):
        raise ValueError('不支持的回放时间单位')
    # 首次保存原仿真参数；已换为秒的回放再次缩放时仍保留同一来源。
    if 'source_simulation' not in result:
        if config['time_unit'] != 'simulation_unit':
            raise ValueError('秒制回放缺少原仿真来源信息')
        result['source_simulation'] = {key: deepcopy(config[key]) for key in (
            'duration', 'warmup', 'mean_holding_time', 'arrival_rate',
            'arrival_rate_unit', 'time_unit', 'offered_load_erlang')}
        result['source_simulation']['schema_version'] = data['schema_version']
    source = result['source_simulation']
    if duration_seconds is not None:
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError('实验总时长必须为有限正数（秒）')
        # 如 100 个时间单位换为 3600 秒，时间乘 36，到达率除 36，Erlang 负载不变。
        factor = duration_seconds / current_duration
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError('时间换算系数超出数值范围')
        for key in ('warmup', 'mean_holding_time'):
            config[key] *= factor
        config['duration'] = float(duration_seconds)
        config['arrival_rate'] /= factor
        config.update(time_unit='s', arrival_rate_unit='three-core groups per second')
        # 只换算业务与回放事件时间；metrics/samples（含 samples.time）保持原仿真单位。
        for service in result['business_groups']:
            for key in ('arrival_time', 'holding_time', 'scheduled_end_time', 'release_time'):
                if service[key] is not None:
                    service[key] *= factor
        for state in result['states']:
            state['time'] *= factor
    config['time_scale'] = config['duration'] / source['duration']
    config['time_scale_unit'] = config['time_unit'] + ' per simulation_unit'
    # 版本 2 缺失的阻塞记录无法由统计值补造，只标记来源缺口。
    config.setdefault('blocked_records_scope', 'not_recorded_in_source_v2')
    # 同刻事件及终点释放均按原数组顺序回放，不重新排序。
    config['event_order'] = 'array_order'
    # 只输出回放实际使用的配置；算法说明和全网构建参数留在批次索引。
    replay_fields = (
        'algorithm', 'seed', 'duration', 'warmup', 'time_unit', 'allow_bidirectional',
        'time_scale', 'time_scale_unit', 'mean_holding_time', 'arrival_rate',
        'arrival_rate_unit', 'offered_load_erlang', 'power_dbm', 'power_reference',
        'allocation_mode', 'directed_links', 'link_lengths_m', 'core_to_physical',
        'forward_cores', 'backward_cores', 'quantum_cores', 'quantum_frequencies_hz',
        'classical_frequencies_hz', 'end_policy', 'blocked_records_scope', 'event_order',
        'skr_model', 'raman_model', 'first_fiber', 'secondary_fiber')
    result['config'] = {key: config[key] for key in replay_fields if key in config}
    if 'skr_model' in result['config']:
        # 保留模型版本和实际数值参数，长篇公式解释已有源码说明。
        result['config']['skr_model'] = {
            key: value for key, value in config['skr_model'].items()
            if not key.endswith('_definition') and key not in ('skr_reference', 'skr_security_scope')}
    # 统计值保持原样；时间窗口已由 source_simulation 给出，无需重复。
    result['simulation_statistics'] = dict(time_unit='simulation_unit')
    # 移除两个顶层重复摘要，其余顶层数据原样保留。
    result.pop('source_sha256', None)
    result.pop('traffic_sha256', None)
    result['schema_version'] = TRACE_SCHEMA_VERSION
    return result


def write_trace(path, data):
    """以 UTF-8 排版写入 JSON 并返回 SHA-256 文件摘要；拒绝覆盖已有文件及非有限数值。"""
    def render(value, level=0):
        """递归排版 JSON：字段和嵌套记录换行，简单数组单行，保持原有顺序。"""
        indent = '  ' * level
        child_indent = indent + '  '
        if isinstance(value, dict) and value:
            rows = [child_indent + json.dumps(key) + ': ' + render(item, level + 1)
                    for key, item in value.items()]
            return '{\n' + ',\n'.join(rows) + '\n' + indent + '}'
        if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
            rows = [child_indent + render(item, level + 1) for item in value]
            return '[\n' + ',\n'.join(rows) + '\n' + indent + ']'
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render(data) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)
    return sha256(path.read_bytes()).hexdigest()


def export_excel(path, tables):
    """将 tables 汇总记录写为黑白工作簿，包含条件、算法、四项指标及相对 FF 的比值。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter

    # tables 如 [(表名, 汇总记录, [(条件键, 显示名)])]，同表用全部条件匹配基准。
    workbook = Workbook()
    workbook.remove(workbook.active)
    # 秘密密钥率（SKR）由 bit/s 转 kbit/s；光信噪比（OSNR）显示 dB，比值使用线性值。
    metrics = [('skr_mean', 'SKR (kbit/s)', .001, '0.000'),
               ('osnr_db_mean', 'OSNR (dB)', 1, '0.000'),
               ('blocking_rate', 'Blocking rate', 1, '0.00%'),
               ('synergy_vs_FF', 'Synergy', 1, '0.000000')]
    ratio_keys = ('skr_mean', 'osnr_linear_mean', 'blocking_rate', 'synergy_vs_FF')
    for name, records, conditions in tables:
        if len(records) > 1_048_575:
            raise ValueError(f'{name} exceeds Excel row limit')
        sheet = workbook.create_sheet(name)
        headers = [label for _, label in conditions] + ['Algorithm']
        headers += [label for _, label, _, _ in metrics]
        headers += ['SKR / FF', 'OSNR / FF (linear)', 'Blocking / FF', 'Synergy / FF']
        sheet.append(headers)
        baseline = {tuple(row[key] for key, _ in conditions): row for row in records
                    if row['algorithm'] == 'FF'}
        for row in records:
            key = tuple(row[key] for key, _ in conditions)
            # 优先使用统计层附带的 FF 参考，内部补跑基准不必作为表格行导出。
            ref = row.get('ff_reference', baseline.get(key, {}))
            values = list(key) + [row['algorithm']]
            values += [row[metric] * factor if row[metric] is not None else None
                       for metric, _, factor, _ in metrics]
            # 比值不减一；缺值或基准为零留空，FF 自身协同度为零，故该比值列为空。
            values += [row[metric] / ref[metric]
                       if row.get(metric) is not None and ref.get(metric) not in (None, 0)
                       else None for metric in ratio_keys]
            sheet.append(values)
        formats = ['General'] * len(conditions) + ['General']
        formats += [fmt for _, _, _, fmt in metrics] + ['0.000000'] * 4
        for column, (label, number_format) in enumerate(zip(headers, formats), 1):
            sheet.column_dimensions[get_column_letter(column)].width = 26 if label == 'Algorithm' else 20
            sheet.cell(1, column).alignment = Alignment(wrap_text=True, vertical='center')
            for cells in sheet.iter_cols(min_col=column, max_col=column, min_row=2):
                for cell in cells:
                    cell.number_format = number_format
        sheet.row_dimensions[1].height = 30
    workbook.save(path)


def annotate_max_gap(ax, points, metric, unit):
    """在 ax 上标注 QCNM 相对 CCA 的最大指标差及对应容差，缺有效配对时不标注。"""
    # points 为 (横轴值, QCNM 值, CCA 值, 容差标签)，数值已转为图中单位且按同工况配对。
    pairs = [(x, proposed, cca, algorithm) for x, proposed, cca, algorithm in points
             if proposed is not None and cca is not None
             and np.isfinite(proposed) and np.isfinite(cca)]
    if not pairs:
        return
    # 最大差相同则选择较小横轴；全相等标零。SKR 用绝对百分比，OSNR 保留 dB 差的符号。
    if metric == 'SKR':
        pairs = [p for p in pairs if p[2] > 0]  # CCA 非正时比例未定义。
        if not pairs:
            return
        x, proposed, cca, algorithm = max(pairs, key=lambda p: (abs(p[1] / p[2] - 1), -p[0]))
        label = f'Max SKR difference vs CCA: {abs(proposed / cca - 1):.2%}'
    else:
        x, proposed, cca, algorithm = max(pairs, key=lambda p: (abs(p[1] - p[2]), -p[0]))
        label = f'Max {metric} gap vs CCA: {proposed - cca:+.3g} {unit}'
    label = algorithm + '\n' + label
    difference = proposed - cca
    if difference != 0:
        ax.annotate('', xy=(x, proposed), xytext=(x, cca),
                    arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
    ax.annotate(label,
                xy=(x, (proposed + cca) / 2), xytext=(.03, .97),
                textcoords='axes fraction', ha='left', va='top', fontsize=9,
                arrowprops=dict(arrowstyle='-', color='#555555', lw=.8),
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))


def plot_scan(output, summary, scan_axis='load'):
    """将 summary 的均值和种子间标准差绘为四指标扫描图，返回 SVG 文件名列表。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # summary 已完成跨种子汇总；负载按场景分图，功率/距离每个横轴值须只有一个场景。
    frame = pd.DataFrame(summary)
    colors = {name: plt.get_cmap('tab10')(i % 10)
              for i, name in enumerate(dict.fromkeys(frame.algorithm))}
    artifacts = []
    x_key, x_label = {
        'load': ('offered_load_erlang', 'Offered traffic (Erlang)'),
        'power': ('power_dbm', 'Power per core per channel (dBm)'),
        'distance': ('length_km', 'Uniform link length (km)'),
    }[scan_axis]
    # 物理量扫描必须将不同场景的点连成一条曲线；配对统计仍在各场景内完成。
    groups = frame.groupby('scenario', sort=False) if scan_axis == 'load' else [(
        f"{scan_axis.capitalize()} sweep | A={frame.offered_load_erlang.iloc[0]:g} Erlang | " + (
            f"observed length={frame.observed_length_m.iloc[0] / 1000:g} km" if scan_axis == 'power'
            else f"power={frame.power_dbm.iloc[0]:g} dBm/core/channel"), frame)]
    for index, (scenario, group) in enumerate(groups, 1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        specs = [('synergy_vs_FF', 'Signed synergy vs FF', 1),
                 ('osnr_db_mean', 'Reference link OSNR (dB)', 1),
                 ('skr_mean', 'Mean link SKR per channel (kbit/s)', .001),
                 ('blocking_rate', 'Blocking probability', 1)]
        for ax, (metric, label, factor) in zip(axes.flat, specs):
            for algorithm, data in group.groupby('algorithm', sort=False):
                data = data.sort_values(x_key)
                x, y = data[x_key].to_numpy(), data[metric].to_numpy(dtype=float)*factor
                # 标准差是种子均值间的样本标准差；单种子不画阴影，缺均值处保留断点。
                sd = data[metric + '_sd'].to_numpy(dtype=float)*factor
                color = colors[algorithm]
                ax.plot(x, y, marker='o', ms=4, lw=1.7, label=algorithm, color=color)
                if np.isfinite(sd).any():
                    ax.fill_between(x, y - sd, y + sd, color=color, alpha=.12)
            ax.set_ylabel(label)
        # 仅 SKR/OSNR 比较全部 QCNM 容差中的最大差，并标出对应容差。
        cca = group[group.algorithm == 'CCA'].set_index(x_key)
        for ax, metric, factor, label, unit in (
                (axes[1, 0], 'skr_mean', .001, 'SKR', 'kbit/s'),
                (axes[0, 1], 'osnr_db_mean', 1, 'OSNR', 'dB')):
            points = [(row[x_key], row[metric] * factor, cca.loc[row[x_key], metric] * factor, row['algorithm'])
                      for _, row in group.iterrows() if row['algorithm'].startswith('QCNM(')
                      and row[x_key] in cca.index and pd.notna(row[metric])
                      and pd.notna(cca.loc[row[x_key], metric])]
            annotate_max_gap(ax, points, label, unit)
        axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, 0].axhline(0, color="#AAAAAA", lw=.8)
        axes[0, 0].legend(fontsize=8)
        for ax in axes.flat:
            ax.set_xlabel(x_label)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        seeds = int(group.seed_count.iloc[0])
        fig.suptitle(f'{scenario} | {seeds} seed(s) | shaded: across-seed SD', fontsize=13)
        path = output / f'traffic_scan_{index}.svg'
        fig.savefig(path)
        artifacts.append(path.name)
        plt.close(fig)
    return artifacts


def export_business_summary(output, runs, metadata, summary):
    """将业务汇总导出为 LoadSweep/PowerSweep 工作表及四指标的 PNG、SVG 图，返回文件名列表。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    specs = [
        ('load_scan', 'LoadSweep', 'load_erlang', 'Three-core group traffic (Erlang)',
         f"Load sweep | {metadata['fixed_power_dbm']:g} dBm/core/channel"),
        ('power_scan', 'PowerSweep', 'power_dbm', 'Power per core per channel (dBm)',
         f"Power sweep | {metadata['fixed_load_erlang']:g} Erlang of three-core groups")]
    algorithms = list(dict.fromkeys(r['algorithm'] for r in runs))
    colors = {name: plt.get_cmap('tab10')(i % 10) for i, name in enumerate(algorithms)}
    # 工作簿只保留两张汇总表，不嵌图；完整配置、种子与样本仍在 JSON 中。
    path = output / 'skr_summary.xlsx'
    tables = [(sheet, summary[group], [('load_erlang', 'Load (Erlang)'), ('power_dbm', 'Power (dBm)')])
              for group, sheet, *_ in specs]
    export_excel(path, tables)
    artifacts = ['skr_summary.xlsx']
    note = (f"{runs[0]['observed_length_m'] / 1000:g} km bidirectional | slots={metadata['slots']}, warmup={metadata['warmup']} | "
            f"{len(metadata['seeds'])} seed(s); " +
            ('error bars: across-seed SD' if len(metadata['seeds']) > 1 else 'single-seed trend'))
    metrics = [('osnr_db_mean', 'osnr_db_mean_sd', 'Reference link OSNR (dB)', 'osnr_trends.png', 'classical_osnr.png'),
               ('synergy_vs_FF', 'synergy_vs_FF_sd', 'Signed synergy vs FF', 'synergy_trends.png', 'synergy.png'),
               ('skr_mean', 'skr_mean_sd', 'Mean link SKR per channel (kbit/s)', 'skr_trends.png', 'total_skr.png'),
               ('blocking_rate', 'blocking_rate_sd', 'Classical blocking probability',
                'blocking_trends.png', 'blocking_rate.png')]
    for metric, sd_key, y_label, overview, filename in metrics:
        blocking = metric == 'blocking_rate'
        factor = .001 if metric == 'skr_mean' else 1
        upper = max(((r[metric] + (r[sd_key] or 0)) * factor
                     for rows in summary.values() for r in rows if r[metric] is not None), default=0)
        lower = min(((r[metric] - (r[sd_key] or 0)) * factor
                     for rows in summary.values() for r in rows if r[metric] is not None), default=0)
        lower = min(0, lower * 1.08)
        upper = min(1, max(.01, upper) * 1.15) if blocking else max(.01, upper * 1.08)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, (group, sheet_name, x_key, x_label, title) in zip(axes, specs):
            single, single_ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
            for algorithm in algorithms:
                points = sorted((r for r in summary[group] if r['algorithm'] == algorithm), key=lambda r: r[x_key])
                x = [r[x_key] for r in points]
                y = [r[metric] * factor if r[metric] is not None else np.nan for r in points]
                # 单种子不画误差棒；缺失值断线，重合曲线保持原位置。
                sd = [r[sd_key] * factor if r[sd_key] is not None else None for r in points]
                for target in (ax, single_ax):
                    target.plot(x, y, marker='o', ms=4, lw=1.8, label=algorithm, color=colors[algorithm])
                    if all(v is not None for v in sd):
                        target.errorbar(x, y, yerr=sd, fmt='none', capsize=3, color=colors[algorithm])
            if blocking:
                for target in (ax, single_ax):
                    target.yaxis.set_major_formatter(PercentFormatter(1))
            elif metric in ('skr_mean', 'osnr_db_mean'):
                label, unit = ('SKR', 'kbit/s') if metric == 'skr_mean' else ('OSNR', 'dB')
                cca = {r[x_key]: r[metric] for r in summary[group] if r['algorithm'] == 'CCA'}
                points = [(r[x_key], r[metric] * factor, cca[r[x_key]] * factor, r['algorithm'])
                          for r in summary[group] if r['algorithm'].startswith('QCNM(')
                          and r[metric] is not None and cca.get(r[x_key]) is not None]
                for target in (ax, single_ax):
                    annotate_max_gap(target, points, label, unit)
            for target in (ax, single_ax):
                target.set(xlabel=x_label, ylabel=y_label, title=title, ylim=(lower, upper))
                target.grid(alpha=.2)
                target.spines[['top', 'right']].set_visible(False)
                target.legend(fontsize=9, loc='lower right' if blocking else 'best')
            single.suptitle(note, fontsize=8)
            figure_path = f'{group}/{filename}'
            single.savefig(output / figure_path, dpi=180)
            vector_path = str(Path(figure_path).with_suffix('.svg'))
            single.savefig(output / vector_path)
            artifacts.append(vector_path)
            plt.close(single)
            artifacts.append(figure_path)
        fig.suptitle(note, fontsize=10)
        fig.savefig(output / overview, dpi=180)
        vector_path = str(Path(overview).with_suffix('.svg'))
        fig.savefig(output / vector_path)
        artifacts.append(vector_path)
        plt.close(fig)
        artifacts.append(overview)
    return artifacts


def export_scan_results(output, data, save_samples=False):
    """输出 traffic_scan.json、xlsx 和四指标对比 SVG；JSON 始终保留 samples，不重算统计。"""
    (output / 'traffic_scan.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    export_excel(output / 'traffic_scan.xlsx', [('Summary', data['summary'], [
        ('offered_load_erlang', 'Load (Erlang)'), ('power_dbm', 'Power (dBm)'),
        ('observed_length_m', 'Link length (m)')])])
    return plot_scan(output, data['summary'], data['config'].get('scan_axis', 'load'))
