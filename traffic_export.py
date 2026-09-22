"""把仿真状态和已算好的统计写成资源回放 JSON、Excel 工作簿与图表。

由 traffic_scan 调用；不生成业务、不决定分配、不推进时间。输出路径由调用方给出。
回放 kind=qkd_resource_timeline、schema_version=2，config 说明频率、芯分组、
功率 dBm 和仿真时间单位。initial_occupancy 为初始状态，states 为每次资源
改变后的 {time, event, group_id, occupancy}，包含预热段。终点清空事件为
horizon_release，带 group_ids，仅结束回放，不冒充自然离去、不改变仿真统计。

本回放仅用于 --export-business 三芯实验。business_groups 每条记录是一组到达：
group_id、source/destination、direction、arrival_time、holding_time、scheduled_end_time、
status（accepted/blocked）、release_time、release_reason（natural/horizon 或 null）。
allocation 对阻塞组为 null；接入组含 path、hops（link/direction/cores/physical_cores）、
channel_index、frequency_hz 和 wavelength_nm。cores 为零起始仿真编号，physical_cores
为物理芯号，不包含硬件端口。同组同方向三芯占用一致，两个方向可复用同一波长。
负载单位为三芯业务组 Erlang，到达率按组/仿真时间计，功率为每芯每信道输入功率。

occupancy 的维度为 [有向链路, 仿真芯, 经典信道]，链路顺序见 directed_links。
-2 表示该方向不可用，-1 表示空闲，非负整数为占用业务 ID（0 也表示占用）。
量子资源只写在 config 中；经典索引从 0 开始，默认对应 C34/C32/C31/C30/C29/C28/C27，
不是 ITU 编号，也不是包含量子频率的仿真内部索引。channel_spacing_hz 只是
基础网格间隔，跳过 C33 后应读取实际 classical_frequencies_hz。

扫描工作簿含 Summary/Runs/Config，可选 Samples；业务工作簿含
LoadSweep/PowerSweep/Runs/Config。JSON/Runs 的 SKR 为 bit/s，趋势图和业务
趋势表为 kbit/s；它们表示密钥生成速率而非累计密钥比特。误差范围是种子
均值间的样本标准差，单种子时不显示；空白表示未定义而非零。
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TRACE_SCHEMA_VERSION = 2
# 七芯专用映射：仿真 0..5 是外圈物理 2..7，仿真 6 是物理中心芯 1。
CORE_TO_PHYSICAL = [2, 3, 4, 5, 6, 7, 1]


class TrafficRecorder:
    """按事件发生后的资源归属记录回放，不参与分配。

    occupancy[有向链路][仿真芯][经典信道] 保存业务 ID，阻塞事件不改变占用。
    固定量子芯/频率单独放在 config 中。物理芯映射只适用于七芯布局。
    """

    def __init__(self, sim, *, seed, warmup):
        if not sim.allocator.bind_three:
            raise ValueError('Business trace requires a three-core experiment allocator')
        self.sim = sim
        self.links = sorted((int(a), int(b)) for a, b in sim.graph.to_directed().edges)
        self.link_index = {tuple(link): i for i, link in enumerate(self.links)}
        self.occupancy = [
            [[-1 if sim.m_resourceMap[a, b, c, w] == 1 else -2
              for w in range(sim.quantum_wave_num, sim.WaveNumber)]
             for c in range(sim.core_num)] for a, b in self.links]
        self.config = dict(
            algorithm='first-fit' if sim.algorithm == 'FF' else sim.algorithm.lower(),
            seed=int(seed), duration=float(sim.Ts), warmup=float(warmup),
            mean_holding_time=float(sim.m_rou1), arrival_rate=float(sim.lambda1),
            offered_load_erlang=float(sim.lambda1 * sim.m_rou1),
            power_dbm=float(10 * math.log10(sim.launch_power / 1e-3)),
            power_reference='per_core_per_classical_channel_at_fiber_input',
            allocation_mode='three_core_bound', traffic_unit='three_core_business_group',
            load_unit='Erlang of three-core business groups',
            arrival_rate_unit='three-core groups per simulation unit',
            direction_definition='forward: source < destination; backward: source > destination; independent resources',
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
        )
        self.initial = deepcopy(self.occupancy)
        self.active = {}
        self.states = []
        self.business_groups = []

    def record(self, event, *, blocked=False):
        """记录每组到达及其真实资源变化；阻塞组无分配，也不生成状态变化。

        业务组和 states 的 group_id 对应；离去沿用到达时的三芯成员。
        每次事件核对回放与真实资源/功率，避免在导出阶段凭空复制三芯占用。
        """
        bid = int(event.m_id)
        arrival = bool(event.m_eventType['Arrival'])
        if arrival:
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
                service['allocation'] = dict(path=path,
                    hops=[dict(link=[a, b], direction='forward' if a < b else 'backward',
                               cores=[int(c) for c in cores],
                               physical_cores=[CORE_TO_PHYSICAL[c] for c in cores])
                          for a, b, cores in zip(path, path[1:], event.m_ocuppiedcore)],
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
        """返回回放；终点仍活跃组标记 horizon 释放，不修改仿真资源和统计。

        scheduled_end_time 保留自然结束计划，release_time 是回放中的释放时间。
        时间为仿真单位，映射实际秒数由使用 JSON 的程序处理。
        """
        self._check_state()
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


def write_trace(path, data):
    """以 UTF-8 写新 JSON，states 每个状态占一行，返回文件 SHA-256。

    使用排他新建模式防止覆盖已有结果，不是文件系统只读保护。
    拒绝 NaN/Infinity，以便其他程序读取；manifest 等字典也可使用此函数。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write('{\n')
        for index, (key, value) in enumerate(data.items()):
            if index:
                stream.write(',\n')
            stream.write(json.dumps(key) + ':')
            if key == 'states':
                stream.write('[\n')
                for i, row in enumerate(value):
                    if i:
                        stream.write(',\n')
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
                stream.write('\n]')
            else:
                stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
        stream.write('\n}\n')
    return sha256(path.read_bytes()).hexdigest()


def configuration_records(metadata):
    """将配置展开为 parameter/value 两列；嵌套结构存为 JSON 文本供 Excel 阅读。"""
    records = []
    for key, value in metadata.items():
        items = [(key, value)] if key != 'physical_configurations' else [
            (f'{key}.{label}', config) for label, config in value.items()]
        for label, config in items:
            records.append(dict(parameter=label, value=json.dumps(config, ensure_ascii=False)
                           if isinstance(config, (dict, list, tuple)) else config))
    return records


def export_excel(path, summary, runs, samples, metadata, save_samples=False, *, summary_tables=None):
    """将已完成统计写为 xlsx，无需安装 Excel，不重新计算科学指标。

    summary_tables 可覆盖默认 Summary 表，Runs/Config 始终写入；仅当
    save_samples 为真时增加 Samples。记录数超过 Excel 行数限制时报错。
    百分比在底层仍存 0..1 数值，None 显示为空白。
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    workbook = Workbook()
    workbook.remove(workbook.active)
    tables = (summary_tables if summary_tables is not None else [('Summary', summary)]) + [
        ('Runs', runs), ('Config', configuration_records(metadata))]
    if save_samples:
        tables.append(('Samples', samples))
    for name, records in tables:
        if len(records) > 1_048_575:
            raise ValueError(f'{name} exceeds Excel row limit; omit --save-samples (JSON retains samples)')
        sheet = workbook.create_sheet(name)
        sheet.sheet_view.showGridLines = False
        headers = list(records[0])
        if name == 'Summary':
            first = ['scenario', 'offered_load_erlang', 'algorithm', 'skr_mean', 'skr_mean_sd',
                     'gain_vs_FF', 'gain_vs_SCWA', 'blocking_rate', 'carried_load_erlang',
                     'fallback_rate', 'zero_skr_fraction_mean', 'seed_count']
            headers = [key for key in first if key in headers] + [key for key in headers if key not in first]
        sheet.append(headers)
        for record in records:
            sheet.append([record.get(key) for key in headers])
        sheet.freeze_panes = 'D2' if name != 'Config' else 'A2'
        for cell in sheet[1]:
            cell.font = Font(name='Arial', bold=True, color='FFFFFF', size=10)
            cell.fill = PatternFill('solid', fgColor='254B72')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        sheet.row_dimensions[1].height = 42
        for column, key in enumerate(headers, 1):
            width = 28 if key == 'algorithm' else min(30, max(15, len(key) + 2))
            if name == 'Config':
                width = 32 if column == 1 else 115
            sheet.column_dimensions[get_column_letter(column)].width = width
            for cells in sheet.iter_cols(min_col=column, max_col=column, min_row=2):
                for cell in cells:
                    cell.font = Font(name='Arial', size=10)
                    cell.alignment = Alignment(vertical='center', wrap_text=name == 'Config')
                    if isinstance(cell.value, float):
                        is_rate = any(x in key for x in ('gain_vs_', 'blocking_rate', 'fallback_rate',
                                                       'zero_skr_fraction', 'channel_utilization'))
                        cell.number_format = ('0.00%' if is_rate else
                                              '0.000E+00' if '_w_' in key or key.startswith('noise_counts') else
                                              '#,##0.000')
        if name == 'Config':
            for row in sheet.iter_rows(min_row=2):
                sheet.row_dimensions[row[0].row].height = max(22, 15 * math.ceil(len(str(row[1].value)) / 105))
        table = Table(displayName=name + 'Table', ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
        sheet.add_table(table)
    workbook.save(path)


def plot_scan(output, summary):
    """按场景画 SKR、阻塞率、承载量、噪声和增益，返回生成的文件名列表。

    只使用 summary 的均值和标准差，不从回放重算。40% 增益虚线是预设研究
    参考目标，不代表实测达到，也不是算法约束；诊断图区分回退率与零 SKR 比例。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {'FF': '#D55E00', 'SCWA': '#0072B2', 'GREEDY_MIN_NOISE': '#009E73',
              'CQLI': '#E69F00'}
    frame = pd.DataFrame(summary)
    artifacts = []
    for index, (scenario, group) in enumerate(frame.groupby('scenario', sort=False), 1):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
        specs = [('skr_mean', 'Usable SKR (kbit/s)', .001),
                 ('blocking_rate', 'Blocking probability', 1),
                 ('carried_load_erlang', 'Carried traffic (Erlang)', 1),
                 ('fwm_w_mean', 'FWM at quantum receivers (pW)', 1e12),
                 ('raman_w_mean', 'Raman at quantum receivers (pW)', 1e12)]
        for ax, (metric, label, factor) in zip(axes.flat, specs):
            for algorithm, data in group.groupby('algorithm', sort=False):
                data = data.sort_values('offered_load_erlang')
                x, y = data['offered_load_erlang'].to_numpy(), data[metric].to_numpy(dtype=float)*factor
                sd = data[metric + '_sd'].to_numpy(dtype=float)*factor
                color = colors[algorithm]
                ax.plot(x, y, marker='o', ms=4, lw=1.7, label=algorithm, color=color)
                if np.isfinite(sd).any():
                    ax.fill_between(x, y - sd, y + sd, color=color, alpha=.12)
            ax.set_ylabel(label)
        axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, 0].legend(fontsize=8)
        gain = axes[1, 2]
        new = group[group.algorithm == 'GREEDY_MIN_NOISE'].sort_values('offered_load_erlang')
        for baseline, marker in [('FF', 'o'), ('SCWA', 's')]:
            if len(new) and new['gain_vs_' + baseline].notna().any():
                gain.plot(new.offered_load_erlang, new['gain_vs_' + baseline], marker=marker,
                          label='vs ' + baseline, color=colors[baseline])
        gain.axhline(.4, color='#666666', ls='--', lw=1, label='40% target vs FF')
        gain.axhline(0, color='#AAAAAA', lw=.7)
        gain.set_ylabel('GREEDY_MIN_NOISE relative SKR gain')
        gain.yaxis.set_major_formatter(PercentFormatter(1))
        gain.legend(fontsize=8)
        for ax in axes.flat:
            ax.set_xlabel('Offered traffic (Erlang)')
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        seeds = int(group.seed_count.iloc[0])
        fig.suptitle(f'{scenario} | {seeds} seed(s) | shaded: across-seed SD', fontsize=13)
        for suffix in ('png', 'svg'):
            path = output / f'traffic_scan_{index}.{suffix}'
            fig.savefig(path, dpi=180)
            artifacts.append(path.name)
        plt.close(fig)
    # 两个基准分别使用纵轴，便于看清数量级不同的增益。
    new = frame[frame.algorithm == 'GREEDY_MIN_NOISE']
    if len(new) and new[['gain_vs_FF', 'gain_vs_SCWA']].notna().any().any():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for ax, baseline in zip(axes, ('FF', 'SCWA')):
            for scenario, data in new.groupby('scenario', sort=False):
                data = data.sort_values('offered_load_erlang')
                if data['gain_vs_' + baseline].notna().any():
                    ax.plot(data.offered_load_erlang, data['gain_vs_' + baseline], marker='o', label=scenario)
            ax.axhline(0, color='#AAAAAA', lw=.8)
            if baseline == 'FF':
                ax.axhline(.4, color='#666666', ls='--', lw=1, label='40% target')
            ax.set(xlabel='Offered traffic (Erlang)', ylabel='SKR gain vs ' + baseline)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.suptitle('GREEDY_MIN_NOISE: ratio of across-seed mean SKR')
        for suffix in ('png', 'svg'):
            path = output / f'traffic_gains.{suffix}'
            fig.savefig(path, dpi=180)
            artifacts.append(path.name)
        plt.close(fig)
    # 回退比例单独绘制，不能当作物理 FWM 功率。
    new = frame[frame.algorithm == 'GREEDY_MIN_NOISE']
    if len(new):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for scenario, group in new.groupby('scenario', sort=False):
            group = group.sort_values('offered_load_erlang')
            for ax, key in zip(axes, ('fallback_rate', 'zero_skr_fraction_mean')):
                ax.plot(group.offered_load_erlang, group[key], marker='o', label=scenario)
        for ax, label in zip(axes, ('FWM fallback / admitted arrivals', 'Zero-SKR channel samples / all channel samples')):
            ax.set(xlabel='Offered traffic (Erlang)', ylabel=label)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, max(.01, float(new.zero_skr_fraction_mean.max()) * 1.1))
        for suffix in ('png', 'svg'):
            path = output / f'traffic_diagnostics.{suffix}'
            fig.savefig(path, dpi=180)
            artifacts.append(path.name)
        plt.close(fig)
    return artifacts


def export_business_summary(output, runs, metadata, summary):
    """用本批统计生成一份 Excel 和六张 PNG，返回相对输出目录的文件名。

    每组输出 SKR/阻塞率趋势，另有两组并排的总览图。Excel 使用可编辑的
    数值横轴图，标准差列保留在表里；PNG 显示误差棒。重合曲线不人为平移。
    阻塞率图不画阈值线。每张 SKR 图只标注 greedy 相对 FF 的最大正提升：
    (greedy 的种子均值 / FF 的种子均值 - 1)。FF 非正或任一值缺失时不计算，
    无正提升则不标；同分选横坐标较小的点。PNG 单图、总览和 Excel 共用该选择。
    标注由本次数据自动生成，不写死负载或百分比。
    """
    import openpyxl
    from openpyxl.chart import Reference, ScatterChart, Series
    from openpyxl.chart.label import DataLabelList
    from openpyxl.chart.legend import LegendEntry
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    specs = [
        ('load_scan', 'LoadSweep', 'load_erlang', 'Three-core group traffic (Erlang)',
         f"Load sweep | {metadata['fixed_power_dbm']:g} dBm/core/channel"),
        ('power_scan', 'PowerSweep', 'power_dbm', 'Power per core per channel (dBm)',
         f"Power sweep | {metadata['fixed_load_erlang']:g} Erlang of three-core groups")]
    series = [('ff', 'first-fit', '2563EB', 'o', '-'),
              ('greedy', 'greedy_min_noise', 'D97706', 's', '--')]
    path = output / 'skr_summary.xlsx'
    export_excel(path, [], runs, [], metadata,
                 summary_tables=[(sheet, summary[group]) for group, sheet, *_ in specs])
    workbook = openpyxl.load_workbook(path)
    artifacts = ['skr_summary.xlsx']
    note = (f"10 km bidirectional | slots={metadata['slots']}, warmup={metadata['warmup']} | "
            f"{len(metadata['seeds'])} seed(s); " +
            ('error bars: across-seed SD' if len(metadata['seeds']) > 1 else 'single-seed trend'))
    metrics = [('skr_kbit_s', 'skr_sd_kbit_s', 'Mean total SKR (kbit/s)', 'skr_trends.png', 'total_skr.png'),
               ('blocking_rate', 'blocking_rate_sd', 'Classical blocking probability',
                'blocking_trends.png', 'blocking_rate.png')]
    for metric_index, (metric, sd_key, y_label, overview, filename) in enumerate(metrics):
        blocking = metric == 'blocking_rate'
        upper = max((r[prefix + '_' + metric] + (r[prefix + '_' + sd_key] or 0)
                     for rows in summary.values() for r in rows for prefix, *_ in series
                     if r[prefix + '_' + metric] is not None), default=0)
        upper = min(1, max(.01, upper) * 1.15) if blocking else max(1, upper * 1.08)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, (group, sheet_name, x_key, x_label, title) in zip(axes, specs):
            sheet = workbook[sheet_name]
            headers = [cell.value for cell in sheet[1]]
            chart = ScatterChart()
            chart.title, chart.x_axis.title, chart.y_axis.title = title, x_label, y_label
            chart.scatterStyle = 'lineMarker'
            chart.width, chart.height = 25, 13
            chart.y_axis.scaling.min, chart.y_axis.scaling.max = 0, upper
            x_values = Reference(sheet, min_col=headers.index(x_key) + 1, min_row=2, max_row=sheet.max_row)
            single, single_ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
            for prefix, label, color, marker, style in series:
                points = [r for r in summary[group] if r[prefix + '_' + metric] is not None]
                if not points:
                    continue
                values = Reference(sheet, min_col=headers.index(prefix + '_' + metric) + 1,
                                   min_row=2, max_row=sheet.max_row)
                curve = Series(values, x_values, title=label)
                curve.graphicalProperties.line.solidFill = color
                curve.graphicalProperties.line.prstDash = 'dash' if style == '--' else 'solid'
                curve.marker.symbol, curve.marker.size = ('square', 7) if marker == 's' else ('circle', 5)
                curve.marker.graphicalProperties.noFill = marker == 's'
                if marker != 's':
                    curve.marker.graphicalProperties.solidFill = color
                curve.marker.graphicalProperties.line.solidFill = color
                chart.series.append(curve)
                x = [r[x_key] for r in points]
                y = [r[prefix + '_' + metric] for r in points]
                sd = [r[prefix + '_' + sd_key] for r in points]
                for target in (ax, single_ax):
                    target.plot(x, y, marker=marker, ms=7 if marker == 's' else 5,
                                markerfacecolor='none' if marker == 's' else '#' + color,
                                ls=style, lw=1.8, label=label, color='#' + color)
                    if all(v is not None for v in sd):
                        target.errorbar(x, y, yerr=sd, fmt='none', capsize=3, color='#' + color)
            if blocking:
                chart.y_axis.numFmt = '0%'
                for target in (ax, single_ax):
                    target.yaxis.set_major_formatter(PercentFormatter(1))
                    if all(r['ff_blocking_rate'] is not None and r['ff_blocking_rate'] == r['greedy_blocking_rate']
                           for r in summary[group]):
                        target.text(.03, .95, 'Algorithm curves overlap', transform=target.transAxes,
                                    va='top', fontsize=9, color='#555555')
            else:
                comparable = [(index, row) for index, row in enumerate(summary[group])
                              if row['ff_skr_kbit_s'] is not None and row['ff_skr_kbit_s'] > 0
                              and row['greedy_skr_kbit_s'] is not None
                              and row['greedy_skr_kbit_s'] > row['ff_skr_kbit_s']]
                if comparable:
                    index, best = max(comparable, key=lambda pair:
                        (pair[1]['greedy_skr_kbit_s'] / pair[1]['ff_skr_kbit_s'] - 1, -pair[1][x_key]))
                    x = best[x_key]
                    low, high = best['ff_skr_kbit_s'], best['greedy_skr_kbit_s']
                    label = f'Max SKR gain: +{(high / low - 1):.1%}'
                    # 两端锚定同一工况下的真实均值；文字向图内偏移，避免边界裁切。
                    midpoint = (min(r[x_key] for r in summary[group])
                                + max(r[x_key] for r in summary[group])) / 2
                    right_half = x > midpoint
                    for target in (ax, single_ax):
                        target.annotate('', xy=(x, high), xytext=(x, low),
                                        arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
                        target.annotate(label, xy=(x, (low + high) / 2),
                                        xytext=(-10 if right_half else 10, 0), textcoords='offset points',
                                        ha='right' if right_half else 'left', va='center', fontsize=9,
                                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
                    # Excel 用一个不可见的单点系列承载同一百分比标签，隐藏它的图例项。
                    excel_row = index + 2
                    annotation = Series(Reference(sheet, min_col=headers.index('greedy_skr_kbit_s') + 1,
                                                  min_row=excel_row, max_row=excel_row),
                                        Reference(sheet, min_col=headers.index(x_key) + 1,
                                                  min_row=excel_row, max_row=excel_row), title=label)
                    annotation.graphicalProperties.line.noFill = True
                    annotation.marker.symbol = 'none'
                    annotation.dLbls = DataLabelList(showSerName=True, showVal=False,
                                                    showLegendKey=False, dLblPos='b')
                    chart.legend.legendEntry = [LegendEntry(idx=len(chart.series), delete=True)]
                    chart.series.append(annotation)
            sheet.add_chart(chart, f'A{sheet.max_row + 4 + 28 * metric_index}')
            for target in (ax, single_ax):
                target.set(xlabel=x_label, ylabel=y_label, title=title, ylim=(0, upper))
                target.grid(alpha=.2)
                target.spines[['top', 'right']].set_visible(False)
                target.legend(fontsize=9, loc='lower right' if blocking else 'best')
            single.suptitle(note, fontsize=8)
            figure_path = f'{group}/{filename}'
            single.savefig(output / figure_path, dpi=180)
            plt.close(single)
            artifacts.append(figure_path)
        fig.suptitle(note, fontsize=10)
        fig.savefig(output / overview, dpi=180)
        plt.close(fig)
        artifacts.append(overview)
    workbook.save(path)
    workbook.close()
    return artifacts


def export_scan_results(output, data, save_samples=False):
    """输出 traffic_scan.json、xlsx 和 PNG/SVG；JSON 始终保留 samples，不重算统计。"""
    (output / 'traffic_scan.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    export_excel(output / 'traffic_scan.xlsx', data['summary'], data['runs'],
                 data['samples'], data['config'], save_samples)
    return plot_scan(output, data['summary'])
