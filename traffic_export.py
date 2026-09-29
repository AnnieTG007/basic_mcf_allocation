"""由 main 或 traffic_scan 调用，将仿真状态及统计结果导出为回放 JSON、Excel 和图表。
输出目录由调用方指定，本模块不生成业务、执行分配或重算统计。

业务回放 JSON 只告诉实验端"哪些纤芯、某时刻占用哪些信道"，格式如下：

    {
      "forward_cores": [3, 4, 5],
      "backward_cores": [6, 7, 1],
      "quantum_frequencies_hz": [193.5e12],
      "classical_frequencies_hz": [194.0e12, 193.9e12, ...],
      "duration_s": 3600,
      "states": [
        {"time_s": 0, "forward_channels": [], "backward_channels": []},
        {"time_s": 10, "forward_channels": [4, 6, 9], "backward_channels": [1]}
      ]
    }

约定：占用信道编号从 0 开始，是 classical_frequencies_hz 的下标；前向指节点号由小到大。
quantum_frequencies_hz 单独记录量子频率，经典候选列表保持仿真顺序，不按实际占用筛选。
纤芯从 1 编号：中心为 1，外围从顶部顺时针为 2–7。
forward_cores/backward_cores 给出本次仿真实际使用的方向纤芯，供实验端与自身设置核对是否一致。
某一时刻未出现在列表里的信道即该时刻不应有波长。states 按事件发生顺序排列，同时刻事件保持原顺序。
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class TrafficRecorder:
    """观察已完成事件并记录所选链路的资源归属，生成三芯业务回放。"""

    def __init__(self, sim):
        """接收三芯实例 sim，按其观测链路的前后向芯建立占用记录。"""
        if not sim.bind_three:
            raise ValueError('业务回放要求三芯实验实例')
        self.sim = sim
        a, b = sim.observed_link
        self.links = [(a, b), (b, a)]
        self.link_index = {tuple(link): i for i, link in enumerate(self.links)}
        # occupancy[有向链路][仿真芯][经典信道]：-2 不可用、-1 空闲、非负整数为业务编号。
        self.occupancy = [
            [[-1 if sim.m_resourceMap[a, b, c, w] == 1 else -2
              for w in range(sim.quantum_wave_num, sim.WaveNumber)]
             for c in range(sim.core_num + 1)] for a, b in self.links]
        # active 按业务编号保存已接入业务，供离去事件找回分配；trace 保存占用状态变化序列。
        self.active = {}
        self.trace = []
        # 仿真从全空闲开始，显式记录起点，实验端无需假设初始占用状态。
        self.trace.append(self._state(0.0, event_kind='initial'))

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
        if not blocked:
            if arrival:
                # 到达先登记本次占用，离去事件按业务编号取回同一份芯组与信道。
                self.active[bid] = dict(
                    cores=[list(cores) for cores in event.m_ocuppiedcore], wave=int(event.m_ocuppiedwave))
                self._update(bid, self.active[bid], 'arrival')
            else:
                self._update(bid, self.active[bid], 'leave')
                self.active.pop(bid)
        # 阻塞到达不改变占用，但要记录该时刻的状态，回放才能反映阻塞发生的时间点。
        # 阻塞到达不改变占用，但仍要留下该时刻的占用状态；占用未变化时 _state 返回 None。
        state = self._state(event.m_time, event_kind='blocked' if blocked else 'change')
        if state is not None:
            self.trace.append(state)
        self._check_state()

    def _update(self, bid, service, kind):
        """回放三芯归属一起更新；到达须空闲，离去须属于同一组，否则报错。"""
        wave = service['wave'] - self.sim.quantum_wave_num
        for cores in service['cores']:
            for core in cores:
                row = self.occupancy[self._direction(core)][core]
                expected = -1 if kind == 'arrival' else bid
                if row[wave] != expected:
                    raise ValueError('回放归属与仿真生命周期不一致')
                row[wave] = bid if kind == 'arrival' else -1

    def _direction(self, core):
        """返回该芯所属的观测链路下标：0 为前向（节点号小到大），1 为后向。"""
        return 0 if core in self.sim.classical_forward_cores else 1

    def _carried(self, link_index):
        """返回该方向上当前被业务占用的信道编号列表，按编号升序；芯号不导出。

        占用状态记录在 occupancy[有向链路][芯][信道]，同一方向的绑定芯必须同占用，
        故任一成员芯给出同一信道列表。"""
        return sorted({wave for core in self.occupancy[link_index]
                       for wave, owner in enumerate(core) if owner >= 0})

    def _state(self, time, *, event_kind):
        """按当前占用生成一条回放状态；event_kind 仅用于说明该状态由哪类事件产生。

        与上一条状态占用相同则返回 None，调用方据此跳过，回放只保留实际发生变化的时刻。"""
        forward, backward = self._carried(0), self._carried(1)
        if self.trace and (forward, backward) == (self.trace[-1]['forward_channels'],
                                                  self.trace[-1]['backward_channels']):
            return None
        return dict(time_s=float(time), event=event_kind,
                    forward_channels=forward, backward_channels=backward)

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
                raise ValueError('三芯回放与实际资源或每芯功率不一致')

    def build_trace(self):
        """返回回放字典：方向纤芯、信道频率表、总时长及逐事件占用状态；不修改仿真资源或指标。

        在业务全部离去前调用时，仍会被本方法补记终点状态：终点强制结束回放，
        计划离去时间可超过终点，不能解释成自然离去。"""
        self._check_state()
        if self.active:
            for bid, service in self.active.items():
                self._update(bid, service, 'leave')
            final = self._state(self.sim.Ts, event_kind='horizon_release')
            if final is not None:
                self.trace.append(final)
            self.active.clear()
        return dict(forward_cores=list(self.sim.classical_forward_cores),
                    backward_cores=list(self.sim.classical_backward_cores),
                    quantum_frequencies_hz=[float(f) for f in self.sim.available_channel[:self.sim.quantum_wave_num]],
                    classical_frequencies_hz=[float(f) for f in self.sim.available_channel[self.sim.quantum_wave_num:]],
                    duration_s=float(self.sim.Ts),
                    states=self.trace)


def rescale_times(trace, duration_seconds):
    """把回放按实验总时长缩放为秒制副本：时间乘系数，占用顺序与信道编号不变。

    duration_seconds 例如 3600；总时长与各状态时间同步缩放，业务与占用关系不重排。"""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('实验总时长必须为有限正数（秒）')
    current = trace['duration_s']
    if not math.isfinite(current) or current <= 0:
        raise ValueError('原回放时长必须为有限正数')
    factor = duration_seconds / current
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('时间换算系数超出数值范围')
    result = deepcopy(trace)
    result['duration_s'] = float(duration_seconds)
    for state in result['states']:
        state['time_s'] = state['time_s'] * factor
    return result


def write_trace(path, trace):
    """以 UTF-8 写入回放 JSON 并返回 SHA-256 文件摘要；拒绝覆盖已有文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(trace, ensure_ascii=False, allow_nan=False) + '\n'
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
               ('blocking_rate', '阻塞率', 1, '0.00%'),
               ('synergy_vs_FF', '协同度', 1, '0.000000')]
    ratio_keys = ('skr_mean', 'osnr_linear_mean', 'blocking_rate', 'synergy_vs_FF')
    for name, records, conditions in tables:
        if len(records) > 1_048_575:
            raise ValueError(f'{name} 超过 Excel 行数上限')
        sheet = workbook.create_sheet(name)
        headers = [label for _, label in conditions] + ['算法']
        headers += [label for _, label, _, _ in metrics]
        headers += ['SKR / FF', 'OSNR / FF（线性）', '阻塞率 / FF', '协同度 / FF']
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
            sheet.column_dimensions[get_column_letter(column)].width = 26 if label == '算法' else 20
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
    """将业务汇总导出为 LoadSweep/PowerSweep 工作表及四指标的 SVG 图，返回文件名列表。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    specs = [
        ('load_scan', '负载扫描', 'load_erlang', 'Three-core group traffic (Erlang)',
         f"Load sweep | {metadata['fixed_power_dbm']:g} dBm/core/channel"),
        ('power_scan', '功率扫描', 'power_dbm', 'Power per core per channel (dBm)',
         f"Power sweep | {metadata['fixed_load_erlang']:g} Erlang of three-core groups")]
    algorithms = list(dict.fromkeys(r['algorithm'] for r in runs))
    colors = {name: plt.get_cmap('tab10')(i % 10) for i, name in enumerate(algorithms)}
    # 工作簿只保留两张汇总表，不嵌图；完整配置、种子与样本仍在 JSON 中。
    path = output / 'skr_summary.xlsx'
    tables = [(sheet, summary[group], [('load_erlang', '负载 (Erlang)'), ('power_dbm', '功率 (dBm)')])
              for group, sheet, *_ in specs]
    export_excel(path, tables)
    artifacts = ['skr_summary.xlsx']
    note = (f"{runs[0]['observed_length_m'] / 1000:g} km bidirectional | slots={metadata['slots']}, warmup={metadata['warmup']} | "
            f"{len(metadata['seeds'])} seed(s); " +
            ('error bars: across-seed SD' if len(metadata['seeds']) > 1 else 'single-seed trend'))
    metrics = [('osnr_db_mean', 'osnr_db_mean_sd', 'Reference link OSNR (dB)', 'osnr_trends.svg', 'classical_osnr.svg'),
               ('synergy_vs_FF', 'synergy_vs_FF_sd', 'Signed synergy vs FF', 'synergy_trends.svg', 'synergy.svg'),
               ('skr_mean', 'skr_mean_sd', 'Mean link SKR per channel (kbit/s)', 'skr_trends.svg', 'total_skr.svg'),
               ('blocking_rate', 'blocking_rate_sd', 'Classical blocking probability',
                'blocking_trends.svg', 'blocking_rate.svg')]
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
            single.savefig(output / figure_path)
            plt.close(single)
            artifacts.append(figure_path)
        fig.suptitle(note, fontsize=10)
        fig.savefig(output / overview)
        plt.close(fig)
        artifacts.append(overview)
    return artifacts


def export_scan_results(output, data):
    """将扫描汇总导出为 traffic_scan.xlsx 和四指标对比 SVG，不重算统计。"""
    export_excel(output / 'traffic_scan.xlsx', [('汇总', data['summary'], [
        ('offered_load_erlang', '负载 (Erlang)'), ('power_dbm', '功率 (dBm)'),
        ('observed_length_m', '链路长度 (m)')])])
    return plot_scan(output, data['summary'], data['config'].get('scan_axis', 'load'))
