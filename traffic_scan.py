"""由 main.py 的扫描或业务导出入口调用，按给定参数与实例工厂编排实验。
复用实例事件循环采样并汇总各随机种子，结果交给 traffic_export 写文件。"""
from argparse import Namespace
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import math
import platform
import time

import numpy as np
import pandas as pd

from noise_calculation import raman_model_config
from skr_calculation import skr_model_config, synergy_skr_bounds
from synergistic_calculation import osnr_db, add_paired_synergy, METRIC_DEFINITIONS
from traffic_export import (TrafficRecorder, export_business_summary,
                            export_scan_results, export_replay_timing, write_trace)


METRICS = ("skr", "raw_skr", "no_fwm_skr", "zero_noise_skr", "raman_w",
           "fwm_w", "noise_counts", "active_services", "occupied_channels",
           "zero_skr_fraction", "osnr_linear", "zero_xt_channels",
           "classical_received_power_w", "classical_noise_power_w", "classical_noise_per_channel_w",
           "classical_xt_w", "classical_floor_w")
GROUP = ["scenario", "offered_load_erlang", "algorithm"]


def measure_run(sim, warmup, event_recorder=None):
    """运行全新实例 sim，返回本种子指标字典及预热后时隙末样本。
可选 event_recorder 接收全部资源变化，warmup 使用仿真时间单位。"""
    if not 0 <= warmup < sim.Ts:
        raise ValueError("warmup 必须满足 0 <= warmup < slots")
    # 观测窗口为 [warmup, sim.Ts)，总时隙包含预热；例如 warmup=10、Ts=30。
    links = sorted((min(a, b), max(a, b)) for a, b in sim.graph.edges)
    capacity = int(np.count_nonzero(sim.m_resourceMap == 1))
    # 禁止双向同芯同频时，共享资源格仅计一次；容量与利用率均按全网统计。
    if not sim.allow_bidirectional:
        capacity -= sum(int(np.count_nonzero((sim.m_resourceMap[a, b] == 1)
                                            & (sim.m_resourceMap[b, a] == 1)))
                        for a, b in links)
    offered = blocked_count = active = 0
    carried_time = occupied_time = 0.0
    digest = sha256()
    samples = []
    started = time.perf_counter()

    def observe_event(event, *, blocked):
        """记录到达序列摘要、窗口计数及占用时长；blocked 表示本次到达是否被阻塞。"""
        nonlocal offered, blocked_count, active, carried_time, occupied_time
        arrival = bool(event.m_eventType['Arrival'])
        measured = event.m_time >= warmup
        if arrival:
            digest.update(repr((event.m_id, event.m_time, event.m_holdTime,
                                int(event.m_sourceNode), int(event.m_destNode))).encode())
        if event_recorder is not None:
            event_recorder.record(event, blocked=blocked)
        if arrival:
            if measured:
                offered += 1
                blocked_count += int(blocked)
            if not blocked:
                active += 1
                # 已接入业务与窗口求交，包含预热期间到达但在窗口内仍活跃的业务。
                duration = max(0.0, min(sim.Ts, event.m_time + event.m_holdTime)
                               - max(warmup, event.m_time))
                carried_time += duration
                # 逐跳累加实际占用芯数：单芯业务每跳一芯，三芯业务每跳三芯。
                occupied_time += duration * sum(np.size(cores) for cores in event.m_ocuppiedcore)
        else:
            active -= 1

    # 每个预热后时隙末采样；不另造事件循环或修改资源。
    for slot in sim.iter_slots(on_event=observe_event):
        if slot < warmup:
            continue
        totals = {key: 0.0 for key in METRICS[:7]}
        quantum_count = zero_count = 0
        for a, b in (sim.observed_link,):
            metrics = sim.quantum_scorer.metrics(
                sim.m_resourceMap[a, b], sim.m_resourceMap[b, a],
                sim.P_link[a, b], sim.P_link[b, a], sim.a_m[a, b])
            for key in totals:
                totals[key] += metrics[key]
            quantum_count += metrics['quantum_channels']
            zero_count += metrics['zero_skr_channels']
        # 秘密密钥率（SKR，bit/s）已逐量子信道截零；raw_skr 仅作诊断，不用于增益或协同度。
        for key in ('skr', 'raw_skr', 'no_fwm_skr', 'zero_noise_skr'):
            totals[key] /= quantum_count
        totals.update(time=slot + 1, active_services=active,
                      occupied_channels=0,
                      zero_skr_fraction=zero_count / quantum_count)
        if not all(math.isfinite(value) for value in totals.values()):
            raise ValueError("Non-finite simulation metric")
        # 光信噪比（OSNR）只测所选链路：平均接收信号 /（平均串扰 + 固定噪声底）。
        totals['osnr_linear'] = sim.measure_classical_osnr()
        classical = sim.classical_osnr_scorer.statistics
        totals['zero_xt_channels'] = classical['zero_noise_channels']
        # occupied_channels 只数观测链路资源格，active_services 则是全网业务数。
        totals['occupied_channels'] = classical['occupied_channels']
        totals['classical_received_power_w'] = classical['signal_sum_w']
        totals['classical_noise_power_w'] = classical['noise_sum_w']
        totals['classical_noise_per_channel_w'] = classical['noise_per_channel_w']
        totals['osnr_db'] = osnr_db(totals['osnr_linear'])
        totals['classical_xt_w'] = classical['xt_sum_w']
        totals['classical_floor_w'] = classical['floor_sum_w']
        samples.append(totals)
    accepted = offered - blocked_count
    # 非空时隙的 OSNR 在线性域等权平均后转 dB；SKR 包含空闲时隙，静态脉冲块不随时隙换算。
    row = {}
    for key in METRICS:
        values = [s[key] for s in samples if s[key] is not None]
        row[f'{key}_mean'] = float(np.mean(values)) if values else None
    row['osnr_idle_slots'] = sum(s['occupied_channels'] == 0 for s in samples)
    row['osnr_db_mean'] = osnr_db(row['osnr_linear_mean'])
    row['osnr_valid_samples'] = sum(s['osnr_linear'] is not None for s in samples)
    row['observed_link'] = list(sim.observed_link)
    row['observed_length_m'] = float(sim.a_m[sim.observed_link])
    row['observed_quantum_channels'] = quantum_count
    _, quantum_indices = np.nonzero(sim.m_resourceMap[sim.observed_link] == 3)
    row.update(synergy_skr_bounds(
        row['observed_length_m'],
        len(set(sim.classical_forward_cores) | set(sim.classical_backward_cores)),
        sim.launch_power, np.asarray(sim.available_channel)[quantum_indices],
        sim.bb84_params, sim.detector_params, sim.noise_model.first_fiber))
    # 无到达时阻塞率未定义；承载量按窗口时长归一，利用率还需除以全网物理容量。
    row.update(offered=offered, accepted=accepted, blocked=blocked_count,
               blocking_rate=blocked_count / offered if offered else None,
               carried_load_erlang=carried_time / (sim.Ts - warmup),
               channel_utilization=occupied_time / (sim.Ts - warmup) / capacity,
               capacity=capacity, samples=len(samples),
               traffic_sha256=digest.hexdigest(), runtime_s=time.perf_counter() - started)
    return row, samples


def summarize(runs):
    """按场景、负载和算法分组，等权汇总各独立种子的指标及样本标准差。"""
    frame = pd.DataFrame(runs)
    statistics = [f"{key}_mean" for key in METRICS] + [
        "blocking_rate", "carried_load_erlang", "channel_utilization",
        "osnr_db_mean", "synergy_vs_FF",
        "delta_osnr_linear_vs_FF", "delta_skr_vs_FF"]
    rows = []
    for keys, group in frame.groupby(GROUP, sort=False):
        row = dict(zip(GROUP, keys))
        rtol = group.iloc[0]['qcnm_noise_rtol']
        row['qcnm_noise_rtol'] = float(rtol) if pd.notna(rtol) else None
        row.update(observed_link=group.iloc[0]['observed_link'], observed_length_m=group.iloc[0]['observed_length_m'], seed_count=len(group), arrival_rate=group.iloc[0]['arrival_rate'],
                   length_km=group.iloc[0]['length_km'], power_dbm=float(group.iloc[0]['power_dbm']))
        # 各种子等权，dB 指标直接平均种子 dB 值；协同度已在同种子内相对 FF 配对计算。
        for key in statistics:
            values = group[key].dropna()
            # OSNR/协同度任一种子缺失时整组留空，不用剩余种子代替。
            if key in ('osnr_linear_mean', 'osnr_db_mean', 'synergy_vs_FF',
                       'delta_osnr_linear_vs_FF') and len(values) != len(group):
                values = values.iloc[:0]
            row[key] = float(values.mean()) if len(values) else None
            # _sd 为种子均值间的样本标准差，不是置信区间；单种子留空。
            row[key + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    # 增益是跨种子平均 SKR 的比值减一；缺基准或基准非正时留空。
    lookup = {(r['scenario'], r['offered_load_erlang'], r['algorithm']): r for r in rows}
    for row in rows:
        for baseline in ('FF', 'SCWA'):
            base = lookup.get((row['scenario'], row['offered_load_erlang'], baseline))
            row['gain_vs_' + baseline] = (row['skr_mean'] / base['skr_mean'] - 1
                if base is not None and base['skr_mean'] > 0 else None)
    return rows


def source_hashes(base, topology):
    """为本次运行计算九个源码文件（含内置拉曼谱）和所用拓扑的 SHA-256 摘要；不读取旧批次。"""
    names = ('main.py', 'traffic_scan.py', 'traffic_export.py', 'algorithm.py',
             'noise_calculation.py', 'skr_calculation.py', 'core_layout.py', 'topology.py',
             'synergistic_calculation.py')
    paths = [base / name for name in names]
    paths += [base / 'topologies' / f'{topology}.json']
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def scan_settings(args):
    """校验命令行扫描参数，返回升序负载、种子、预热时长和场景列表。"""
    # 普通扫描负载以 Erlang 计：负载轴默认 30，功率/距离轴固定负载默认 10。
    axis = getattr(args, 'scan_axis', 'load')
    loads = (args.loads if args.loads is not None else [30]) if axis == 'load' else [
        10 if args.fixed_load is None else args.fixed_load]
    seeds = args.seeds if args.seeds is not None else [args.seed]
    warmup = args.warmup if args.warmup is not None else 10
    if not loads or any(not math.isfinite(v) or v <= 0 for v in loads):
        raise ValueError('扫描负载必须为有限正数')
    if not seeds or len(loads) != len(set(loads)) or len(seeds) != len(set(seeds)):
        raise ValueError('负载和随机种子不能重复')
    if not math.isfinite(args.holding_time) or args.holding_time <= 0:
        raise ValueError('holding-time 必须为有限正数')
    if not 0 <= warmup < args.slots:
        raise ValueError('必须满足 0 <= warmup < slots')
    # 场景为 (全边长度 km, 每芯每信道功率 dBm)，如 (10, 10.5)；长度 None 保留拓扑边长。
    scenarios = []
    if axis == 'power':
        powers = args.powers if args.powers is not None else [7, 8, 9, 10, 10.5]
        scenarios = [(args.link_length_km, power) for power in sorted(powers)]
    elif axis == 'distance':
        distances = args.distances if args.distances is not None else [1, 5, 10, 20, 30, 40, 50]
        scenarios = [(length, args.launch_power_dbm) for length in sorted(distances)]
    # 显式场景逐项配对长度和功率，不取笛卡尔积。
    elif args.scan_scenarios:
        if args.link_length_km is not None:
            raise ValueError('--scan-scenarios 与 --link-length-km 不能同时使用')
        for text in args.scan_scenarios:
            try:
                length, power = map(float, text.split(':'))
            except ValueError as exc:
                raise ValueError('场景格式为 KM:DBM，例如 1:13.5 10:10.5') from exc
            scenarios.append((length, power))
    else:
        scenarios.append((args.link_length_km, args.launch_power_dbm))
    for length, power in scenarios:
        if length is not None and (not math.isfinite(length) or length <= 0):
            raise ValueError('链路长度必须为有限正数')
        if not math.isfinite(power) or not -3000 < power < 3000:
            raise ValueError('发射功率必须有限且可转换成正的瓦特数')
    if len(scenarios) != len(set(scenarios)):
        raise ValueError('扫描场景不能重复')
    return sorted(loads), seeds, warmup, scenarios


def run_traffic_scans(args, build_simulation, base, *, variants):
    """按所选维度分别运行扫描，返回各组结果；多组输出分别放入对应子目录。"""
    # 各轴独立运行，不取负载×功率×距离的笛卡尔积。
    axes = [axis for axis in ('load', 'power', 'distance') if getattr(args, 'scan_' + axis)]
    if args.scan_scenarios and axes != ['load']:
        raise ValueError('--scan-scenarios 仅用于单独负载扫描；多维扫描请指定 --powers/--distances')
    # 距离扫描禁止单边覆盖，避免横轴变化却未改变观测距离。
    if 'distance' in axes and args.observe_link_length_km is not None:
        raise ValueError('距离扫描不能使用 --observe-link-length-km 覆盖扫描距离')
    output = Path(args.output_dir or base / 'results' / (
        'traffic_scan_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('扫描输出目录必须为空，请指定新的 --output-dir')
    # 先校验全部扫描参数，再逐组运行，避免后一组参数错误留下半批输出。
    jobs = []
    for axis in axes:
        job = Namespace(**vars(args))
        job.scan_axis = axis
        job.output_dir = output / (axis + '_scan') if len(axes) > 1 else output
        scan_settings(job)
        jobs.append(job)
    results = {}
    for job in jobs:
        results[job.scan_axis] = run_load_scan(job, build_simulation, base, variants=variants)
    return results



def select_comparison_rows(rows, variants, keys):
    """保留所选曲线，并附加同工况 FF 四指标供 Excel 比值计算；内部基准不冒充所选算法。"""
    selected = {v['label'] for v in variants if v['export']}
    baseline = {tuple(row[k] for k in keys): row for row in rows if row['algorithm'] == 'FF'}
    metrics = ('skr_mean', 'osnr_linear_mean', 'blocking_rate', 'synergy_vs_FF')
    result = []
    for row in rows:
        if row['algorithm'] in selected:
            ref = baseline.get(tuple(row[k] for k in keys), {})
            row['ff_reference'] = {metric: ref.get(metric) for metric in metrics}
            result.append(row)
    return result


def run_load_scan(args, build_simulation, base, *, variants):
    """遍历主控传入的场景、负载、种子与算法组合，返回含配置、汇总、单次指标和样本的字典。"""
    loads, seeds, warmup, scenarios = scan_settings(args)
    # 先检查导出依赖，避免长时间仿真完成后才发现无法生成图表。
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401
    algorithms = [v['label'] for v in variants if v['export']]
    output = args.output_dir or base / 'results' / ('traffic_scan_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('traffic_scan*')) or any(output.glob('traffic_diagnostics*')):
        raise ValueError('输出目录已有业务扫描结果，请指定新的 --output-dir')
    metadata = dict(allow_bidirectional=args.allow_bidirectional, scan_axis=getattr(args, 'scan_axis', 'load'), topology=args.topology, loads_erlang=loads, seeds=seeds, slots=args.slots,
                    warmup=warmup, holding_time=args.holding_time, algorithms=algorithms,
                    scenarios=scenarios, k=args.k, skip_c33=not args.include_c33,
                    classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
                    core_layout='fixed seven-core algorithm layout',
                    qcnm_noise_rtols=args.qcnm_noise_rtol,
                    comparison_variants=variants,
                    qcnm_objective='Require N <= Nmin + rtol*abs(Nmin) for incremental Raman + FWM power at quantum receivers (W); then minimize occupied co-directional same-frequency first-neighbor count, noise, and channel index',
                    qcnm_frequency_preference='Disabled: replaced by co-directional same-frequency nearest-neighbor occupancy count',
                    qcnm_neighbor_objective='Count occupied co-directional same-frequency classical channels on first neighbors; no secondary neighbors, opposite direction, power weights or XT calculation',
                    scwa_rule='Reference seven-core groups: forward odd [4], even [3,5]; backward odd [6], even [0,2]. Odd/even lists use original channel indices and actual channel count. Count every state != 1 in the original parity cells; swap when count >= nominal capacity - 10. Search original indices in ascending order, with no fallback',
                    scwa_version='reference_seven_core_fixed_reserve_10_no_fallback_v2',
                    allocation_mode='single_core_per_hop', three_core_binding=False,
                    load_definition='A = arrival_rate * holding_time; network-wide offered traffic',
                    observation_window=f'[{warmup}, {args.slots}) time units',
                    time_unit='One simulation time unit per slot; arrival_rate is per time unit',
                    carried_load_definition='Exact integral of active accepted services / observation duration',
                    gain_definition='Ratio of equally weighted seed mean SKR minus one; blank if baseline absent or zero',
                    uncertainty='Sample SD across independent seed means; blank for one seed; not a confidence interval',
                    blank_definition='Unavailable or undefined, not zero',
                    python=platform.python_version(),
                    dependencies={name: version(name) for name in ('numpy', 'pandas', 'networkx', 'openpyxl', 'matplotlib')})
    metadata['metric_units'] = dict(skr='bit/s', raw_skr='bit/s', no_fwm_skr='bit/s',
        zero_noise_skr='bit/s', raman_w='W', fwm_w='W', offered_load_erlang='Erlang',
        carried_load_erlang='Erlang', rates_and_gains='fraction; Excel displays percent',
        noise_counts='counts per gate, summed over quantum receivers')
    metadata.update(METRIC_DEFINITIONS, observe_link=args.observe_link or 'shortest topology edge; node-order tie break',
                    observe_link_length_km=args.observe_link_length_km,
                    observed_length_policy='Default: use original topology edge lengths without scaling; explicit uniform or observed-edge overrides are separate experiments; actual observed length stored in runs')
    metadata['metric_units'].update(osnr_linear='power ratio', osnr_db='dB',
        classical_xt_w='W', classical_floor_w='W',
        synergy_vs_FF='dimensionless')
    metadata['source_sha256'] = source_hashes(base, args.topology)
    # variants 已由主控展开容差并补 FF 内部基准；这里只运行指定组合，按 export 标记筛选输出。
    runs, samples, configurations = [], [], {}
    # 跨扫描点也比较输入业务摘要；距离改变路由时，仍只要求到达业务一致。
    traffic_hashes = {}
    total = len(scenarios) * len(loads) * len(seeds) * len(variants)
    for scenario_index, (length, power) in enumerate(scenarios, 1):
        # 场景名称也是配对键，repr 保留浮点精度，防止相近参数显示重名后混算。
        scenario = f'{length!r} km / {power!r} dBm' if length is not None else f'{args.topology} topology lengths / {power!r} dBm'
        for load in loads:
            for seed in seeds:
                reference_hash = None
                for variant in variants:
                    algorithm = variant["algorithm"]
                    label = variant["label"]
                    # 负载 A=到达率×平均保持时间，扫描由 A 反推到达率，不采用普通运行的 arrival-rate。
                    sim = build_simulation(base / 'topologies' / f'{args.topology}.json',
                        algorithm=algorithm, slots=args.slots, arrival_rate=load / args.holding_time,
                        holding_time=args.holding_time, k=args.k, seed=seed,
                        classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
                        launch_power=1e-3 * 10 ** (power / 10), link_length_km=length,
                        skip_c33=not args.include_c33,
                        allow_bidirectional=args.allow_bidirectional, qcnm_noise_rtol=variant["qcnm_noise_rtol"] or 0.0, observe_link=args.observe_link,
                        observe_link_length_km=args.observe_link_length_km,
                        key_pulses=args.key_pulses, key_gamma=args.key_gamma)
                    metadata.setdefault('skr_model', skr_model_config(sim.bb84_params, sim.detector_params))
                    config_key = f'{scenario_index}:{label}'
                    if config_key not in configurations:
                        configurations[config_key] = dict(qcnm_noise_rtol=variant["qcnm_noise_rtol"], frequencies_hz=sim.available_channel,
                            forward_cores=sim.classical_forward_cores, backward_cores=sim.classical_backward_cores,
                            quantum_cores=sim.quantum_cores, observed_link=list(sim.observed_link),
                            length_scaling=sim.graph.graph['length_scaling'],
                            edges_m=[(int(a), int(b), float(sim.a_m[a,b])) for a,b in sim.graph.edges],
                            detector=asdict(sim.detector_params), bb84=asdict(sim.bb84_params),
                            raman_model=raman_model_config(),
                            first_fiber=asdict(sim.noise_model.first_fiber),
                            secondary_fiber=asdict(sim.noise_model.secondary_fiber))

                    common = dict(scenario=scenario, offered_load_erlang=load, algorithm=label,
                                  qcnm_noise_rtol=variant["qcnm_noise_rtol"],
                                  seed=seed, length_km=length, power_dbm=power,
                                  arrival_rate=load / args.holding_time)
                    row, trace = measure_run(sim, warmup)
                    # 校验输入到达序列相同，不意味着不同算法接入了相同业务。
                    if reference_hash is not None and row['traffic_sha256'] != reference_hash:
                        raise ValueError('Paired algorithms received different traffic traces')
                    reference_hash = row['traffic_sha256']
                    paired_hash = traffic_hashes.setdefault((load, seed), reference_hash)
                    if paired_hash != reference_hash:
                        raise ValueError('Scan points received different traffic traces')
                    runs.append({**common, **row})
                    samples.extend({**common, **sample} for sample in trace)
                    block = 'n/a' if row['blocking_rate'] is None else f"{row['blocking_rate']:.2%}"
                    print(f"[{len(runs)}/{total}] {scenario}, A={load:g}, seed={seed}, {label}: "
                          f"SKR={row['skr_mean']:.1f} bit/s, blocking={block}", flush=True)
    metadata['physical_configurations'] = configurations
    add_paired_synergy(runs, ('scenario', 'offered_load_erlang', 'seed'))
    summary = summarize(runs)
    summary = select_comparison_rows(summary, variants, ('scenario', 'offered_load_erlang'))
    runs = select_comparison_rows(runs, variants, ('scenario', 'offered_load_erlang', 'seed'))
    samples = [row for row in samples if row['algorithm'] in algorithms]
    # JSON 始终保留逐时隙样本；save_samples 仅兼容旧参数，Excel 仅输出汇总简表。
    data = dict(config=metadata, summary=summary, runs=runs, samples=samples)
    figures = export_scan_results(output, data, args.save_samples)
    print(f"扫描完成：{output}\nExcel: traffic_scan.xlsx\nJSON: traffic_scan.json\n图: {', '.join(figures)}")
    return data


def summarize_business(runs):
    """按扫描组、工况和算法容差等权汇总种子结果，每个容差单独占一行。"""
    frame = pd.DataFrame(runs)
    result = {'load_scan': [], 'power_scan': []}
    metrics = ('skr_mean', 'osnr_linear_mean', 'osnr_db_mean', 'blocking_rate',
               'synergy_vs_FF', 'carried_load_erlang', 'zero_skr_fraction_mean',
               'delta_osnr_linear_vs_FF', 'delta_skr_vs_FF')
    for (group, load, power, algorithm), data in frame.groupby(
            ['group', 'load_erlang', 'power_dbm', 'algorithm'], sort=False):
        rtol = data.iloc[0]['qcnm_noise_rtol']
        row = dict(group=group, load_erlang=float(load), power_dbm=float(power),
                   algorithm=algorithm, seed_count=len(data),
                   qcnm_noise_rtol=float(rtol) if pd.notna(rtol) else None)
        for metric in metrics:
            values = data[metric].dropna()
            # 任一种子 OSNR/协同度缺失时整组留空；标准差仍按独立种子计算。
            if metric in ('osnr_linear_mean', 'osnr_db_mean', 'synergy_vs_FF',
                          'delta_osnr_linear_vs_FF') and len(values) != len(data):
                values = values.iloc[:0]
            row[metric] = float(values.mean()) if len(values) else None
            row[metric + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        result[group].append(row)
    return result


def run_business_export(args, build_simulation, base, *, variants):
    """运行主控指定的三芯业务组合并导出所选链路回放，返回批次索引。
负载按全网双向合计业务组计数，每芯每信道功率保持输入值。"""
    # 在仿真前检查工作簿和图表所需依赖。
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401

    if args.scan_load or args.scan_scenarios or args.save_samples:
        raise ValueError('--export-business 不与统计扫描模式同时使用')
    if args.classical_channels <= 0:
        raise ValueError('实验经典信道数必须为正整数')
    # 三芯实验固定量子频率 193.5 THz（C35），排除经典频率 193.3 THz（C33）。
    if args.quantum_channels != 1 or args.include_c33:
        raise ValueError('实验业务导出使用 C35 量子信道，经典信道跳过 C33')
    # 三芯组负载不除以三，也不为达到阻塞率目标自动降载；默认保持时间为 4。
    loads, seeds, warmup, _ = scan_settings(args)
    if args.loads is None:
        loads = [5, 10, 15, 20, 25, 30, 35, 40]
    fixed_load = 10 if args.fixed_load is None else args.fixed_load
    powers = args.powers if args.powers is not None else [7, 8, 9, 10, 10.5]
    if not math.isfinite(fixed_load) or fixed_load <= 0:
        raise ValueError('fixed-load 必须为有限正数')
    if not powers or len(powers) != len(set(powers)):
        raise ValueError('powers 必须非空且不能重复')
    for power in [*powers, args.fixed_power]:
        if not math.isfinite(power) or not -100 < power < 100:
            raise ValueError('实验功率必须为 (-100, 100) 内有限 dBm 数值')
    algorithms = [v['label'] for v in variants if v['export']]
    output = Path(args.output_dir or base / 'results' / ('business_export_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('业务导出目录必须为空，避免覆盖实验依据')
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(base, args.topology)
    # 负载组与功率组独立扫描，交叉工况在两组各保留一份。
    experiments = [('load_scan', load, args.fixed_power) for load in loads]
    experiments += [('power_scan', fixed_load, power) for power in sorted(powers)]
    metadata = dict(allow_bidirectional=args.allow_bidirectional, loads_erlang=loads, powers_dbm=sorted(powers), fixed_load_erlang=fixed_load,
                    classical_channels=args.classical_channels, quantum_channels=1,
                    fixed_power_dbm=args.fixed_power, seeds=seeds, slots=args.slots, warmup=warmup,
                    holding_time=args.holding_time, topology=args.topology, length_km=args.link_length_km, algorithms=algorithms,
                    skr_units='Sweep sheets/figures: kbit/s; runs and trace JSON: bit/s. Not accumulated secret bits.',
                    uncertainty='Equal-weight seed means; sample SD across seeds, blank for one seed; not a confidence interval',
                    blank_definition='Unavailable or undefined, not zero',
                    simulation_time_unit='simulation_unit',
                    experiment_duration_seconds=args.experiment_duration_seconds,
                    trace_schema_version=3,
                    timing='Full warmup and resource transitions retained; experiment duration maps linearly to simulation time',
                    power_reference='Per core per classical channel at fiber input, dBm', source_sha256=hashes,
                    allocation_mode='three_core_bound', traffic_unit='three_core_business_group',
                    qcnm_objective='Require N <= Nmin + rtol*abs(Nmin) for incremental Raman + FWM power at quantum receivers (W); then minimize occupied co-directional same-frequency first-neighbor count, noise, and channel index',
                    qcnm_frequency_preference='All three cores must use the same frequency; fixed directional groups can yield identical results across tolerances',
                    qcnm_noise_rtols=args.qcnm_noise_rtol, comparison_variants=variants,
                    load_definition='Erlang of three-core business groups; A = group arrival rate * mean holding time',
                    default_load_scaling='Load scan defaults to 5,10,15,20,25,30,35,40 Erlang three-core groups; power scan defaults to 10; explicit group loads are not divided by three or reduced to meet a blocking target',
                    carried_load_definition='Integral of active accepted groups / observation duration',
                    resource_definition='capacity/utilization are network-wide; occupied_channels counts only the selected link; all count physical core-channel cells')
    # 5% 仅是人为实验目标，不参与接入决策，也不保证任何给定工况达到该目标。
    metadata.update(blocking_rate_limit=.05,
                    blocking_definition='Blocked three-core group arrivals / offered group arrivals in [warmup, slots); not packet loss or BER',
                    blocking_limit_definition='User-selected experiment target, strictly below 5%; not an enforced admission rule or universal standard')
    metadata.update(METRIC_DEFINITIONS, observe_link=args.observe_link or 'shortest topology edge; node-order tie break',
                    observe_link_length_km=args.observe_link_length_km,
                    observed_length_policy='Default: use original topology edge lengths without scaling; explicit uniform or observed-edge overrides are separate experiments; actual observed length stored in runs',
                    synergy_baseline_layout=dict(quantum=[6], forward=[0, 1, 2], backward=[3, 4, 5]),
                    synergy_baseline_resource_policy='Three-core binding; disjoint directional groups; ascending actual frequency')
    metadata['cca_experiment_policy'] = 'Three-core adaptation: quantum core 0; forward [1,2,3]; backward [4,5,6]; ascending actual frequency; disjoint directions, not reference six-core shared CCA'
    manifest = dict(schema_version=4, kind='qkd_business_experiments',
                    description='Network allocation; selected-link replay and metrics; full warmup retained',
                    config=metadata, files=[])
    # 三芯导出仅选择 CCA/QCNM；主控补跑的 FF 仅用于配对基准，不导出其回放或曲线。
    references, runs = {}, []
    for group, load, power in experiments:
        for seed in seeds:
            for variant in variants:
                algorithm = variant["algorithm"]
                label = variant["label"]
                sim = build_simulation(base / 'topologies' / f'{args.topology}.json',
                    algorithm=algorithm, slots=args.slots, arrival_rate=load / args.holding_time,
                    holding_time=args.holding_time, k=args.k, seed=seed,
                    classical_channels=args.classical_channels, quantum_channels=1,
                    launch_power=1e-3 * 10 ** (power / 10), link_length_km=args.link_length_km,
                    skip_c33=True,
                    allow_bidirectional=args.allow_bidirectional, qcnm_noise_rtol=variant["qcnm_noise_rtol"] or 0.0, bind_three=True, observe_link=args.observe_link,
                        observe_link_length_km=args.observe_link_length_km,
                        key_pulses=args.key_pulses, key_gamma=args.key_gamma)
                metadata.setdefault('skr_model', skr_model_config(sim.bb84_params, sim.detector_params))
                recorder = TrafficRecorder(sim, seed=seed, warmup=warmup)
                metrics, samples = measure_run(sim, warmup, event_recorder=recorder)
                key = (load, seed)
                digest = metrics['traffic_sha256']
                if key in references and references[key] != digest:
                    raise ValueError('不同算法或功率的业务到达序列不一致')
                references[key] = digest
                data = recorder.finish(metrics, hashes)
                data['samples'] = samples
                data['config']['algorithm'] = label
                runs.append(dict(group=group, algorithm=label, load_erlang=load,
                                 power_dbm=power, seed=seed, qcnm_noise_rtol=variant['qcnm_noise_rtol'], **metrics))
                if not variant['export']:
                    continue
                file_label = f'QCNM_rtol_{variant["qcnm_noise_rtol"]!r}' if algorithm == 'QCNM' else algorithm
                filename = f'{group}/{file_label}_A{load:g}_P{power:g}_seed{seed}.json'
                data = export_replay_timing(data, args.experiment_duration_seconds)
                file_hash = write_trace(output / filename, data)
                manifest['files'].append(dict(file=filename, sha256=file_hash, group=group,
                    algorithm=label, qcnm_noise_rtol=variant['qcnm_noise_rtol'], load_erlang=load, power_dbm=power, seed=seed,
                    traffic_sha256=digest, state_count=len(data['states'])))
                print(f"[{len(manifest['files'])}/{len(experiments)*len(seeds)*len(algorithms)}] "
                      f"{filename}: {len(data['states'])} states, SKR={metrics['skr_mean']:.1f} bit/s", flush=True)
    add_paired_synergy(runs, ('group', 'load_erlang', 'power_dbm', 'seed'))
    summary = summarize_business(runs)
    summary = {group: select_comparison_rows(rows, variants, ('load_erlang', 'power_dbm'))
               for group, rows in summary.items()}
    runs = select_comparison_rows(runs, variants, ('group', 'load_erlang', 'power_dbm', 'seed'))
    # 配对协同度在所有算法完成后计算；索引保留每种子值和跨种子汇总。
    manifest.update(runs=runs, summary=summary)
    artifacts = export_business_summary(output, runs, metadata, summary)
    manifest['artifacts'] = [dict(file=name, sha256=sha256((output / name).read_bytes()).hexdigest())
                             for name in artifacts]
    # 索引最后写入才表示批次完成；中途失败可能留下不完整文件，重跑应使用新目录。
    write_trace(output / 'manifest.json', manifest)
    print(f'业务导出完成：{output}', flush=True)
    return manifest
