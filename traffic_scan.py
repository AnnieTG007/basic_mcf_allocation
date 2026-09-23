"""组织负载/功率/距离实验、记录时隙末指标并汇总多个随机种子的结果。

由 main.py --scan-load/--scan-power/--scan-distance/--scan-all 或 --export-business 调用，不能单独启动实验。
输入为命令行参数和创建仿真实例的函数；事件始终由 sim.iter_slots 推进，
本模块不另造业务或占用资源。完成的统计交给 traffic_export 写 JSON、Excel 和图。

负载 A=到达率*平均保持时间，单位 Erlang，表示全网络输入负载而非每链路负载。
普通负载/功率/距离扫描按单芯业务计数；仅 --export-business 按三芯业务组计数。
默认 topology7 为两节点链路，边长直接读取拓扑文件，负载是两个方向的合计，方向独立等概率抽取。
普通负载扫描默认30 Erlang，普通功率/距离扫描固定负载默认10 Erlang；
三芯导出负载扫描默认5至40、步长5 Erlang业务组，功率扫描固定10业务组。
保持时间均默认4。
扫描用 A/holding_time 设置到达率，不使用 --arrival-rate。
统计窗口为 [warmup, slots)，slots 包含预热；每个预热后的时隙末采样一次。
SKR（秘密密钥率）采用双诱骗态有限样本估算，各量子信道先截零，再对选中链路
量子信道和窗口时隙平均，单位bit/s；raw_skr仅作截零前差值诊断。
每个资源快照使用固定脉冲块，块长不由时隙数推算。单次运行也复用此统计。

同一场景、负载和种子的算法使用相同到达序列，并比较序列摘要确认一致；
多跳网络仍可能因资源位置不同得到不同阻塞率。统计和绘图不保证算法达到某个增益。
"""
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

from algorithm import ALGORITHMS
from skr_calculation import skr_model_config, synergy_skr_bounds
from synergistic_calculation import osnr_db, add_paired_synergy, METRIC_DEFINITIONS
from traffic_export import (TrafficRecorder, export_business_summary,
                            export_scan_results, write_trace)


METRICS = ("skr", "raw_skr", "no_fwm_skr", "zero_noise_skr", "raman_w",
           "fwm_w", "noise_counts", "active_services", "occupied_channels",
           "zero_skr_fraction", "osnr_linear", "zero_xt_channels",
           "classical_received_power_w", "classical_noise_power_w", "classical_noise_per_channel_w",
           "classical_xt_w", "classical_floor_w")
GROUP = ["scenario", "offered_load_erlang", "algorithm"]


def measure_run(sim, warmup, event_recorder=None):
    """运行一个全新实例，返回 (本种子指标字典, 时隙末样本列表)。
    
    到达/接入/阻塞数仅计入窗口内的到达，阻塞率=阻塞数/到达数；没有到达时
    为 None，不能解释为零。承载负载是已接入业务持续时间与窗口的交集之和
    除以窗口长度，包含预热期间到达但仍活跃的业务。每条业务按实际占用的芯数逐跳累计
    占用时间（普通仿真每跳一芯，实验业务组每跳三芯），再除以窗口长度和初始
    可用经典资源数得到 channel_utilization。
    
    active_services 是全网当时活跃业务数（实验为组数）；occupied_channels 只按观测
    链路的芯和信道逐格计数，三芯组在该链路占三格。capacity 按全网可同时占用的物理格计数：
    FF/CCA 两方向开放的同芯同频互斥，因此共享格只计一次；其余模式按方向计数。
    零SKR比例只计算所选链路。可用SKR按有限样本模型逐信道截零后平均；
    raw_skr为单独诊断列，不参与增益、协同度或正式SKR曲线。
    OSNR 按参考业务入口公式的单链路版本，在每个非空时隙求平均接收信号
    除以（平均串扰+固定噪声底），再在线性域按时隙等权平均，最后转 dB。
    空闲不参与 OSNR 平均；有占用但零串扰仍因噪声底而有限。
    经典功率/噪声及占用数只统计所选链路；active_services、阻塞率、承载量、
    channel_utilization 保持全网业务口径，不能解释成单链路业务统计。
    相同到达序列不等于逐个比较相同已接入业务。
    greedy 直接比较总噪声，不再统计安全等级或回退率。可选 recorder 记录全部资源变化。
    """
    if not 0 <= warmup < sim.Ts:
        raise ValueError("warmup 必须满足 0 <= warmup < slots")
    links = sorted((min(a, b), max(a, b)) for a, b in sim.graph.edges)
    capacity = int(np.count_nonzero(sim.m_resourceMap == 1))
    if sim.allocator.reverse_exclusive:
        capacity -= sum(int(np.count_nonzero((sim.m_resourceMap[a, b] == 1)
                                            & (sim.m_resourceMap[b, a] == 1)))
                        for a, b in links)
    offered = blocked_count = active = 0
    carried_time = occupied_time = 0.0
    digest = sha256()
    samples = []
    started = time.perf_counter()

    def observe_event(event, *, blocked):
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
                duration = max(0.0, min(sim.Ts, event.m_time + event.m_holdTime)
                               - max(warmup, event.m_time))
                carried_time += duration
                occupied_time += duration * sum(np.size(cores) for cores in event.m_ocuppiedcore)
        else:
            active -= 1

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
        for key in ('skr', 'raw_skr', 'no_fwm_skr', 'zero_noise_skr'):
            totals[key] /= quantum_count
        totals.update(time=slot + 1, active_services=active,
                      occupied_channels=0,
                      zero_skr_fraction=zero_count / quantum_count)
        if not all(math.isfinite(value) for value in totals.values()):
            raise ValueError("Non-finite simulation metric")
        totals['osnr_linear'] = sim.measure_classical_osnr()
        classical = sim.classical_osnr_scorer.statistics
        totals['zero_xt_channels'] = classical['zero_noise_channels']
        totals['occupied_channels'] = classical['occupied_channels']
        totals['classical_received_power_w'] = classical['signal_sum_w']
        totals['classical_noise_power_w'] = classical['noise_sum_w']
        totals['classical_noise_per_channel_w'] = classical['noise_per_channel_w']
        totals['osnr_db'] = osnr_db(totals['osnr_linear'])
        totals['classical_xt_w'] = classical['xt_sum_w']
        totals['classical_floor_w'] = classical['floor_sum_w']
        samples.append(totals)
    accepted = offered - blocked_count
    # 空时隙不参与 OSNR 比值平均；SKR 包含空闲时隙。
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
        sim.bb84_params, sim.detector_params))
    row.update(offered=offered, accepted=accepted, blocked=blocked_count,
               blocking_rate=blocked_count / offered if offered else None,
               carried_load_erlang=carried_time / (sim.Ts - warmup),
               channel_utilization=occupied_time / (sim.Ts - warmup) / capacity,
               capacity=capacity, samples=len(samples),
               traffic_sha256=digest.hexdigest(), runtime_s=time.perf_counter() - started)
    return row, samples


def summarize(runs):
    """按场景、负载、算法分组，各独立种子的均值等权平均。

    OSNR/协同度任一种子未定义时整体留空，不以剩余种子代替完整种子组。
    
    _sd 是种子均值之间的样本标准差，不是置信区间；仅一个有效种子时为 None。
    synergy_vs_FF 是同工况同种子先配对计算、再跨种子等权平均的有符号协同度；
    delta_osnr_linear_vs_FF 和 delta_skr_vs_FF 保留方向，避免乘积为零掩盖退化。
    osnr_db_mean 为各种子 dB 均值的平均，不等于跨种子线性均值再转 dB。
    增益=算法跨种子平均 SKR/基准跨种子平均 SKR-1，不是各种子增益的平均。
    基准 FF/SCWA 未运行或均值非正时不计算增益，None 在 Excel 中显示为空白。
    """
    frame = pd.DataFrame(runs)
    statistics = [f"{key}_mean" for key in METRICS] + [
        "blocking_rate", "carried_load_erlang", "channel_utilization",
        "osnr_db_mean", "synergy_vs_FF",
        "delta_osnr_linear_vs_FF", "delta_skr_vs_FF"]
    rows = []
    for keys, group in frame.groupby(GROUP, sort=False):
        row = dict(zip(GROUP, keys))
        row.update(observed_link=group.iloc[0]['observed_link'], observed_length_m=group.iloc[0]['observed_length_m'], seed_count=len(group), arrival_rate=group.iloc[0]['arrival_rate'],
                   length_km=group.iloc[0]['length_km'], power_dbm=float(group.iloc[0]['power_dbm']))
        for key in statistics:
            values = group[key].dropna()
            if key in ('osnr_linear_mean', 'osnr_db_mean', 'synergy_vs_FF',
                       'delta_osnr_linear_vs_FF') and len(values) != len(group):
                values = values.iloc[:0]
            row[key] = float(values.mean()) if len(values) else None
            row[key + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    lookup = {(r['scenario'], r['offered_load_erlang'], r['algorithm']): r for r in rows}
    for row in rows:
        for baseline in ('FF', 'SCWA'):
            base = lookup.get((row['scenario'], row['offered_load_erlang'], baseline))
            row['gain_vs_' + baseline] = (row['skr_mean'] / base['skr_mean'] - 1
                if base is not None and base['skr_mean'] > 0 else None)
    return rows


def source_hashes(base, topology, raman_file):
    """为本次运行计算九个源码文件、所用拓扑和拉曼表的 SHA-256 摘要；不读取旧批次。"""
    names = ('main.py', 'traffic_scan.py', 'traffic_export.py', 'algorithm.py',
             'noise_calculation.py', 'skr_calculation.py', 'core_layout.py', 'topology.py',
             'synergistic_calculation.py')
    paths = [base / name for name in names]
    paths += [base / 'topologies' / f'{topology}.json', Path(raman_file)]
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def scan_settings(args):
    """解析并校验扫描点，返回 (升序负载列表, 种子列表, 预热时长, 场景列表)。
    
    场景为 (全边覆盖长度km, 每经典信道功率dBm)；长度 None 表示不覆盖全网边长，
    默认由 build_simulation 直接读取拓扑文件中的各边长度；显式观测长度另行覆盖。
    --scan-scenarios 将长度和功率逐项配对，不取所有交叉组合；负载与种子不能重复。
    scan_axis 为 power/distance 时场景按该物理量升序排列，负载固定为 fixed_load。
    """
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
    scenarios = []
    if axis == 'power':
        powers = args.powers if args.powers is not None else [7, 8, 9, 10, 10.5]
        scenarios = [(args.link_length_km, power) for power in sorted(powers)]
    elif axis == 'distance':
        distances = args.distances if args.distances is not None else [1, 5, 10, 20, 30, 40, 50]
        scenarios = [(length, args.launch_power_dbm) for length in sorted(distances)]
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


def run_traffic_scans(args, build_simulation, base):
    """一次运行所选维度的独立扫描，不取负载×功率×距离的笛卡尔积。

    负载组使用 loads；功率、距离组固定为 fixed_load（默认10 Erlang）。
    功率组使用 powers/dBm，距离组使用 distances/km；距离组全边等长，
    其余组使用拓扑距离或 link_length_km。负载和距离组固定 launch_power_dbm。
    距离组禁止单边长度覆盖，防止横轴变化而观测距离不变。
    多组结果分别写入同一批次的 load_scan/power_scan/distance_scan，单组保持原目录结构。
    先校验全部扫描参数，再运行；每组仍复用唯一事件循环和原统计/导出逻辑。
    """
    axes = [axis for axis in ('load', 'power', 'distance') if getattr(args, 'scan_' + axis)]
    if args.scan_scenarios and axes != ['load']:
        raise ValueError('--scan-scenarios 仅用于单独负载扫描；多维扫描请指定 --powers/--distances')
    if 'distance' in axes and args.observe_link_length_km is not None:
        raise ValueError('距离扫描不能使用 --observe-link-length-km 覆盖扫描距离')
    output = Path(args.output_dir or base / 'results' / (
        'traffic_scan_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('扫描输出目录必须为空，请指定新的 --output-dir')
    jobs = []
    for axis in axes:
        job = Namespace(**vars(args))
        job.scan_axis = axis
        job.output_dir = output / (axis + '_scan') if len(axes) > 1 else output
        scan_settings(job)
        jobs.append(job)
    results = {}
    for job in jobs:
        results[job.scan_axis] = run_load_scan(job, build_simulation, base)
    return results


def run_load_scan(args, build_simulation, base):
    """运行场景×负载×种子×算法的组合，通过相同业务序列公平比较算法。
    
    ALL 运行五种算法；单独选择算法时自动补跑新 FF 基准。返回含 config/summary/runs/samples 的字典，JSON 始终
    保存逐时隙样本，--save-samples 只决定是否附加 Excel Samples 表。
    参数、物理配置及本次依赖版本随结果保存；输出位置由 args.output_dir 决定。
    """
    loads, seeds, warmup, scenarios = scan_settings(args)
    # 先检查导出依赖，避免长时间仿真完成后才发现无法生成图表。
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401
    algorithms = list(ALGORITHMS) if args.algorithm == 'ALL' else list(dict.fromkeys((args.algorithm, 'FF')))
    output = args.output_dir or base / 'results' / ('traffic_scan_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('traffic_scan*')) or any(output.glob('traffic_diagnostics*')):
        raise ValueError('输出目录已有业务扫描结果，请指定新的 --output-dir')
    metadata = dict(scan_axis=getattr(args, 'scan_axis', 'load'), topology=args.topology, loads_erlang=loads, seeds=seeds, slots=args.slots,
                    warmup=warmup, holding_time=args.holding_time, algorithms=algorithms,
                    scenarios=scenarios, k=args.k, skip_c33=not args.include_c33,
                    classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
                    core_layout=args.core_layout or 'algorithm default',
                    greedy_noise_rtol=args.greedy_noise_rtol,
                    greedy_objective='Minimum incremental Raman + FWM optical power at quantum receivers (W); no safety tiers or parity/separation priority',
                    greedy_frequency_preference='Disabled: replaced by co-directional same-frequency nearest-neighbor occupancy count',
                    greedy_neighbor_objective='Single-core only: count occupied co-directional same-frequency classical channels on first neighbors; no secondary neighbors, opposite direction, power weights or XT calculation',
                    scwa_rule='Adaptive SCWA, not exact reference: ascending actual frequency ranks including quantum frequencies; first core per direction prefers odd ranks, others even; swap preference at 7/12 occupancy of eligible preferred classical cells (states 1/2 only); search preferred sets across the whole path first, then all directional cores with per-hop preference; no parity-only blocking',
                    scwa_version='eligible_occupancy_7_12_with_path_fallback_v1',
                    allocation_mode='single_core_per_hop', three_core_binding=False,
                    load_definition='A = arrival_rate * holding_time; network-wide offered traffic',
                    observation_window=f'[{warmup}, {args.slots}) time units',
                    time_unit='One simulation time unit per slot; arrival_rate is per time unit',
                    carried_load_definition='Exact integral of active accepted services / observation duration',
                    gain_definition='Ratio of equally weighted seed mean SKR minus one; blank if baseline absent or zero',
                    uncertainty='Sample SD across independent seed means; blank for one seed; not a confidence interval',
                    blank_definition='Unavailable or undefined, not zero',
                    python=platform.python_version(),
                    dependencies={name: version(name) for name in ('numpy', 'pandas', 'networkx', 'xlrd', 'openpyxl', 'matplotlib')})
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
    metadata['source_sha256'] = source_hashes(base, args.topology, args.raman_file)
    runs, samples, configurations = [], [], {}
    # 跨扫描点也比较输入业务摘要；距离改变路由时，仍只要求到达业务一致。
    traffic_hashes = {}
    total = len(scenarios) * len(loads) * len(seeds) * len(algorithms)
    for scenario_index, (length, power) in enumerate(scenarios, 1):
        # 场景名称也是配对键，repr 保留浮点精度，防止相近参数显示重名后混算。
        scenario = f'{length!r} km / {power!r} dBm' if length is not None else f'{args.topology} topology lengths / {power!r} dBm'
        for load in loads:
            for seed in seeds:
                reference_hash = None
                for algorithm in algorithms:
                    sim = build_simulation(base / 'topologies' / f'{args.topology}.json', args.raman_file,
                        algorithm=algorithm, slots=args.slots, arrival_rate=load / args.holding_time,
                        holding_time=args.holding_time, k=args.k, seed=seed,
                        classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
                        launch_power=1e-3 * 10 ** (power / 10), link_length_km=length,
                        core_layout=None if algorithm == "FF" else args.core_layout, skip_c33=not args.include_c33,
                        greedy_noise_rtol=args.greedy_noise_rtol, observe_link=args.observe_link,
                        observe_link_length_km=args.observe_link_length_km,
                        key_pulses=args.key_pulses, key_gamma=args.key_gamma)
                    metadata.setdefault('skr_model', skr_model_config(sim.bb84_params, sim.detector_params))
                    config_key = f'{scenario_index}:{algorithm}'
                    if config_key not in configurations:
                        configurations[config_key] = dict(frequencies_hz=sim.available_channel,
                            forward_cores=sim.classical_forward_cores, backward_cores=sim.classical_backward_cores,
                            quantum_cores=sim.quantum_cores, observed_link=list(sim.observed_link),
                            length_scaling=sim.graph.graph['length_scaling'],
                            edges_m=[(int(a), int(b), float(sim.a_m[a,b])) for a,b in sim.graph.edges],
                            detector=asdict(sim.detector_params), bb84=asdict(sim.bb84_params))

                    common = dict(scenario=scenario, offered_load_erlang=load, algorithm=algorithm,
                                  seed=seed, length_km=length, power_dbm=power,
                                  arrival_rate=load / args.holding_time)
                    row, trace = measure_run(sim, warmup)
                    if reference_hash is not None and row['traffic_sha256'] != reference_hash:
                        raise ValueError('Paired algorithms received different traffic traces')
                    reference_hash = row['traffic_sha256']
                    paired_hash = traffic_hashes.setdefault((load, seed), reference_hash)
                    if paired_hash != reference_hash:
                        raise ValueError('Scan points received different traffic traces')
                    runs.append({**common, **row})
                    samples.extend({**common, **sample} for sample in trace)
                    block = 'n/a' if row['blocking_rate'] is None else f"{row['blocking_rate']:.2%}"
                    print(f"[{len(runs)}/{total}] {scenario}, A={load:g}, seed={seed}, {algorithm}: "
                          f"SKR={row['skr_mean']:.1f} bit/s, blocking={block}", flush=True)
    metadata['physical_configurations'] = configurations
    add_paired_synergy(runs, ('scenario', 'offered_load_erlang', 'seed'))
    summary = summarize(runs)
    data = dict(config=metadata, summary=summary, runs=runs, samples=samples)
    figures = export_scan_results(output, data, args.save_samples)
    print(f"扫描完成：{output}\nExcel: traffic_scan.xlsx\nJSON: traffic_scan.json\n图: {', '.join(figures)}")
    return data


def summarize_business(runs):
    """分别汇总负载组和功率组，有效种子值等权平均；趋势表 SKR 为 kbit/s，runs 为 bit/s。

    gain_vs_FF/CCA 为 greedy 均值除以对应基准均值再减一，无量纲，保留零或负值。
    基准缺失或非正、greedy 缺失时留空；图表另行筛选最大正提升。
    """
    frame = pd.DataFrame(runs)
    result = {'load_scan': [], 'power_scan': []}
    for (group, load, power), points in frame.groupby(['group', 'load_erlang', 'power_dbm'], sort=True):
        row = dict(load_erlang=float(load), power_dbm=float(power))
        for algorithm, prefix in [('first-fit', 'ff'), ('cca', 'cca'), ('greedy_min_noise', 'greedy')]:
            data = points[points.algorithm == algorithm]
            row[prefix + '_seed_count'] = len(data)
            row[prefix + '_skr_kbit_s'] = float(data.skr_mean.mean() / 1000) if len(data) else None
            row[prefix + '_skr_sd_kbit_s'] = float(data.skr_mean.std(ddof=1) / 1000) if len(data) > 1 else None
            for key in ('blocking_rate', 'carried_load_erlang', 'zero_skr_fraction_mean',
                        'osnr_linear_mean', 'osnr_db_mean', 'synergy_vs_FF',
                        'delta_osnr_linear_vs_FF', 'delta_skr_vs_FF'):
                values = data[key].dropna()
                if key in ('osnr_linear_mean', 'osnr_db_mean', 'synergy_vs_FF',
                           'delta_osnr_linear_vs_FF') and len(values) != len(data):
                    values = values.iloc[:0]
                row[prefix + '_' + key] = float(values.mean()) if len(values) else None
                row[prefix + '_' + key + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        greedy = row['greedy_skr_kbit_s']
        for baseline in ('FF', 'CCA'):
            value = row[baseline.lower() + '_skr_kbit_s']
            row['gain_vs_' + baseline] = greedy / value - 1 if value is not None and value > 0 and greedy is not None else None
        result[group].append(row)
    return result


def run_business_export(args, build_simulation, base):
    """运行全网三芯业务并导出所选链路，返回批次索引并输出资源状态回放。
    
    ALL 在本模式运行 FF、CCA 与 GREEDY_MIN_NOISE；固定 C35 量子信道和跳过 C33
    的经典候选，数量由 --classical-channels 指定（默认10）。仅本入口开启三芯绑定；到达/阻塞/承载负载按业务组统计，
    每芯每信道功率不变，普通运行与 --scan-load 仍按单芯业务运行。
    默认在 topology7 的两节点链路随机生成双向业务，距离读取拓扑文件。
    负载组默认5/10/15/20/25/30/35/40 Erlang业务组、固定10.5 dBm；功率组默认
    7/8/9/10/10.5 dBm、固定10 Erlang，可用 loads/powers/fixed-load/fixed-power 修改。
    负载均为两个方向的业务组合计；保持时间默认4，到达率为负载除以4。
    每组占三芯；功率扫描固定10 Erlang时，到达率为2.5组/仿真时间。
    显式 loads/fixed-load 已按三芯组计数，不再除以三，也不按阻塞率自动降载。
    相同负载、种子在各算法和功率下必须有相同到达序列。交叉工况在两组各留一份。
    
    5% 阻塞率是人为选定的实验参考线，不是通用标准，也不是算法接入限制；
    不保证默认工况或任意指定负载均低于此线。输出目录必须为空。
    manifest.json 最后生成，用作完成批次的索引；回放、图表和工作簿生成失败时
    可能留有不完整文件，应换新目录重跑，不把目录存在视作成功。
    """
    # 在仿真前检查工作簿和图表所需依赖。
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401

    if args.scan_load or args.scan_scenarios or args.save_samples:
        raise ValueError('--export-business 不与统计扫描模式同时使用')
    if args.classical_channels <= 0:
        raise ValueError('实验经典信道数必须为正整数')
    if args.quantum_channels != 1 or args.include_c33:
        raise ValueError('实验业务导出使用 C35 量子信道，经典信道跳过 C33')
    if args.algorithm not in ('ALL', 'FF', 'CCA', 'GREEDY_MIN_NOISE'):
        raise ValueError('实验业务支持 first-fit/FF、CCA 和 GREEDY_MIN_NOISE')
    if args.core_layout is not None:
        raise ValueError('三芯实验保留各算法默认分组，不支持 --core-layout')
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
    algorithms = ['FF', 'CCA', 'GREEDY_MIN_NOISE'] if args.algorithm == 'ALL' else list(dict.fromkeys((args.algorithm, 'FF')))
    output = Path(args.output_dir or base / 'results' / ('business_export_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('业务导出目录必须为空，避免覆盖实验依据')
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(base, args.topology, args.raman_file)
    experiments = [('load_scan', load, args.fixed_power) for load in loads]
    experiments += [('power_scan', fixed_load, power) for power in sorted(powers)]
    metadata = dict(loads_erlang=loads, powers_dbm=sorted(powers), fixed_load_erlang=fixed_load,
                    classical_channels=args.classical_channels, quantum_channels=1,
                    fixed_power_dbm=args.fixed_power, seeds=seeds, slots=args.slots, warmup=warmup,
                    holding_time=args.holding_time, topology=args.topology, length_km=args.link_length_km, algorithms=algorithms,
                    skr_units='Sweep sheets/figures: kbit/s; Runs and trace JSON: bit/s. Not accumulated secret bits.',
                    uncertainty='Equal-weight seed means; sample SD across seeds, blank for one seed; not a confidence interval',
                    gain_definition='Ratio of seed mean SKR minus one; blank if baseline absent or zero',
                    blank_definition='Unavailable or undefined, not zero',
                    timing='Full warmup and resource transitions retained; experiment duration maps linearly to simulation time',
                    power_reference='Per core per classical channel at fiber input, dBm', source_sha256=hashes,
                    allocation_mode='three_core_bound', traffic_unit='three_core_business_group',
                    greedy_objective='Minimum incremental Raman + FWM optical power at quantum receivers (W); no safety tiers or parity/separation priority',
                    greedy_frequency_preference='Disabled: all three cores must use the same frequency',
                    greedy_noise_rtol=None,
                    load_definition='Erlang of three-core business groups; A = group arrival rate * mean holding time',
                    default_load_scaling='Load scan defaults to 5,10,15,20,25,30,35,40 Erlang three-core groups; power scan defaults to 10; explicit group loads are not divided by three or reduced to meet a blocking target',
                    carried_load_definition='Integral of active accepted groups / observation duration',
                    resource_definition='capacity/utilization are network-wide; occupied_channels counts only the selected link; all count physical core-channel cells')
    metadata.update(blocking_rate_limit=.05,
                    blocking_definition='Blocked three-core group arrivals / offered group arrivals in [warmup, slots); not packet loss or BER',
                    blocking_limit_definition='User-selected experiment target, strictly below 5%; not an enforced admission rule or universal standard')
    metadata.update(METRIC_DEFINITIONS, observe_link=args.observe_link or 'shortest topology edge; node-order tie break',
                    observe_link_length_km=args.observe_link_length_km,
                    observed_length_policy='Default: use original topology edge lengths without scaling; explicit uniform or observed-edge overrides are separate experiments; actual observed length stored in runs',
                    synergy_baseline_layout=dict(quantum=[6], forward=[0, 1, 2], backward=[3, 4, 5]),
                    synergy_baseline_resource_policy='Three-core binding; disjoint directional groups; ascending actual frequency')
    metadata['cca_experiment_policy'] = 'Three-core adaptation: quantum core 0; forward [1,2,3]; backward [4,5,6]; ascending actual frequency; disjoint directions, not reference six-core shared CCA'
    manifest = dict(schema_version=3, kind='qkd_business_experiments',
                    description='Network allocation; selected-link replay and metrics; full warmup retained',
                    config=metadata, files=[])
    references, runs = {}, []
    for group, load, power in experiments:
        for seed in seeds:
            for algorithm in algorithms:
                sim = build_simulation(base / 'topologies' / f'{args.topology}.json', args.raman_file,
                    algorithm=algorithm, slots=args.slots, arrival_rate=load / args.holding_time,
                    holding_time=args.holding_time, k=args.k, seed=seed,
                    classical_channels=args.classical_channels, quantum_channels=1,
                    launch_power=1e-3 * 10 ** (power / 10), link_length_km=args.link_length_km,
                    core_layout=args.core_layout, skip_c33=True,
                    greedy_noise_rtol=args.greedy_noise_rtol, bind_three=True, observe_link=args.observe_link,
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
                data['config'].update(METRIC_DEFINITIONS,
                    cca_experiment_policy=metadata['cca_experiment_policy'],
                    synergy_baseline_layout=metadata['synergy_baseline_layout'],
                    synergy_baseline_resource_policy=metadata['synergy_baseline_resource_policy'],
                    greedy_objective='Minimum incremental Raman + FWM optical power at quantum receivers (W); no safety tiers or parity/separation priority',
                    greedy_frequency_preference='Disabled: three-core same-frequency binding', greedy_noise_rtol=None)
                label = data['config']['algorithm']
                filename = f'{group}/{label}_A{load:g}_P{power:g}_seed{seed}.json'
                file_hash = write_trace(output / filename, data)
                manifest['files'].append(dict(file=filename, sha256=file_hash, group=group,
                    algorithm=label, load_erlang=load, power_dbm=power, seed=seed,
                    traffic_sha256=digest, state_count=len(data['states'])))
                runs.append(dict(group=group, algorithm=label, load_erlang=load,
                                 power_dbm=power, seed=seed, **metrics))
                print(f"[{len(manifest['files'])}/{len(experiments)*len(seeds)*len(algorithms)}] "
                      f"{filename}: {len(data['states'])} states, SKR={metrics['skr_mean']:.1f} bit/s", flush=True)
    add_paired_synergy(runs, ('group', 'load_erlang', 'power_dbm', 'seed'), baseline='first-fit')
    summary = summarize_business(runs)
    # 配对协同度在所有算法完成后计算；索引保留每种子值和跨种子汇总。
    manifest.update(runs=runs, summary=summary)
    artifacts = export_business_summary(output, runs, metadata, summary)
    manifest['artifacts'] = [dict(file=name, sha256=sha256((output / name).read_bytes()).hexdigest())
                             for name in artifacts]
    # 最后写入批次索引，表示本批所有输出均已完成。
    write_trace(output / 'manifest.json', manifest)
    print(f'业务导出完成：{output}', flush=True)
    return manifest
