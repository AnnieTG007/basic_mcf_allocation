"""组织负载/功率实验、记录时隙末指标并汇总多个随机种子的结果。

由 main.py --scan-load 或 --export-business 调用，不能单独启动实验。
输入为命令行参数和创建仿真实例的函数；事件始终由 sim.iter_slots 推进，
本模块不另造业务或占用资源。完成的统计交给 traffic_export 写 JSON、Excel 和图。

负载 A=到达率*平均保持时间，单位 Erlang，表示全网络输入负载而非每链路负载。
扫描用 A/holding_time 设置到达率，不使用 --arrival-rate。
统计窗口为 [warmup, slots)，slots 包含预热；每个预热后的时隙末采样一次。
SKR（秘密密钥率）先按每量子信道截为非负，再对无向链路求和，单位 bit/s。
普通单次 main.run 的历史窗口和原始 SKR 与此不同，不能混算增益。

同一场景、负载和种子的算法使用相同到达序列，并比较序列摘要确认一致；
多跳网络仍可能因资源位置不同得到不同阻塞率。统计和绘图不保证算法达到某个增益。
"""
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
from traffic_export import (TrafficRecorder, export_business_summary,
                            export_scan_results, write_trace)


METRICS = ("skr", "raw_skr", "no_fwm_skr", "zero_noise_skr", "raman_w",
           "fwm_w", "noise_counts", "active_services", "occupied_channels",
           "zero_skr_fraction")
GROUP = ["scenario", "offered_load_erlang", "algorithm"]


def measure_run(sim, warmup, event_recorder=None):
    """运行一个全新实例，返回 (本种子指标字典, 时隙末样本列表)。
    
    到达/接入/阻塞数仅计入窗口内的到达，阻塞率=阻塞数/到达数；没有到达时
    为 None，不能解释为零。承载负载是已接入业务持续时间与窗口的交集之和
    除以窗口长度，包含预热期间到达但仍活跃的业务。每条多跳业务按跳数累计
    占用时间，除以窗口长度和初始可用经典资源数得到 channel_utilization。
    
    active_services 是当时活跃业务数，occupied_channels 是占用的链路信道数；
    零 SKR 比例是各时隙“零 SKR 量子信道数/全部量子信道数”的平均。
    fallback_rate 仅用于 GREEDY_MIN_NOISE，分母为窗口内成功接入数，分子为
    其中选择等级 2 的次数；不是 FWM 功率占比。可选 recorder 记录全部资源变化。
    """
    if not 0 <= warmup < sim.Ts:
        raise ValueError("warmup 必须满足 0 <= warmup < slots")
    links = sorted((min(a, b), max(a, b)) for a, b in sim.graph.edges)
    capacity = int(np.count_nonzero(sim.m_resourceMap == 1))
    offered = blocked_count = active = 0
    carried_time = occupied_time = 0.0
    tiers = [0, 0, 0]
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
                occupied_time += duration * (len(event.m_workPath) - 1)
                if measured and sim.allocator.noise_policy is not None:
                    tiers[sim.allocator.noise_policy.last_tier] += 1
        else:
            active -= 1

    for slot in sim.iter_slots(on_event=observe_event):
        if slot < warmup:
            continue
        totals = {key: 0.0 for key in METRICS[:7]}
        quantum_count = zero_count = 0
        for a, b in links:
            metrics = sim.quantum_scorer.metrics(
                sim.m_resourceMap[a, b], sim.m_resourceMap[b, a],
                sim.P_link[a, b], sim.P_link[b, a], sim.a_m[a, b])
            for key in totals:
                totals[key] += metrics[key]
            quantum_count += metrics['quantum_channels']
            zero_count += metrics['zero_skr_channels']
        totals.update(time=slot + 1, active_services=active,
                      occupied_channels=int(np.count_nonzero(sim.m_resourceMap == 2)),
                      zero_skr_fraction=zero_count / quantum_count)
        if not all(math.isfinite(value) for value in totals.values()):
            raise ValueError("Non-finite simulation metric")
        samples.append(totals)
    accepted = offered - blocked_count
    row = {f"{key}_mean": float(np.mean([s[key] for s in samples])) for key in METRICS}
    row.update(offered=offered, accepted=accepted, blocked=blocked_count,
               blocking_rate=blocked_count / offered if offered else None,
               carried_load_erlang=carried_time / (sim.Ts - warmup),
               channel_utilization=occupied_time / (sim.Ts - warmup) / capacity,
               capacity=capacity, samples=len(samples),
               fallback_rate=(tiers[2] / accepted if accepted else None)
               if sim.allocator.noise_policy is not None else None,
               preferred_safe_admissions=tiers[0] if sim.allocator.noise_policy is not None else None,
               other_safe_admissions=tiers[1] if sim.allocator.noise_policy is not None else None,
               fallback_admissions=tiers[2] if sim.allocator.noise_policy is not None else None,
               traffic_sha256=digest.hexdigest(), runtime_s=time.perf_counter() - started)
    return row, samples


def summarize(runs):
    """按场景、负载、算法分组，各独立种子的均值等权平均，忽略未定义值。
    
    _sd 是种子均值之间的样本标准差，不是置信区间；仅一个有效种子时为 None。
    增益=算法跨种子平均 SKR/基准跨种子平均 SKR-1，不是各种子增益的平均。
    基准 FF/SCWA 未运行或均值为零时不计算增益，None 在 Excel 中显示为空白。
    """
    frame = pd.DataFrame(runs)
    statistics = [f"{key}_mean" for key in METRICS] + [
        "blocking_rate", "carried_load_erlang", "channel_utilization", "fallback_rate"]
    rows = []
    for keys, group in frame.groupby(GROUP, sort=False):
        row = dict(zip(GROUP, keys))
        row.update(seed_count=len(group), arrival_rate=group.iloc[0]['arrival_rate'],
                   length_km=group.iloc[0]['length_km'], power_dbm=group.iloc[0]['power_dbm'])
        for key in statistics:
            values = group[key].dropna()
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
    """为本次运行计算八个源码文件、所用拓扑和拉曼表的 SHA-256 摘要；不读取旧批次。"""
    names = ('main.py', 'traffic_scan.py', 'traffic_export.py', 'algorithm.py',
             'noise_calculation.py', 'skr_calculation.py', 'core_layout.py', 'topology.py')
    paths = [base / name for name in names]
    paths += [base / 'topologies' / f'{topology}.json', Path(raman_file)]
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def scan_settings(args):
    """解析并校验扫描点，返回 (升序负载列表, 种子列表, 预热时长, 场景列表)。
    
    场景为 (每边长度km, 每经典信道功率dBm)，长度 None 表示保留拓扑边长。
    --scan-scenarios 将长度和功率逐项配对，不取所有交叉组合；负载与种子不能重复。
    """
    loads = args.loads if args.loads is not None else [10, 15, 20, 25, 30, 35, 40]
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
    if args.scan_scenarios:
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


def run_load_scan(args, build_simulation, base):
    """运行场景×负载×种子×算法的组合，通过相同业务序列公平比较算法。
    
    ALL 运行四种算法。返回含 config/summary/runs/samples 的字典，JSON 始终
    保存逐时隙样本，--save-samples 只决定是否附加 Excel Samples 表。
    参数、物理配置及本次依赖版本随结果保存；输出位置由 args.output_dir 决定。
    """
    loads, seeds, warmup, scenarios = scan_settings(args)
    # 先检查导出依赖，避免长时间仿真完成后才发现无法生成图表。
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401
    algorithms = list(ALGORITHMS) if args.algorithm == 'ALL' else [args.algorithm]
    output = args.output_dir or base / 'results' / ('traffic_scan_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('traffic_scan*')) or any(output.glob('traffic_diagnostics*')):
        raise ValueError('输出目录已有业务扫描结果，请指定新的 --output-dir')
    metadata = dict(topology=args.topology, loads_erlang=loads, seeds=seeds, slots=args.slots,
                    warmup=warmup, holding_time=args.holding_time, algorithms=algorithms,
                    scenarios=scenarios, k=args.k, skip_c33=not args.include_c33,
                    classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
                    core_layout=args.core_layout or 'algorithm default',
                    load_definition='A = arrival_rate * holding_time; network-wide offered traffic',
                    observation_window=f'[{warmup}, {args.slots}) time units',
                    skr_definition='Sum over undirected links, clipped per quantum channel; sampled at slot ends',
                    time_unit='One simulation time unit per slot; arrival_rate is per time unit',
                    carried_load_definition='Exact integral of active accepted services / observation duration',
                    gain_definition='Ratio of equally weighted seed mean SKR minus one; blank if baseline absent or zero',
                    uncertainty='Sample SD across independent seed means; blank for one seed; not a confidence interval',
                    fallback_definition='Tier 2 admissions / all admissions within observation window; only GREEDY_MIN_NOISE',
                    blank_definition='Unavailable or undefined, not zero',
                    python=platform.python_version(),
                    dependencies={name: version(name) for name in ('numpy', 'pandas', 'networkx', 'xlrd', 'openpyxl', 'matplotlib')})
    metadata['metric_units'] = dict(skr='bit/s', raw_skr='bit/s', no_fwm_skr='bit/s',
        zero_noise_skr='bit/s', raman_w='W', fwm_w='W', offered_load_erlang='Erlang',
        carried_load_erlang='Erlang', rates_and_gains='fraction; Excel displays percent',
        noise_counts='counts per gate, summed over quantum receivers')
    metadata['source_sha256'] = source_hashes(base, args.topology, args.raman_file)
    runs, samples, configurations = [], [], {}
    total = len(scenarios) * len(loads) * len(seeds) * len(algorithms)
    for scenario_index, (length, power) in enumerate(scenarios, 1):
        scenario = f'{length:g} km / {power:g} dBm' if length is not None else f'{args.topology} edge lengths / {power:g} dBm'
        for load in loads:
            for seed in seeds:
                reference_hash = None
                for algorithm in algorithms:
                    sim = build_simulation(base / 'topologies' / f'{args.topology}.json', args.raman_file,
                        algorithm=algorithm, slots=args.slots, arrival_rate=load / args.holding_time,
                        holding_time=args.holding_time, k=args.k, seed=seed,
                        classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
                        launch_power=1e-3 * 10 ** (power / 10), link_length_km=length,
                        core_layout=args.core_layout, skip_c33=not args.include_c33)
                    config_key = f'{scenario_index}:{algorithm}'
                    if config_key not in configurations:
                        configurations[config_key] = dict(frequencies_hz=sim.available_channel,
                            forward_cores=sim.classical_forward_cores, backward_cores=sim.classical_backward_cores,
                            quantum_cores=sim.quantum_cores,
                            edges_m=[(int(a), int(b), float(sim.a_m[a,b])) for a,b in sim.graph.edges],
                            detector=asdict(sim.detector_params), bb84=asdict(sim.bb84_params))
                    common = dict(scenario=scenario, offered_load_erlang=load, algorithm=algorithm,
                                  seed=seed, length_km=length, power_dbm=power,
                                  arrival_rate=load / args.holding_time)
                    row, trace = measure_run(sim, warmup)
                    if reference_hash is not None and row['traffic_sha256'] != reference_hash:
                        raise ValueError('Paired algorithms received different traffic traces')
                    reference_hash = row['traffic_sha256']
                    runs.append({**common, **row})
                    samples.extend({**common, **sample} for sample in trace)
                    block = 'n/a' if row['blocking_rate'] is None else f"{row['blocking_rate']:.2%}"
                    print(f"[{len(runs)}/{total}] {scenario}, A={load:g}, seed={seed}, {algorithm}: "
                          f"SKR={row['skr_mean']:.1f} bit/s, blocking={block}", flush=True)
    metadata['physical_configurations'] = configurations
    summary = summarize(runs)
    data = dict(config=metadata, summary=summary, runs=runs, samples=samples)
    figures = export_scan_results(output, data, args.save_samples)
    print(f"扫描完成：{output}\nExcel: traffic_scan.xlsx\nJSON: traffic_scan.json\n图: {', '.join(figures)}")
    return data


def summarize_business(runs):
    """分别汇总负载组和功率组，各种子等权平均；趋势表 SKR 转成 kbit/s，原始 runs 仍为 bit/s。"""
    frame = pd.DataFrame(runs)
    result = {'load_scan': [], 'power_scan': []}
    for (group, load, power), points in frame.groupby(['group', 'load_erlang', 'power_dbm'], sort=True):
        row = dict(load_erlang=float(load), power_dbm=float(power))
        for algorithm, prefix in [('first-fit', 'ff'), ('greedy_min_noise', 'greedy')]:
            data = points[points.algorithm == algorithm]
            row[prefix + '_seed_count'] = len(data)
            row[prefix + '_skr_kbit_s'] = float(data.skr_mean.mean() / 1000) if len(data) else None
            row[prefix + '_skr_sd_kbit_s'] = float(data.skr_mean.std(ddof=1) / 1000) if len(data) > 1 else None
            for key in ('blocking_rate', 'carried_load_erlang', 'zero_skr_fraction_mean'):
                values = data[key].dropna()
                row[prefix + '_' + key] = float(values.mean()) if len(values) else None
                if key == 'blocking_rate':
                    row[prefix + '_blocking_rate_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        ff, greedy = row['ff_skr_kbit_s'], row['greedy_skr_kbit_s']
        row['gain_vs_FF'] = greedy / ff - 1 if ff is not None and ff > 0 and greedy is not None else None
        result[group].append(row)
    return result


def run_business_export(args, build_simulation, base):
    """运行固定 10 km 双节点实验，返回批次索引并输出资源状态回放。
    
    ALL 在本模式仅运行 FF 与 GREEDY_MIN_NOISE；固定 C35 量子信道和跳过 C33
    的七个经典候选。负载组默认 10..40 Erlang、固定 10.5 dBm；功率组默认
    7/8/9/10/10.5 dBm、固定 30 Erlang，可用 loads/powers/fixed-load/fixed-power 修改。
    相同负载、种子在各算法和功率下必须有相同到达序列。交叉工况在两组各留一份。
    
    5% 阻塞率是人为选定的实验参考线，不是通用标准，也不是算法接入限制；
    现存材料不足以保证任意种子下 30 Erlang 都低于此线。输出目录必须为空。
    manifest.json 最后生成，用作完成批次的索引；回放、图表和工作簿生成失败时
    可能留有不完整文件，应换新目录重跑，不把目录存在视作成功。
    """
    # 在仿真前检查工作簿和图表所需依赖。
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401

    if args.scan_load or args.scan_scenarios or args.save_samples:
        raise ValueError('--export-business 不与统计扫描模式同时使用')
    if args.topology != 'topology7' or args.link_length_km not in (None, 10):
        raise ValueError('实验业务导出使用 topology7 的 10 km 双向直连链路')
    if args.classical_channels != 7 or args.quantum_channels != 1 or args.include_c33:
        raise ValueError('实验业务导出使用 C35 量子信道和跳过 C33 的七个经典信道')
    if args.algorithm not in ('ALL', 'FF', 'GREEDY_MIN_NOISE'):
        raise ValueError('实验业务仅支持 first-fit/FF 和 GREEDY_MIN_NOISE')
    loads, seeds, warmup, _ = scan_settings(args)
    powers = args.powers if args.powers is not None else [7, 8, 9, 10, 10.5]
    if not math.isfinite(args.fixed_load) or args.fixed_load <= 0:
        raise ValueError('fixed-load 必须为有限正数')
    if not powers or len(powers) != len(set(powers)):
        raise ValueError('powers 必须非空且不能重复')
    for power in [*powers, args.fixed_power]:
        if not math.isfinite(power) or not -100 < power < 100:
            raise ValueError('实验功率必须为 (-100, 100) 内有限 dBm 数值')
    algorithms = ['FF', 'GREEDY_MIN_NOISE'] if args.algorithm == 'ALL' else [args.algorithm]
    output = Path(args.output_dir or base / 'results' / ('business_export_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('业务导出目录必须为空，避免覆盖实验依据')
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(base, 'topology7', args.raman_file)
    experiments = [('load_scan', load, args.fixed_power) for load in loads]
    experiments += [('power_scan', args.fixed_load, power) for power in sorted(powers)]
    metadata = dict(loads_erlang=loads, powers_dbm=sorted(powers), fixed_load_erlang=args.fixed_load,
                    fixed_power_dbm=args.fixed_power, seeds=seeds, slots=args.slots, warmup=warmup,
                    holding_time=args.holding_time, length_km=10, algorithms=algorithms,
                    skr_definition='Time mean of slot-end total SKR after warmup; each quantum channel clipped at zero; undirected links counted once',
                    skr_units='Sweep sheets/figures: kbit/s; Runs and trace JSON: bit/s. Not accumulated secret bits.',
                    uncertainty='Equal-weight seed means; sample SD across seeds, blank for one seed; not a confidence interval',
                    gain_definition='Ratio of seed mean SKR minus one; blank if baseline absent or zero',
                    blank_definition='Unavailable or undefined, not zero',
                    timing='Full warmup and resource transitions retained; experiment duration maps linearly to simulation time',
                    power_reference='Per classical channel at fiber input, dBm', source_sha256=hashes)
    metadata.update(blocking_rate_limit=.05,
                    blocking_definition='Blocked classical arrivals / offered arrivals in [warmup, slots); not packet loss or BER',
                    blocking_limit_definition='User-selected experiment target, strictly below 5%; not an enforced admission rule or universal standard')
    manifest = dict(schema_version=2, kind='qkd_business_experiments',
                    description='10 km; paired traffic across algorithms and powers; full warmup retained',
                    config=metadata, files=[])
    references, runs = {}, []
    for group, load, power in experiments:
        for seed in seeds:
            for algorithm in algorithms:
                sim = build_simulation(base / 'topologies/topology7.json', args.raman_file,
                    algorithm=algorithm, slots=args.slots, arrival_rate=load / args.holding_time,
                    holding_time=args.holding_time, k=args.k, seed=seed,
                    classical_channels=7, quantum_channels=1,
                    launch_power=1e-3 * 10 ** (power / 10), link_length_km=10,
                    core_layout=args.core_layout, skip_c33=True)
                recorder = TrafficRecorder(sim, seed=seed, warmup=warmup)
                metrics, _ = measure_run(sim, warmup, event_recorder=recorder)
                key = (load, seed)
                digest = metrics['traffic_sha256']
                if key in references and references[key] != digest:
                    raise ValueError('不同算法或功率的业务到达序列不一致')
                references[key] = digest
                data = recorder.finish(metrics, hashes)
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
    artifacts = export_business_summary(output, runs, metadata, summarize_business(runs))
    manifest['artifacts'] = [dict(file=name, sha256=sha256((output / name).read_bytes()).hexdigest())
                             for name in artifacts]
    # 最后写入批次索引，表示本批所有输出均已完成。
    write_trace(output / 'manifest.json', manifest)
    print(f'业务导出完成：{output}', flush=True)
    return manifest
