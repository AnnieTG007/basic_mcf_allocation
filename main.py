"""通过 python main.py 读取拓扑和命令行参数，运行共纤仿真并打印指标。
扫描与业务回放由 traffic_scan 编排，文件导出由 traffic_export 完成。"""
import pandas as pd
import numpy as np
import random
import math
from copy import deepcopy

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from algorithm import ALGORITHMS, ResourceAllocator
from core_layout import cores_code, SEVEN_CORE_LAYOUTS, SEVEN_CORE_EXPERIMENT_LAYOUTS
from skr_calculation import BB84Parameters, DetectorParameters, skr_model_config
from synergistic_calculation import add_paired_synergy
from topology import load_topology, distance_matrix, k_shortest_paths, validate_graph
from noise_calculation import MulticoreFiber, calculate_classical_osnr

# QCNM_NOISE_RTOL 是普通单次运行与三芯业务回放共用的默认容忍系数，例如 0.3。
# 扫描遍历的默认列表是 QCNM_NOISE_RTOLS；两者都在主控顶部这一处维护，不要在其他模块另设默认值。
QCNM_NOISE_RTOL = 0.3
# QCNM_NOISE_RTOLS 是扫描遍历的默认容忍系数列表，也是扫描场合该参数的唯一默认来源。
# 现按 0 到 0.5、步长 0.1 遍历；更大的 1、2、3 对协同度仍有提升，但提升速度明显低于 0 到 0.3，故不默认遍历。
QCNM_NOISE_RTOLS = (0, 0.1, 0.2, 0.3, 0.4, 0.5)


def validated_rtols(rtols, *, source):
    """按统一规则校验容忍系数列表并返回列表；source 用于指出出错来源，如命令行或默认表。"""
    # rtols 例如 [0, 0.1, 0.2]；容忍系数无量纲，可大于1，但不允许负数、重复或非有限值。
    if any(not np.isfinite(value) or value < 0 for value in rtols):
        raise ValueError(f'{source} 的 QCNM 容忍系数必须为有限非负数')
    if len(set(rtols)) != len(rtols):
        raise ValueError(f'{source} 的 QCNM 容忍系数不能重复')
    return list(rtols)


@dataclass(frozen=True)
class SimulationParameters:
    """保存单次仿真的规模、业务、频率和资源使用参数。"""
    core_num: int  # 固定七芯，例如 7。
    Ts: int  # 总时隙数，例如 30；每时隙长 1 个仿真时间单位。
    lambda1: float  # 全网双向合计到达率，例如每仿真时间单位 7.5 条。
    rou1: float  # 平均保持时间，例如 4；负载为 lambda1 * rou1，单位 Erlang。
    knum: int  # 每对节点最多保留的候选路径数，例如 1。
    classical_wave_num: int  # 经典候选信道数，例如 10。
    quantum_wave_num: int  # 量子候选信道数，例如 1。
    max_frequency: float  # 最高量子频率，Hz，例如 193.5e12。
    wave_interval: float  # 基本网格间隔，Hz，例如 100e9。
    launch_power: float  # 每芯每经典信道输入功率，W，例如 1e-3。
    seed: int  # 本实例随机业务序列的种子，例如 42。
    excluded_classical_frequencies_hz: tuple = ()  # 排除的经典频率，Hz，例如 (193.3e12,)。
    # allow_bidirectional 为通用传输开关；例如 False 禁止同一纤芯、同一信道内双向同频数据信号同时传输。
    allow_bidirectional: bool = False
    qcnm_noise_rtol: float = QCNM_NOISE_RTOL  # 量子信道噪声抑制（QCNM）的相对噪声容差，无量纲。


class Event:
    """保存一条到达或离去事件；同一业务的两条事件共享编号和分配信息。"""
    def __init__(self, launch_power):
        """建立事件记录，launch_power 为每芯每信道功率（W），例如 1e-3。"""
        # m_eventType 的 Arrival/End 两位互斥，1 表示该事件是对应的到达或离去。
        self.m_eventType = pd.Series([0, 0], index=['Arrival', 'End'])
        self.m_time = 0  # 事件时刻，仿真时间单位，可落在时隙内。
        self.m_holdTime = 0  # 业务保持时间，与 m_time 同单位。
        self.m_id = 0  # 业务编号；同一条业务的两条事件共用此编号。
        self.m_sourceNode = 0  # 业务源节点
        self.m_destNode = 0  # 业务目的节点
        self.m_ocuppiedwave = 0  # 仿真信道索引，不是国际电信联盟（ITU）信道号。
        self.m_ocuppiedcore = []  # 逐跳芯分配：普通业务为芯编号，三芯业务为芯编号列表。
        self.m_workPath = []  # 业务的完整路由
        self.P = launch_power  # 每条业务在本芯本信道的发射功率，单位 W；三芯组内每芯都按此功率加载。


class ClassicalService:
    """管理一个仿真实例的业务事件与资源状态，由 iter_slots 推进一次完整仿真。"""
    def __init__(self, graph, params, *, algorithm, core_config,
                 first_fiber, secondary_fiber,
                 detector_params, bb84_params,
                 first_neighbors, secondary_neighbors, bind_three=False, observe_link=None):
        """用拓扑、SimulationParameters 及固定芯表构建实例，物理模型与邻芯表由调用方提供。"""
        # first_fiber/secondary_fiber 分别为最近邻、次近邻耦合使用的 MulticoreFiber 对象。
        # observe_link 如 (0, 1) 只选观测边；默认取最短边，同长按节点编号选择。
        validate_graph(graph)
        self.graph = graph.copy()
        self.observed_link = tuple(sorted(observe_link)) if observe_link is not None else min(
            ((min(a, b), max(a, b)) for a, b in graph.edges),
            key=lambda edge: (graph[edge[0]][edge[1]]['length_km'], *edge))
        if len(self.observed_link) != 2 or not self.graph.has_edge(*self.observed_link):
            raise ValueError('--observe-link 必须指定拓扑中存在的一条边')
        self.MAXINUM = len(graph)
        if self.MAXINUM < 2:
            raise ValueError("仿真至少需要两个节点")
        # core_num 固定为七芯；与 Ts、knum 一起只检查是否为正整数。
        core_num, Ts = params.core_num, params.Ts
        lambda1, rou1, knum = params.lambda1, params.rou1, params.knum
        classical_wave_num = params.classical_wave_num
        quantum_wave_num = params.quantum_wave_num
        for name, count in (("core_num", core_num), ("Ts", Ts), ("knum", knum),
                            ("classical_wave_num", classical_wave_num),
                            ("quantum_wave_num", quantum_wave_num)):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
                raise ValueError(f"{name} 必须为正整数")
        for name, value in (("lambda1", lambda1), ("rou1", rou1),
                            ("launch_power", params.launch_power),
                            ("max_frequency", params.max_frequency),
                            ("wave_interval", params.wave_interval)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        self.first_fiber = first_fiber
        self.secondary_fiber = secondary_fiber
        self.detector_params = detector_params
        self.bb84_params = bb84_params
        self.first_neighbor = deepcopy(first_neighbors)
        self.secondary_neighbor = deepcopy(secondary_neighbors)
        self.launch_power = params.launch_power
        self.core_num = core_num  # 纤芯个数，正式构建使用七芯
        self.algorithm = algorithm
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.classical_forward_cores = list(core_config["classical_forward"])
        self.classical_backward_cores = list(core_config["classical_backward"])
        self.quantum_cores = list(core_config["quantum"])
        all_cores = self.classical_forward_cores + self.classical_backward_cores + self.quantum_cores
        if any(not isinstance(c, (int, np.integer)) or not 0 <= c < core_num for c in all_cores):
            raise ValueError("芯编号必须为 [0, core_num) 内的整数")
        # 固定表可显式指定两方向共享经典芯；仅禁止组内重复或经典芯与量子芯重叠。
        if (any(len(group) != len(set(group)) for group in (
                self.classical_forward_cores, self.classical_backward_cores, self.quantum_cores))
                or set(self.quantum_cores) & (set(self.classical_forward_cores) | set(self.classical_backward_cores))):
            raise ValueError("芯分组不得重复，经典芯与量子芯不得重叠")

        self.classical_wave_num = classical_wave_num
        self.quantum_wave_num = quantum_wave_num
        self.WaveNumber = classical_wave_num + quantum_wave_num  # 经典与量子信道总数
        self.Ts = Ts  # 时隙个数
        self.m_currentTime = 0  # 当前仿真时刻
        self.lambda1 = lambda1  # 全网双向合计到达率，用于抽样到达间隔
        self.m_lambda1 = 1.0/self.lambda1  # 到达间隔的指数分布均值为到达率的倒数。
        self.m_rou1 = rou1  # 指数保持时间的均值，不是离去率
        self.a_m = distance_matrix(self.graph, unit="m")

        self.knum = knum
        self.m_cost = np.full((self.MAXINUM, self.MAXINUM, self.knum), np.inf, dtype=float)  # 候选路径总长，km。
        self.m_path = [[[[]for _ in range(self.knum)]for _ in range(self.MAXINUM)]for _ in range(self.MAXINUM)]
        # 数组轴为 [源节点, 目的节点, 芯, 信道]；状态 0/1/2/3 为不可用/空闲/占用/量子保留。
        self.m_resourceMap = np.zeros((self.MAXINUM, self.MAXINUM,self.core_num, self.WaveNumber), dtype=np.float32)

        # P_link 为每芯每信道已占用功率，单位 W，数组轴与资源状态一致。
        self.P_link = np.zeros((self.MAXINUM, self.MAXINUM, self.core_num, self.WaveNumber),dtype=np.float32)
        self.ServiceQuantity = 0  # 已到达业务总数
        self.m_nextServiceId = 0  # 下一条业务的编号
        self.m_pq = []  # 按事件时刻排序的事件队列

        self.max_frequency = params.max_frequency
        self.excluded_classical_frequencies = tuple(params.excluded_classical_frequencies_hz)
        if any(not np.isfinite(f) or f <= 0 for f in self.excluded_classical_frequencies):
            raise ValueError("被排除的经典频率必须为有限正数")
        self.wave_interval = params.wave_interval
        self.rng = random.Random(params.seed)

        self.initialize()
        # bind_three 是主控的资源绑定要求，例如 True 时每条业务占用固定的三个方向芯。
        self.bind_three = bind_three
        if bind_three and (len(self.classical_forward_cores) != 3
                           or len(self.classical_backward_cores) != 3
                           or len(set(self.classical_forward_cores + self.classical_backward_cores)) != 6):
            raise ValueError('三芯实验要求两个互不重叠的三芯方向组')
        # forward_groups/backward_groups 为算法可选择的原子组，例如 [(0,), (2,)] 或 [(0, 2, 4)]。
        forward_groups = ([tuple(self.classical_forward_cores)] if bind_three
                          else [(core,) for core in self.classical_forward_cores])
        backward_groups = ([tuple(self.classical_backward_cores)] if bind_three
                           else [(core,) for core in self.classical_backward_cores])
        # allow_bidirectional 由通用仿真参数指定，默认 False，不由算法或纤芯布局决定。
        self.allow_bidirectional = params.allow_bidirectional
        self.allocator = ResourceAllocator(
            self.algorithm, forward_groups, backward_groups, self.available_channel,
            allow_bidirectional=self.allow_bidirectional)
        # allocation_options 为所选算法的专用输入，例如 QCNM 的 0.1 容差和物理模型与邻芯表；其他算法为空。
        self.allocation_options = (
            dict(noise_rtol=params.qcnm_noise_rtol,
                 first_fiber=self.first_fiber, secondary_fiber=self.secondary_fiber,
                 first_neighbors=self.first_neighbor,
                 secondary_neighbors=self.secondary_neighbor)
            if self.algorithm == 'QCNM' else {})

    def initialize(self):
        """生成量子和经典信道频率，开放固定方向芯的资源，并预先计算候选路径。"""
        # C35/C33 为 ITU 信道标记，分别对应 193.5/193.3 THz。
        # 量子频率占数组前段，不应用经典排除表；多量子信道仍可能包含 C33。
        self.available_channel = [
            self.max_frequency - w * self.wave_interval
            for w in range(self.quantum_wave_num)
        ]
        classical_frequencies = []
        # 经典频率分布在量子段两侧；奇数个信道时低频侧多一个，排除后继续补足数量。
        high_count = self.classical_wave_num // 2
        # 高频从最高量子频率上方开始；低频从最低量子频率下方开始，避免重叠。
        for count, index, step in (
                (high_count, -1, -1),
                (self.classical_wave_num - high_count, self.quantum_wave_num, 1)):
            accepted = 0
            while accepted < count:
                frequency = self.max_frequency - index * self.wave_interval
                index += step
                if frequency <= 0:
                    raise ValueError("没有足够的正频率可作为经典信道")
                # 频差小于 100 kHz 视为命中排除频率；保留既有比较容差。
                if any(abs(frequency - excluded) < 1e5 for excluded in self.excluded_classical_frequencies):
                    continue
                classical_frequencies.append(frequency)
                accepted += 1
        # 数组内顺序即信道编号：量子信道在前，经典信道按频率降序。
        # 搜索顺序由算法决定：FF/CCA/CQLI 按实际频率升序（故实际从最低频开始），
        # SCWA 与 QCNM 按数组内的原索引 0、1、2……升序。
        self.available_channel.extend(sorted(classical_frequencies, reverse=True))
        if min(self.available_channel) <= 0:
            raise ValueError("所有信道频率必须为正数")

        # 节点号小到大为前向，反之为后向；前后向同芯时，本开关禁止两者在同一信道内同时传输。
        for i in range(self.MAXINUM):
            for j in range(self.MAXINUM):
                if i<j and self.graph.has_edge(i, j):
                    for c in self.classical_forward_cores:
                        for w in range(self.quantum_wave_num, self.WaveNumber):
                            self.m_resourceMap[i][j][c][w] = 1
                if i>j and self.graph.has_edge(i, j):
                    for c in self.classical_backward_cores:
                        for w in range(self.quantum_wave_num, self.WaveNumber):
                            self.m_resourceMap[i][j][c][w] = 1

        for i in range(self.MAXINUM):
            for j in range(self.MAXINUM):
                if i != j and self.graph.has_edge(i, j):
                    for c in self.quantum_cores:
                        for w in range(self.quantum_wave_num):
                            self.m_resourceMap[i][j][c][w] = 3

        for source in range(self.MAXINUM):
            self.path_calculate(source)

    def path_calculate(self, source):
        """按路径总长（km）保存至多 k 条简单路径，未找到的路径代价保持无穷。"""
        # m_path 保存节点列表，m_cost 保存 km 路径总长；节点编号直接作为数组下标。
        for target in range(self.MAXINUM):
            for index, (path, cost) in enumerate(
                    k_shortest_paths(self.graph, source, target, self.knum)):
                self.m_path[source][target][index] = path
                self.m_cost[source, target, index] = cost

    def generate_service_event(self, service_id):
        """生成下一条到达事件并放入时间队列；成功接入后才创建配对的离去事件。"""
        event = Event(self.launch_power)
        event.m_eventType["Arrival"] = 1  # 业务到达
        event.m_eventType["End"] = 0
        event.m_id = service_id
        # 到达间隔与保持时间分别独立抽样，尚未生成离去事件。
        event.m_time = self.m_currentTime + self.arrive_time_gen(self.m_lambda1)
        event.m_holdTime = self.arrive_time_gen(self.m_rou1)  # 业务持续时间

        node = self.random_src_dst()
        event.m_sourceNode = node["first"]
        event.m_destNode = node["second"]
        self.m_pq.append(event)
        self.m_pq.sort(key=lambda x: x.m_time)  # 稳定排序使同刻事件保留插入顺序。

    def random_src_dst(self):
        """等概率抽取不同的源目的节点，返回 first（源）和 second（目的）。"""
        # 两节点时每次独立选择 0→1 或 1→0，不额外生成反向配对业务。
        src_dst = pd.Series([0, 0], index=['first', 'second'])
        src_dst['second'] = self.genrandom(0, self.MAXINUM - 1)
        while True:
            src_dst['first'] = self.genrandom(0, self.MAXINUM - 1)
            if src_dst['first'] != src_dst['second']:
                break
        return src_dst

    def genrandom(self, ia, ib):
        """从闭区间 [ia, ib] 均匀抽取一个整数，使用本实例的随机数生成器。"""
        return self.rng.randint(ia, ib)

    def arrive_time_gen(self, mean_interval):
        """返回均值为 mean_interval 的指数分布时间间隔；到达间隔传入到达率倒数，保持时间传入 rou1。"""
        while True:
            u = self.rng.random()  # 排除零，避免逆变换采样计算 ln(0)。
            if u != 0:
                break
        # 逆变换采样：间隔 = -均值 * ln(u)，u 为 (0, 1] 上的均匀随机数。
        return -mean_interval * math.log(u)

    def generate_leaving_event(self, event):
        """复制已接入事件，保留业务编号、路径和资源，将发生时刻改成到达时刻加保持时间。"""
        leaving = deepcopy(event)  # 深拷贝保留独立的分配信息，避免与到达事件共享可变列表
        leaving.m_eventType["End"] = 1
        leaving.m_eventType["Arrival"] = 0
        leaving.m_time = event.m_time + event.m_holdTime  # 离去时刻 = 到达时刻 + 保持时间
        return leaving

    def deal_with_event(self, event):
        """处理到达或离去事件，更新资源、功率和后续事件队列。"""
        self.m_currentTime = event.m_time  # 当前时刻 = 本事件的到达或离去时刻
        if event.m_eventType['Arrival'] == 1:
            self.ServiceQuantity += 1
            core_list, reg_wave = self.show_path_core_exchange(event)
            if reg_wave == -1:
                # 无可用共同信道即阻塞；阻塞业务不占用资源，也不生成离去事件。
                self.m_sumOfFailedService += 1
            else:
                event.m_ocuppiedwave = reg_wave
                event.m_ocuppiedcore = core_list  # 普通业务为单个芯编号，三芯业务为逐跳芯编号列表。
                # 同一跳的绑定芯共用信道，每芯写入完整功率 P，而非按芯数平分。
                for hop in range(len(event.m_workPath) - 1):
                    self.m_resourceMap[event.m_workPath[hop], event.m_workPath[hop + 1], event.m_ocuppiedcore[hop], event.m_ocuppiedwave] = 2
                    self.P_link[event.m_workPath[hop], event.m_workPath[hop + 1], event.m_ocuppiedcore[hop], event.m_ocuppiedwave] = event.P
                self.m_pq.append(self.generate_leaving_event(event))
                self.m_pq.sort(key=lambda x: x.m_time)
            # 接入或阻塞都继续生成下一条到达，保持算法间的输入业务序列一致。
            self.generate_service_event(self.m_nextServiceId)
            self.m_nextServiceId += 1
            self.m_pq.remove(event)  # 只移除已处理的当前事件

        # 与离去事件对称：接入只写了实际传输方向的资源，反向共享芯的格子本就未被改写。
        # 仅恢复实际占用方向；反向共享资源无需改写。
        if event.m_eventType['End'] == 1:
            for hop in range(len(event.m_workPath) - 1):
                self.m_resourceMap[event.m_workPath[hop], event.m_workPath[hop + 1], event.m_ocuppiedcore[hop], event.m_ocuppiedwave] = 1
                self.P_link[event.m_workPath[hop], event.m_workPath[hop + 1], event.m_ocuppiedcore[hop], event.m_ocuppiedwave] = 0
            self.m_pq.remove(event)

    def show_path_core_exchange(self, event):
        """按候选路径顺序寻找分配，返回逐跳芯分配和共同信道索引；失败返回 (None, -1)。"""
        s = event.m_sourceNode
        d = event.m_destNode
        # 首条可接入路径立即采用，不跨路径比较噪声；中间节点可换芯但不能换信道。
        for ki in range(self.knum):
            if not np.isfinite(self.m_cost[s][d][ki]):
                return None, -1  # 没有更多候选路径
            event.m_workPath = list(self.m_path[s][d][ki])
            core_list, w = self.allocator.allocate(
                event.m_workPath, event.P, resources=self.m_resourceMap,
                powers=self.P_link, distances=self.a_m, **self.allocation_options,
            )
            if core_list is not None:
                # 算法统一返回芯组，如 [[0], [2]]；主控转换为事件所需的单芯或多芯格式：
                # 普通业务每跳取组内唯一芯号，三芯业务保留整组。
                return [group if self.bind_three else group[0] for group in core_list], w
        # 所有候选路径都无法分配共同信道，业务阻塞
        return None, -1

    def iter_slots(self, on_event=None):
        """推进唯一事件循环，逐个返回已完成的零起始时隙；on_event 仅用于观察资源更新后的事件。"""
        if self.m_pq or self.ServiceQuantity:
            raise ValueError("仿真必须使用未运行过的实例")
        self.m_sumOfFailedService = 0
        self.generate_service_event(0)
        self.m_nextServiceId = 1
        # 时隙右边界事件留到下一时隙；同刻事件保持队列插入顺序。
        for slot in range(self.Ts):
            while self.m_pq[0].m_time < slot + 1:
                event = self.m_pq[0]
                before = self.m_sumOfFailedService
                self.deal_with_event(event)
                if on_event is not None:
                    on_event(event, blocked=self.m_sumOfFailedService > before)
            yield slot

    def measure_classical_osnr(self):
        """返回所选链路的经典 OSNR 及功率统计字典，不修改资源。"""
        return calculate_classical_osnr(
            self.m_resourceMap, self.P_link, self.a_m,
            self.first_neighbor, self.secondary_neighbor,
            self.observed_link, self.first_fiber, self.secondary_fiber)

    def run(self):
        """复用扫描采样并打印秘密密钥率（SKR）等指标；超过 10 时隙时预热 10 时隙，否则全部采样。"""
        from traffic_scan import measure_run
        row, _ = measure_run(self, 10 if self.Ts > 10 else 0)
        row['algorithm'] = self.algorithm
        row['skr_model'] = skr_model_config(self.bb84_params, self.detector_params)
        print('SKR model:', row['skr_model']['model'],
              'pulses:', self.bb84_params.pulse_count,
              'gamma:', self.bb84_params.fluctuation_gamma,
              'stationary block (s):', row['skr_model']['block_duration_s'])
        print('observed link:', self.observed_link, 'length (m):', row['observed_length_m'])
        print('network blocking rate:', row['blocking_rate'])
        print('average link SKR per quantum channel (bit/s):', row['skr_mean'])
        print('average link classical OSNR (dB):', row['osnr_db_mean'])
        return row


def build_simulation(topology_path, *, algorithm="SCWA", slots=100,
                     arrival_rate=7.5, holding_time=4, k=1, seed=42,
                     classical_channels=10, quantum_channels=1, launch_power=1e-3,
                     link_length_km=None, skip_c33=True,
                     qcnm_noise_rtol=QCNM_NOISE_RTOL, bind_three=False, allow_bidirectional=False, observe_link=None,
                     observe_link_length_km=None, key_pulses=1e10, key_gamma=5.3):
    """读取 topology_path 指定的拓扑，返回尚未运行的七芯仿真实例。
业务与频率参数沿用 SimulationParameters 的含义，launch_power 使用 W。"""
    # 默认使用拓扑文件的真实边长；显式覆盖只修改内存图，且在计算路由前生效。
    graph = load_topology(topology_path)
    if link_length_km is None:
        graph.graph['length_scaling'] = dict(mode='topology', factor=1.0)
    else:
        graph.graph['length_scaling'] = dict(mode='uniform_override', length_km=link_length_km)
        if not np.isfinite(link_length_km) or link_length_km <= 0:
            raise ValueError('Link length must be finite and positive')
        for a, b in graph.edges:
            graph[a][b]['length_km'] = link_length_km
    # 全边覆盖后选最短观测边，同长按节点编号；选边本身不改变长度。
    selected = tuple(sorted(observe_link)) if observe_link is not None else min(
        ((min(a, b), max(a, b)) for a, b in graph.edges),
            key=lambda edge: (graph[edge[0]][edge[1]]['length_km'], *edge))
    if len(selected) != 2 or not graph.has_edge(*selected):
        raise ValueError('--observe-link 必须指定拓扑中存在的一条边')
    # 单边覆盖（km）优先于全边覆盖；覆盖后不重新选择观测边。
    selected_length = observe_link_length_km
    if selected_length is not None:
        if not np.isfinite(selected_length) or selected_length <= 0:
            raise ValueError('观测边长度必须为有限正数')
        graph[selected[0]][selected[1]]['length_km'] = selected_length
        graph.graph['length_scaling']['observed_edge_override'] = dict(link=list(selected), length_km=selected_length)
    # layout_table 是固定七芯配置表，例如普通算法用 SEVEN_CORE_LAYOUTS；实验表由主控选择。
    layout_table = SEVEN_CORE_EXPERIMENT_LAYOUTS if bind_three else SEVEN_CORE_LAYOUTS
    if algorithm not in layout_table:
        raise ValueError(f"算法 {algorithm} 未配置在当前七芯实验表中")
    # core_config 是本次固定配置，例如 FF 的 quantum=(6,)，不根据芯数或列表位置推导布局。
    core_config = layout_table[algorithm]
    # bind_three 选择实验表的固定三芯方向组；量子芯仍取表中配置，算法只支持七芯。
    params = SimulationParameters(
        core_num=7, Ts=slots, lambda1=arrival_rate, rou1=holding_time, knum=k,
        classical_wave_num=classical_channels, quantum_wave_num=quantum_channels,
        max_frequency=193.5e12, wave_interval=100e9, launch_power=launch_power, seed=seed,
        excluded_classical_frequencies_hz=(193.3e12,) if skip_c33 else (),
        qcnm_noise_rtol=qcnm_noise_rtol, allow_bidirectional=allow_bidirectional,
    )
    # 探测参数含义与单位见 DetectorParameters；命令行功率在调用前由 dBm 换为 W。
    detector = DetectorParameters(efficiency=0.2, gate_time=1e-9,
                                  insertion_loss_db=8, rate_hz=50e6)
    # key_pulses 为静态块脉冲数，key_gamma 为高斯波动倍数；1e10 脉冲在 50 MHz 下为 200 秒。
    # 此物理块时长与仿真时隙无换算关系；诱骗态强度与概率取 BB84Parameters 的默认信号、弱诱骗与真空态。
    bb84 = BB84Parameters(loss_per_m=4.61e-5, dark_count=1e-6,
                           error_opt=0.01, sifting_efficiency=0.5, correct_error_eff=1.15,
                           pulse_count=key_pulses, fluctuation_gamma=key_gamma)
    # 参数定义与单位见 MulticoreFiber；原公式中未确认的单位继续保留待确认说明。
    fiber = MulticoreFiber(
        loss=0.00004605111673958094,
        loss_c=0.00004605111673958094,
        loss_q=0.000046074142297950725,
        D_c=0.000017,
        D_s=56,
        A_eff=7e-11,
        reference_frequency=193.4e12,
        c=299792458,
        e3=6.1796e-14,
        n=1.45,
        hmn=1e-9,
        recapture_factor_Rayleigh=0.0015,
        loss_Rayleigh=0.000032,
        width=1.2e-10,
        temperature=300.0,  # K，室温近似；不是实验测量值。
    )
    # 最近邻/次近邻的 hmn 分别沿用 1e-9/1e-10 m^-1，不是几何函数返回的耦合矩阵。
    first_fiber = fiber
    secondary_fiber = replace(fiber, hmn=1e-10)
    # 这里只取几何邻接关系；返回的耦合矩阵不参与噪声计算，噪声使用上面的 hmn。
    first_neighbors, secondary_neighbors, _ = cores_code(
        params.core_num, core_spacing=10, first_coupling=1e-6,
        secondary_coupling=1e-7, farthest_coupling=10**(-7.5),
    )
    return ClassicalService(
        graph, params, algorithm=algorithm, core_config=core_config,
        first_fiber=first_fiber, secondary_fiber=secondary_fiber,
        detector_params=detector, bb84_params=bb84,
        first_neighbors=first_neighbors, secondary_neighbors=secondary_neighbors, bind_three=bind_three, observe_link=selected,
    )


def comparison_plan(args, default=('QCNM', 'CCA', 'FF'), *, three_core=False):
    """生成算法与单个容忍系数的运行组合，label 用于结果标识，export 标记是否导出。"""
    # supported 是用户可选的导出算法，例如三芯实验仅 CCA/QCNM；FF 只在下方补为内部基准。
    supported = ('CCA', 'QCNM') if three_core else ALGORITHMS
    selected = list(dict.fromkeys(args.algorithm or default))
    if selected == ['ALL']:
        selected = list(supported)
    if any(name not in supported for name in selected):
        raise ValueError('实验导出仅支持 CCA、QCNM；其他算法请使用普通运行或扫描')
    # 只在主控展开容忍系数；算法和扫描执行器均接收确定的运行组合。
    variants = []
    for name in selected:
        for rtol in args.qcnm_noise_rtol if name == 'QCNM' else [None]:
            variants.append(dict(algorithm=name,
                label=f'QCNM(rtol={rtol!r})' if name == 'QCNM' else name,
                qcnm_noise_rtol=rtol, export=True))
    # 未选 FF 时补跑内部基准，用于配对指标，但不额外导出曲线或回放。
    if 'FF' not in selected:
        variants.append(dict(algorithm='FF', label='FF', qcnm_noise_rtol=None, export=False))
    return variants


def main(argv=None):
    """解析命令行并选择单次运行、负载/功率/距离扫描或业务回放导出；ALL 的算法范围随模式而定。"""
    parser = argparse.ArgumentParser(description="多芯光纤 QKD 共纤资源分配仿真")
    parser.add_argument("--topology", choices=("topology1", "topology6", "topology7"), default="topology7")
    parser.add_argument('--observe-link', type=int, nargs=2, metavar=('U', 'V'),
                        help='只观测此无向链路，节点从0编号；默认观测拓扑中的最短边（同长按节点编号排序），全网照常分配')
    parser.add_argument("--algorithm",
                        choices=(*ALGORITHMS, "ALL"), nargs="+", default=None,
                        help="可多选；扫描默认 QCNM CCA FF，实验导出默认 CCA QCNM，ALL 选择当前模式全部算法")
    parser.add_argument('--allow-bidirectional', action='store_true',
                        help='允许同一纤芯、同一信道内双向同频数据信号同时传输（默认不允许，与算法无关）')
    parser.add_argument("--slots", type=int, default=30,
                        help="仿真总时隙数，默认30；每时隙长 1 个仿真时间单位")
    parser.add_argument("--arrival-rate", type=float, default=7.5,
                        help="普通运行的双向合计到达率，默认7.5；保持时间4时为30 Erlang")
    parser.add_argument("--holding-time", type=float, default=4, help="业务平均保持时间，默认4个仿真时间单位")
    parser.add_argument("--k", type=int, default=1, help="每对节点保留的候选路径数，默认1")
    parser.add_argument("--seed", type=int, default=42, help="业务序列随机种子，默认42；多个种子用 42 43 44")
    parser.add_argument("--classical-channels", type=int, default=10,
                        help="经典候选信道数，默认10，量子频段两侧各5个；奇数时低频侧多一个，跳过C33")
    parser.add_argument("--quantum-channels", type=int, default=1, help="量子候选信道数，默认1")
    parser.add_argument("--include-c33", action="store_true", help="允许低频侧使用C33，保留两侧分布及经典信道总数")
    parser.add_argument("--launch-power-dbm", type=float, default=10, help="普通运行的每芯每经典信道发射功率/dBm，默认10")
    parser.add_argument("--link-length-km", type=float, default=None,
                        help="把全网每条边的长度统一覆盖为该值/km，用于受控敏感性实验；默认使用拓扑文件中的实际边长")
    parser.add_argument('--observe-link-length-km', type=float, default=None,
                        help='显式覆盖观测边长度/km；默认直接使用拓扑文件中的实际边长')
    base = Path(__file__).resolve().parent
    parser.add_argument("--scan-load", action="store_true", help="扫描 A=lambda*E[H] 并导出 Excel 和四指标对比 SVG")
    parser.add_argument("--scan-power", action="store_true", help="固定负载和距离，扫描每芯每信道功率")
    parser.add_argument("--scan-distance", action="store_true", help="固定负载和功率，扫描全网统一边长")
    parser.add_argument("--scan-all", action="store_true", help="一次运行负载、功率、距离三组独立扫描")
    parser.add_argument("--distances", type=float, nargs="+", help="距离扫描点/km，默认 1 5 10 20 30 40 50")
    parser.add_argument("--loads", type=float, nargs="+", default=None,
                        help="双向合计负载/Erlang；普通扫描默认30，三芯回放默认5 10 15 20 25 30 35 40业务组")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="扫描种子列表；省略时仅使用 --seed")
    parser.add_argument("--warmup", type=int, default=None, help="扫描预热时隙，默认10")
    parser.add_argument("--scan-scenarios", nargs="+", default=None, metavar="KM:DBM",
                        help="全网边长与每信道功率配对，例如 1:13.5 10:10.5；不指定则使用单次参数")
    parser.add_argument("--output-dir", type=Path, default=None, help="扫描输出目录，默认 results/traffic_scan_时间戳")
    parser.add_argument("--export-business", action="store_true", help="运行 CCA、QCNM 的负载/功率两组仿真并导出回放；配合 --experiment-duration-seconds 直接映射为秒")
    parser.add_argument("--powers", type=float, nargs='+', help="功率扫描点，默认 7 8 9 10 10.5 dBm")
    parser.add_argument("--fixed-load", type=float, default=None, help="功率/距离扫描固定负载/Erlang（默认10）；仅业务导出按三芯组计数")
    parser.add_argument("--fixed-power", type=float, default=10.5, help="实验负载扫描的固定每芯每信道功率/dBm")
    parser.add_argument('--qcnm-noise-rtol', dest='qcnm_noise_rtol',
                        type=float, nargs='+', default=None,
                        help=('QCNM 相对噪声增量容差，须为有限非负数，可多选且可大于1；'
                              '1、2 分别允许比最低噪声增量高 100%%、200%%；'
                              f'省略时扫描遍历 {QCNM_NOISE_RTOLS}，普通运行与业务导出都用 {QCNM_NOISE_RTOL}'))
    parser.add_argument("--key-pulses", type=float, default=1e10,
                        help="有限样本SKR的总发射脉冲数，默认1e10；与仿真时隙数无关")
    parser.add_argument("--key-gamma", type=float, default=5.3,
                        help="诱骗态计数高斯波动标准差倍数，默认5.3；不是可组合安全参数")
    parser.add_argument('--experiment-duration-seconds', type=float, default=None,
                        help='每份业务回放的总时长/秒（包含同比例预热）；export-business 中直接完成映射，省略则保留仿真时间单位')
    parser.add_argument('--reexport-traffic', type=Path, default=None,
                        help='离线重导出现有业务JSON，不运行仿真；配合实验秒数和output-dir')
    args = parser.parse_args(argv)
    if args.experiment_duration_seconds is not None:
        if not math.isfinite(args.experiment_duration_seconds) or args.experiment_duration_seconds <= 0:
            parser.error('--experiment-duration-seconds 必须为有限正数')
        if not (args.export_business or args.reexport_traffic):
            parser.error('--experiment-duration-seconds 需要 --export-business 或 --reexport-traffic')
    if args.reexport_traffic is not None:
        if args.export_business or args.scan_load or args.scan_power or args.scan_distance or args.scan_all:
            parser.error('--reexport-traffic 不能与仿真导出或扫描模式混用')
        if args.output_dir is None:
            parser.error('--reexport-traffic 需要 --output-dir')
        import json
        from traffic_export import rescale_times, write_trace
        try:
            data = json.loads(args.reexport_traffic.read_text(encoding='utf-8'))
            # 未指定秒数时按原时长直接另存，便于改目录或改名而不改内容。
            if args.experiment_duration_seconds is not None:
                data = rescale_times(data, args.experiment_duration_seconds)
            target = args.output_dir / args.reexport_traffic.name
            write_trace(target, data)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        print(f'离线回放导出完成：{target.resolve()}')
        return data
    if not math.isfinite(args.key_pulses) or args.key_pulses < 1 or not args.key_pulses.is_integer():
        parser.error("--key-pulses 必须为有限正整数，可使用1e10形式")
    if not math.isfinite(args.key_gamma) or args.key_gamma < 0:
        parser.error("--key-gamma 必须为有限非负数")
    if args.algorithm and 'ALL' in args.algorithm and len(args.algorithm) > 1:
        parser.error('ALL 不能与具体算法混用')
    if args.scan_all:
        args.scan_load = args.scan_power = args.scan_distance = True
    scanning = args.scan_load or args.scan_power or args.scan_distance
    # 容忍系数默认值分两种场合：扫描遍历整张列表，普通单次运行与业务导出共用同一个单点默认值；
    # 用户显式传入时以命令行为准，各条路径都按同一规则校验。
    if args.qcnm_noise_rtol is not None:
        source = '--qcnm-noise-rtol'
        explicit_rtols = args.qcnm_noise_rtol
    elif scanning:
        source = 'main.py 中的 QCNM_NOISE_RTOLS'
        explicit_rtols = QCNM_NOISE_RTOLS
    else:
        source = 'main.py 中的 QCNM_NOISE_RTOL'
        explicit_rtols = [QCNM_NOISE_RTOL]
    try:
        args.qcnm_noise_rtol = validated_rtols(explicit_rtols, source=source)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        variants = comparison_plan(
            args, default=(('CCA', 'QCNM') if args.export_business
                           else ('QCNM', 'CCA', 'FF') if scanning else ALGORITHMS),
            three_core=args.export_business)
    except ValueError as exc:
        parser.error(str(exc))
    if args.export_business:
        if scanning or args.distances is not None:
            parser.error("--export-business 不与普通负载/功率/距离扫描同时使用")
        from traffic_scan import run_business_export
        try:
            return run_business_export(args, build_simulation, base, variants=variants)
        except ValueError as exc:
            parser.error(str(exc))
    if args.fixed_power != 10.5:
        parser.error("--fixed-power 需要 --export-business；普通扫描使用 --launch-power-dbm")
    if args.powers is not None and not args.scan_power:
        parser.error("--powers 需要 --scan-power 或 --scan-all")
    if args.distances is not None and not args.scan_distance:
        parser.error("--distances 需要 --scan-distance 或 --scan-all")
    if args.fixed_load is not None and not (args.scan_power or args.scan_distance):
        parser.error("--fixed-load 需要功率或距离扫描")
    if args.loads is not None and not args.scan_load:
        parser.error("--loads 需要负载扫描；功率/距离扫描使用 --fixed-load")
    if scanning:
        from traffic_scan import run_traffic_scans
        try:
            return run_traffic_scans(args, build_simulation, base, variants=variants)
        except ValueError as exc:
            parser.error(str(exc))
    if any(value is not None for value in (args.loads, args.seeds, args.warmup,
                                          args.scan_scenarios, args.output_dir)):
        parser.error("业务扫描参数需要与 --scan-load/--scan-power/--scan-distance/--scan-all 一起使用")
    results = []
    for variant in variants:
        name = variant["algorithm"]
        label = variant["label"]
        print(f"开始运行算法：{label}，拓扑：{args.topology}")
        sim = build_simulation(
            base / "topologies" / f"{args.topology}.json",
            algorithm=name, slots=args.slots, arrival_rate=args.arrival_rate,
            holding_time=args.holding_time, k=args.k, seed=args.seed,
            classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
            launch_power=1e-3 * 10 ** (args.launch_power_dbm / 10),
            link_length_km=args.link_length_km,
            skip_c33=not args.include_c33,
            # QCNM 使用本变体的容忍系数；补跑的 FF 基准为 None，统一填 0.0，FF 本身不使用该参数。
            qcnm_noise_rtol=variant["qcnm_noise_rtol"] or 0.0, allow_bidirectional=args.allow_bidirectional,
            observe_link=args.observe_link,
            observe_link_length_km=args.observe_link_length_km,
            key_pulses=args.key_pulses, key_gamma=args.key_gamma,
        )
        row = sim.run()
        row.update(algorithm=label, qcnm_noise_rtol=variant["qcnm_noise_rtol"])
        results.append(row)
    add_paired_synergy(results, ())
    for row in results:
        if row['algorithm'] != 'FF':
            value = row['synergy_vs_FF']
            print(f"{row['algorithm']} synergy vs FF:",
                  f'{value:.6g}' if value is not None else 'n/a (requires FF and valid metrics)')
    return results


if __name__ == "__main__":
    main()
