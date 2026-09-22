"""从命令行启动共纤仿真：生成经典业务、调用分配算法、占用并释放资源。

运行示例：python main.py --algorithm ALL --slots 5 --arrival-rate 2
普通运行在终端打印结果；--scan-load 和 --export-business 交给 traffic_scan
组织实验。其余脚本是被导入的计算模块，直接运行不会启动仿真。

输入为拓扑 JSON、拉曼系数 XLS 和命令行参数。命令行长度用 km、功率用
每经典信道 dBm；内部统一转成 m、W，频率为 Hz，SKR（秘密密钥率）为 bit/s。
时间是仿真单位：一个时隙长 1；到达事件可以发生在时隙内任意时刻。
资源数组顺序为 [源节点, 目的节点, 纤芯, 信道]，0/1/2/3 分别表示
不可用/空闲经典/占用经典/量子保留。信道索引不是 ITU 信道号。
"""
import pandas as pd
import numpy as np
import random
import math
from copy import deepcopy

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from algorithm import ALGORITHMS, ResourceAllocator, normalize_algorithm
from core_layout import cores_code
from skr_calculation import (BB84Parameters, DetectorParameters, QuantumLinkScorer,
                           calculate_SKR_all)
from topology import load_topology, distance_matrix, k_shortest_paths, validate_graph
from noise_calculation import (FiberParameters, MulticoreFiber, RamanSpectrum,
                             NoiseModel, ClassicalFWMScorer)

@dataclass(frozen=True)
class SimulationParameters:
    """单次仿真的输入参数，不在此处生成随机业务。
    
    core_num 为芯数；Ts 为总时隙数；lambda1 为全网络每时间单位到达率；
    rou1 为平均保持时间（不是离去率），两者乘积为负载 Erlang。
    knum 为每对节点最多保留的候选路径数。classical_wave_num 和
    quantum_wave_num 为经典、量子候选信道数；max_frequency/wave_interval
    分别为最高频率和基本网格间隔（Hz）；排除频率后数组可能不再等间隔。
    launch_power 为每经典信道功率 W；seed 固定本实例随机业务序列。
    """
    core_num: int
    Ts: int
    lambda1: float
    rou1: float
    knum: int
    classical_wave_num: int
    quantum_wave_num: int
    max_frequency: float
    wave_interval: float
    launch_power: float
    seed: int
    excluded_classical_frequencies_hz: tuple = ()


class Event:
    """一条到达或离去事件；同一业务的两条事件共享 ID 和分配信息。
    
    m_time/m_holdTime 使用仿真时间单位；m_ocuppiedwave 是唯一信道索引，
    m_ocuppiedcore 按路径各跳排列：普通仿真是单芯编号，实验导出是三芯列表。
    同一索引写入对应的全部芯，每芯功率均为 P；离去事件在成功接入后才建立。
    """
    def __init__(self, launch_power):
        self.m_eventType = pd.Series([0, 0], index=['Arrival', 'End'])                         # 业务的类型(到达或离去)
        self.m_time = 0                                                                             # 业务的到达时间/离去时间
        self.m_holdTime = 0                                                                         # 业务的持续时间
        self.m_id = 0                                                                               # 业务id
        self.m_sourceNode = 0                                                                       # 业务源节点
        self.m_destNode = 0                                                                         # 业务目的节点
        self.m_ocuppiedwave = 0                                                                     # 业务占有波长的编号
        self.m_ocuppiedcore = []                                                                    # 业务占有纤芯的数组
        self.m_workPath = []                                                                        # 业务的完整路由
        self.P = launch_power  # W


class ClassicalService:
    """持有一个仿真实例的资源、功率、候选路径和按时间排序的事件列表。
    
    到达间隔与保持时间独立采样指数分布，源目的节点随机且不同；算法只选择
    资源，资源实际改动集中在 dealWithEvent。每个实例只能推进一次完整仿真。
    """
    def __init__(self, graph, params, *, algorithm, core_config, noise_model,
                 detector_params, bb84_params, classical_fiber,
                 first_neighbors, secondary_neighbors):
        validate_graph(graph)
        self.graph = graph.copy()
        self.MAXINUM = len(graph)
        if self.MAXINUM < 2:
            raise ValueError("Simulation requires at least two nodes")
        core_num, Ts = params.core_num, params.Ts
        lambda1, rou1, knum = params.lambda1, params.rou1, params.knum
        classical_wave_num = params.classical_wave_num
        quantum_wave_num = params.quantum_wave_num
        for name, count in (("core_num", core_num), ("Ts", Ts), ("knum", knum)):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("lambda1", lambda1), ("rou1", rou1),
                            ("launch_power", params.launch_power),
                            ("max_frequency", params.max_frequency),
                            ("wave_interval", params.wave_interval)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.noise_model = noise_model
        self.detector_params = detector_params
        self.bb84_params = bb84_params
        self.first_neighbor = deepcopy(first_neighbors)
        self.secondary_neighbor = deepcopy(secondary_neighbors)
        self.launch_power = params.launch_power
        self.core_num = core_num                                                                    # 纤芯个数，正式构建使用七芯
        for name, count in (("classical_wave_num", classical_wave_num), ("quantum_wave_num", quantum_wave_num)):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.algorithm = normalize_algorithm(algorithm)
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.classical_forward_cores = list(core_config["classical_forward"])
        self.classical_backward_cores = list(core_config["classical_backward"])
        self.quantum_cores = list(core_config["quantum"])
        all_cores = self.classical_forward_cores + self.classical_backward_cores + self.quantum_cores
        if any(not isinstance(c, (int, np.integer)) or not 0 <= c < core_num for c in all_cores):
            raise ValueError("Core indices must be integers in [0, core_num)")
        if len(all_cores) != len(set(all_cores)):
            raise ValueError("Core groups must not contain duplicates or overlap")

        self.classical_wave_num = classical_wave_num
        self.quantum_wave_num = quantum_wave_num
        self.WaveNumber = classical_wave_num + quantum_wave_num                                     #经典与量子波长总数
        self.Ts = Ts                                                                                #时隙个数，
        self.m_currentTime = 0                                                                      #当前的时间
        self.lambda1 = lambda1                                                                      #业务的到达率（产生到达时间）
        self.m_lambda1 = 1.0/self.lambda1                                                           #泊松分布转化为指数分布，到达率要取反
        self.m_rou1 = rou1                                                                          # 指数保持时间的均值，不是离去率
        self.a_m = distance_matrix(self.graph, unit="m")

        self.knum = knum
        self.m_cost = np.full((self.MAXINUM, self.MAXINUM, self.knum), np.inf, dtype=float)
        self.m_path = [[[[]for _ in range(self.knum)]for _ in range(self.MAXINUM)]for _ in range(self.MAXINUM)]
        self.m_resourceMap = np.zeros((self.MAXINUM, self.MAXINUM,self.core_num, self.WaveNumber), dtype=np.float32)
                                                    # 起点，终点，纤芯，波长。1为可用，0为不可用，2为正在被使用，3为量子信道。
        self.P_link = np.zeros((self.MAXINUM, self.MAXINUM, self.core_num, self.WaveNumber),dtype=np.float32)        # 每个波长上的光功率
        self.ServiceQuantity = 0                                                                    #业务总数
        self.m_nextServiceId = 0                                                                    #新一次服务的业务id
        self.m_pq = []                                                                              #事件的容器

        self.max_frequency = params.max_frequency
        self.excluded_classical_frequencies = tuple(params.excluded_classical_frequencies_hz)
        if any(not np.isfinite(f) or f <= 0 for f in self.excluded_classical_frequencies):
            raise ValueError("Excluded frequencies must be finite and positive")
        self.wave_interval = params.wave_interval
        self.rng = random.Random(params.seed)

        self.c_s_amount = []                                                            # 事件队列长度，含下一次到达；通常为活跃业务数加一
        self.s_timeList = []                                                            #各个时刻网络各链路业务量的和，与c_s_amount是不同的
        self.SKR_timeList = []                                                          #各个时刻网络总体的SKR
        self.Link_SKR_timeList = np.zeros((self.Ts, self.MAXINUM, self.MAXINUM))        #这个数组存储每个时刻的网络各个链路的SKR
        self.Link_s_timeList = []                                                       #这个数组存储每个时刻的网络各个链路的业务数量

        self.initialize()  # 初始化网络参数
        scorer = (ClassicalFWMScorer(
            self.available_channel, range(self.quantum_wave_num, self.WaveNumber), classical_fiber
        ) if self.algorithm == "SCWA" else None)
        self.quantum_scorer = QuantumLinkScorer(
            self.available_channel, self.first_neighbor, self.secondary_neighbor,
            self.noise_model, self.detector_params, self.bb84_params)
        self.allocator = ResourceAllocator(self.algorithm, self.classical_forward_cores,
                                           self.classical_backward_cores, scorer, self.quantum_scorer)

    def initialize(self):  # 初始化
        """生成实际频率、标记允许使用的资源，并预先计算候选路径。
        
        量子频率排在数组前端；经典频率继续降序排列，跳过指定排除频率并补足
        候选数。默认量子为 C35，经典为 C34/C32/C31/C30/C29/C28/C27。
        节点号小到大为前向、大到小为后向，仅开放对应方向组的经典芯。
        """
        self.available_channel = [
            self.max_frequency - w * self.wave_interval
            for w in range(self.quantum_wave_num)
        ]
        index = self.quantum_wave_num
        while len(self.available_channel) < self.WaveNumber:
            frequency = self.max_frequency - index * self.wave_interval
            index += 1
            if frequency <= 0:
                raise ValueError("Not enough positive classical frequencies")
            if any(abs(frequency - excluded) < 1e5 for excluded in self.excluded_classical_frequencies):
                continue
            self.available_channel.append(frequency)
        if min(self.available_channel) <= 0:
            raise ValueError("All channel frequencies must be positive")

        # 按所选算法配置纤芯；经典使用后段频率，每个量子纤芯使用前段频率
        # 前向纤芯和后向纤芯也分配好
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
            self.pathCalculate(source)

    def pathCalculate(self, source):
        """按路径总长（km）保存至多 k 条不重复节点的路径；不足的项保持无穷。
        
        m_path 保存节点列表，m_cost 保存路径总长；节点编号直接对应数组下标。
        """
        for target in range(self.MAXINUM):
            for index, (path, cost) in enumerate(
                    k_shortest_paths(self.graph, source, target, self.knum)):
                self.m_path[source][target][index] = path
                self.m_cost[source, target, index] = cost

    def generateServiceEventPair(self, id):  # 生成业务，更新业务列表m_pq
        """生成下一条到达事件并放入时间队列；函数名虽含 Pair，此时尚无离去事件。
        
        以当前事件时刻加指数到达间隔，独立生成保持时间和源目的节点；
        只有这条业务成功占用资源后，dealWithEvent 才创建配对的离去事件。
        """
        event0 = Event(self.launch_power)
        event0.m_eventType["Arrival"] = 1  # 业务到达
        event0.m_eventType["End"] = 0
        event0.m_id = id
        event0.m_time = self.m_currentTime + self.arrive_time_gen(self.m_lambda1)
        event0.m_holdTime = self.arrive_time_gen(self.m_rou1)  # 业务持续时间


        # 为event0事件生成源目的节点
        node = self.randomSrcDst()
        event0.m_sourceNode = node["first"]
        event0.m_destNode = node["second"]
        self.m_pq.append(event0)  # 将event0加入到m_pq中
        self.m_pq.sort(key=lambda x: x.m_time)  # 排序会让先到达的排在前面  #必要步骤  # 排序耗费时间

    def randomSrcDst(self):  # 随机生成源节点和目的节点，且源节点和目的节点不能相同
        """返回随机且不同的源目的节点，first 为源节点，second 为目的节点。"""
        SrcDst = pd.Series([0, 0], index=['first', 'second'])
        SrcDst['second'] = self.genrandom(0, self.MAXINUM - 1)
        while True:
            SrcDst['first'] = self.genrandom(0, self.MAXINUM - 1)
            if SrcDst['first'] != SrcDst['second']:
                break
        return SrcDst


    def genrandom(self, ia, ib):
        """从闭区间 [ia, ib] 均匀抽取一个整数，使用本实例的随机数生成器。"""
        fr = self.rng.randint(ia, ib)
        return fr


    def arrive_time_gen(self,beta):  # 生成业务到达离去时间，都是生成一个服从指数分布的时间差
        """返回均值为 beta 的指数分布时间间隔：-beta * ln(U)。
        
        到达间隔传入 1/lambda1，保持时间传入 rou1；U 取 (0, 1)，排除零以避免 ln(0)。
        """
        while True:
            u = self.rng.random()   # [0,1) 内的随机浮点数
            if u != 0:
                break
        ln = math.log(u)    # 默认底数为e
        x = beta*ln        # 计算时beta=1/lambda1
        x *= (-1)
        return x

    def generateLeavingevent(self, event):

        """复制已接入事件，保留业务 ID、路径和资源，将发生时刻改成到达时刻加保持时间。"""
        event1 = deepcopy(event)  # 保留独立分配信息，避免与到达事件共享可变列表
        event1.m_eventType["End"]=1
        event1.m_eventType["Arrival"]=0
        event1.m_time = event.m_time + event.m_holdTime   # "业务离去事件的发生时间" = "业务到达事件发生的时间" + "业务持续时间"
        return event1


    def dealWithEvent(self, event):

        """处理一条事件并更新队列、资源和功率，不计算 SKR。
        
        到达时按候选路径顺序寻找可用分配：失败计入阻塞，成功把资源状态置 2、
        写入功率并安排离去；两种情况都继续生成下一条到达。离去将资源恢复为 1，
        功率清零。列表按时间稳定排序，同一时刻保留已有插入顺序。
        """
        self.m_currentTime = event.m_time  # 当前时间=业务的到达时间/离去时间
        if event.m_eventType['Arrival'] == 1:
            self.ServiceQuantity += 1

            core_list, reg_wave = self.showPath_core_exchange(event)

            if reg_wave == -1:
                # 业务阻塞，统计阻塞率
                self.m_sumOfFailedService += 1
            else:
                # 为业务分配资源
                event.m_ocuppiedwave = reg_wave
                event.m_ocuppiedcore = core_list
                # 标记链路属性为正在被占用
                for it in range(len(event.m_workPath) - 1):  # link, link, core, wave
                    self.m_resourceMap[event.m_workPath[it], event.m_workPath[it + 1], event.m_ocuppiedcore[it], event.m_ocuppiedwave] = 2
                    self.P_link[event.m_workPath[it], event.m_workPath[it + 1], event.m_ocuppiedcore[it], event.m_ocuppiedwave] = event.P
                # 生成该业务的离去事件
                self.m_pq.append(self.generateLeavingevent(event))

                # 新加入离去事件后，重新按时间排序
                self.m_pq.sort(key=lambda x: x.m_time)

            # 生成下一个到达事件
            self.generateServiceEventPair(self.m_nextServiceId)
            self.m_nextServiceId += 1
            # 删除当前事件
            self.m_pq.remove(event)  # 只移除已处理的当前事件

        # 业务离去事件
        if event.m_eventType['End'] == 1:

            for it in range(len(event.m_workPath) - 1):
                self.m_resourceMap[event.m_workPath[it], event.m_workPath[it + 1], event.m_ocuppiedcore[it], event.m_ocuppiedwave] = 1
                self.P_link[event.m_workPath[it], event.m_workPath[it + 1], event.m_ocuppiedcore[it], event.m_ocuppiedwave] = 0
            self.m_pq.remove(event)  # 删除当前事件


    def showPath_core_exchange(self,event):
        """按候选路径顺序尝试分配，返回 (逐跳芯分配列表, 全路径共同信道索引)。
        
        普通仿真每跳为单芯编号，实验导出每跳为三芯列表；分配规则由 allocator 决定。
        中间节点允许换芯，但不允许换频率；第一条能接入的路径即被选中，
        不跨路径比较噪声。成功时 event.m_workPath 为选中路径，失败返回 (None, -1)。
        """
        s = event.m_sourceNode
        d = event.m_destNode
        for ki in range(self.knum):
            if not np.isfinite(self.m_cost[s][d][ki]):
                return None, -1
            event.m_workPath = []
            for node in self.m_path[s][d][ki]:
                event.m_workPath.append(node)

            if event.m_workPath == []:
                # 没有更多候选路径
                return None, -1
            core_list, w = self.allocator.allocate(
                event.m_workPath, event.P, resources=self.m_resourceMap,
                powers=self.P_link, distances=self.a_m,
            )
            if core_list is not None:
                return core_list, w
        # 所有候选路径都无法分配共同信道，业务阻塞
        return None, -1

    def iter_slots(self, on_event=None):
        """推进新实例的唯一事件循环，逐个返回已完成的零起始时隙。

        同时刻事件保持队列原顺序；时隙右边界的事件在下一时隙处理。
        on_event(event, blocked=...) 在资源更新后调用，只用于观察和记录。
        """
        if self.m_pq or self.ServiceQuantity:
            raise ValueError("Simulation requires a fresh instance")
        self.m_sumOfFailedService = 0
        self.generateServiceEventPair(0)
        self.m_nextServiceId = 1
        for slot in range(self.Ts):
            while self.m_pq[0].m_time < slot + 1:
                event = self.m_pq[0]
                before = self.m_sumOfFailedService
                self.dealWithEvent(event)
                if on_event is not None:
                    on_event(event, blocked=self.m_sumOfFailedService > before)
            yield slot

    def run(self):
        """推进单次仿真并打印阻塞率、网络和每量子信道平均 SKR，不写结果文件。
        
        阻塞率统计全部到达事件；SKR 保留可为负的公式原值。历史 SKR 窗口为
        列表切片 [10:Ts-1]，即零起始时隙 10..Ts-2（采样时刻 11..Ts-1）；
        切片为空时使用全部时隙。此口径不同于 traffic_scan 的预热窗口及非负 SKR。
        """
        for slot in self.iter_slots():
            SKR_at_Ts = calculate_SKR_all(
                resource_map=self.m_resourceMap, powers=self.P_link, distances_m=self.a_m,
                frequencies_hz=self.available_channel, first_neighbors=self.first_neighbor,
                secondary_neighbors=self.secondary_neighbor, noise_model=self.noise_model,
                detector_params=self.detector_params, bb84_params=self.bb84_params,
            )
            self.Link_SKR_timeList[slot] = SKR_at_Ts
            self.SKR_timeList.append(np.sum(SKR_at_Ts))

            #计算每个时刻每个链路的业务量：
            Link_s_amount0 = np.zeros((self.MAXINUM,self.MAXINUM),int)
            for i in range(self.MAXINUM):
                for j in range(self.MAXINUM):
                    for c in range(self.core_num):
                        for w in range(self.WaveNumber):
                            if self.m_resourceMap[i][j][c][w] == 2:
                                Link_s_amount0[i][j] += 1

            Link_s_amount = deepcopy(Link_s_amount0)
            for i in range(self.MAXINUM):
                for j in range(self.MAXINUM):
                    if i < j:
                        Link_s_amount[i][j] += Link_s_amount0[j][i]
                    else:
                        Link_s_amount[i][j] = 0
            self.Link_s_timeList.append(Link_s_amount)
            self.s_timeList.append(np.sum(Link_s_amount))

            self.c_s_amount.append(len(self.m_pq))

        # 所有时隙累积的阻塞率
        print('failed service amount/service amount', self.m_sumOfFailedService / max(1, self.ServiceQuantity))
        # 业务量
        print('erlang:',self.lambda1 * self.m_rou1)
        # 长仿真沿用原统计窗口；短仿真没有预热后样本时，使用全部时隙。
        skr_samples = self.SKR_timeList[10:self.Ts-1] or self.SKR_timeList
        average_SKR = float(np.mean(skr_samples)) if skr_samples else 0.0
        # 与 calculate_SKR_all 一致：仅统计有效链路的上三角，避免双向重复计数。
        quantum_channel_count = sum(
            np.count_nonzero(self.m_resourceMap[i, j] == 3)
            for i in range(self.MAXINUM)
            for j in range(i + 1, self.MAXINUM)
            if self.graph.has_edge(i, j)
        )
        average_channel_SKR = average_SKR / quantum_channel_count if quantum_channel_count else 0.0
        print('SKR of the whole network25',average_SKR)
        print('average SKR of each channel:', average_channel_SKR)
        return


def load_raman_spectrum(path, *, index_center, frequency_step_hz, coefficient_scale):
    """读取 XLS 第一张表的第二列数值，反转顺序并乘 coefficient_scale。
    
    文件列不能含文字表头；index_center 为反转后零频差所在下标，
    frequency_step_hz 为光谱采样间隔。本项目分别传入 300、25e9 和 1e6。
    系数换算沿用现有数据约定，原始表的来源及测量标定需由数据提供者确认。
    """
    import xlrd
    with xlrd.open_workbook(str(path)) as workbook:
        coefficients = np.asarray(workbook.sheets()[0].col_values(1), dtype=float)[::-1]
    return RamanSpectrum(tuple(coefficients * coefficient_scale), index_center, frequency_step_hz)


def build_simulation(topology_path, raman_path, *, algorithm="SCWA", slots=100,
                     arrival_rate=50, holding_time=0.5, k=1, seed=53,
                     classical_channels=7, quantum_channels=1, launch_power=1e-3,
                     link_length_km=None, core_layout=None, skip_c33=True):
    """读取输入文件并返回尚未运行的七芯仿真实例；这里集中放置默认物理参数。
    
    topology_path/raman_path 是文件路径；arrival_rate 为每时间单位到达率，
    holding_time 为平均保持时间，launch_power 为 W（与命令行 dBm 不同）。
    link_length_km 若提供，只覆盖内存中各边长度。core_layout 可独立指定方向
    分组；默认 CQLI 量子芯为 6，其他算法为 0，所有芯编号从零开始。
    Python 接口默认 algorithm=SCWA、launch_power=1e-3 W；命令行另有默认值。
    """
    graph = load_topology(topology_path)
    if link_length_km is not None:
        if not np.isfinite(link_length_km) or link_length_km <= 0:
            raise ValueError('Link length must be finite and positive')
        for a, b in graph.edges:
            graph[a][b]['length_km'] = link_length_km
    # 编号是零起始仿真芯号；CQLI 中心芯 6 为量子芯，其余算法量子芯为外圈 0。
    # 分组列表顺序也是 SCWA 同分时的选芯顺序，不应随意排序。
    core_groups = {
        "CQLI": {"classical_forward": [0, 2, 4], "classical_backward": [1, 3, 5], "quantum": [6]},
        "GREEDY_MIN_NOISE": {"classical_forward": [2, 3, 4], "classical_backward": [1, 5, 6], "quantum": [0]},
        "SCWA": {"classical_forward": [3, 2, 4], "classical_backward": [6, 1, 5], "quantum": [0]},
        "FF": {"classical_forward": [1, 2, 3], "classical_backward": [4, 5, 6], "quantum": [0]},
    }
    algorithm = normalize_algorithm(algorithm)
    if algorithm not in core_groups:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    layout = algorithm if core_layout is None else normalize_algorithm(core_layout)
    if layout not in core_groups:
        raise ValueError(f"Unknown core layout: {core_layout}")
    params = SimulationParameters(
        core_num=7, Ts=slots, lambda1=arrival_rate, rou1=holding_time, knum=k,
        classical_wave_num=classical_channels, quantum_wave_num=quantum_channels,
        max_frequency=193.5e12, wave_interval=100e9, launch_power=launch_power, seed=seed,
        excluded_classical_frequencies_hz=(193.3e12,) if skip_c33 else (),
    )
    detector = DetectorParameters(efficiency=0.1, gate_time=1e-9,
                                  insertion_loss_db=8, rate_hz=1e9)
    bb84 = BB84Parameters(loss_per_m=4.61e-5, dark_count=1e-6, photon_launch=0.1,
                           error_opt=0.01, sifting_efficiency=0.5, correct_error_eff=1.15)
    fiber_params = FiberParameters(
        loss=0.00004605111673958094,
        loss_c=0.00004605111673958094,
        loss_q=0.000046074142297950725,
        D_c=0.000017,
        D_s=56,
        A_eff=7e-11,
        FW=193400000000000,
        c=299792458,
        e3=6.1796e-14,
        n=1.45,
        hmn=1e-9,
        recapture_factor_Rayleigh=0.0015,
        loss_Rayleigh=0.000032,
        width=1.2e-10,
    )
    raman = load_raman_spectrum(raman_path, index_center=300,
                                frequency_step_hz=25e9, coefficient_scale=1e6)
    noise_model = NoiseModel(
        MulticoreFiber(replace(fiber_params, hmn=1e-9)),
        MulticoreFiber(replace(fiber_params, hmn=1e-10)), raman,
    )
    # 这里只取几何邻接关系；返回的耦合矩阵不参与噪声计算，噪声使用上面的 hmn。
    first_neighbors, secondary_neighbors, _ = cores_code(
        params.core_num, core_spacing=10, first_coupling=1e-6,
        secondary_coupling=1e-7, farthest_coupling=10**(-7.5),
    )
    return ClassicalService(
        graph, params, algorithm=algorithm, core_config=core_groups[layout],
        noise_model=noise_model, detector_params=detector, bb84_params=bb84,
        classical_fiber=MulticoreFiber(fiber_params),
        first_neighbors=first_neighbors, secondary_neighbors=secondary_neighbors,
    )


def main(argv=None):
    """解析命令行并选择单次运行、负载扫描或业务回放导出；ALL 的算法范围随模式而定。"""
    parser = argparse.ArgumentParser(description="多芯光纤 QKD 共纤资源分配仿真")
    parser.add_argument("--topology", choices=("topology1", "topology6", "topology7"), default="topology7")
    parser.add_argument("--algorithm", type=normalize_algorithm,
                        choices=(*ALGORITHMS, "ALL"), default="ALL")
    parser.add_argument("--slots", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=50)
    parser.add_argument("--holding-time", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--classical-channels", type=int, default=7)
    parser.add_argument("--quantum-channels", type=int, default=1)
    parser.add_argument("--include-c33", action="store_true", help="Use the legacy contiguous grid for controlled ablation")
    parser.add_argument("--launch-power-dbm", type=float, default=10.5)
    parser.add_argument("--core-layout", choices=("FF", "CQLI", "SCWA"), default=None,
                        help="Override core groups independently of the allocation objective")
    parser.add_argument("--link-length-km", type=float, default=None,
                        help="Override every edge length for controlled sensitivity experiments")
    base = Path(__file__).resolve().parent
    parser.add_argument("--raman-file", type=Path,
                        default=base / "Ramancrosssection25GHz（25GHz间隔）.xls")
    parser.add_argument("--scan-load", action="store_true", help="扫描 A=lambda*E[H] 并导出 Excel、JSON、PNG/SVG")
    parser.add_argument("--loads", type=float, nargs="+", default=None,
                        help="负载/Erlang；普通扫描默认 10 15 20 25 30 35 40，三芯实验默认 3 5 7 8 10 12 13")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="扫描种子列表；省略时仅使用 --seed")
    parser.add_argument("--warmup", type=int, default=None, help="扫描预热时隙，默认10")
    parser.add_argument("--scan-scenarios", nargs="+", default=None, metavar="KM:DBM",
                        help="长度与每信道功率配对，例如 1:13.5 10:10.5；不指定则使用单次参数")
    parser.add_argument("--output-dir", type=Path, default=None, help="扫描输出目录，默认 results/traffic_scan_时间戳")
    parser.add_argument("--save-samples", action="store_true", help="Excel 中附加预热后的逐时隙样本（JSON始终保留）")
    parser.add_argument("--export-business", action="store_true", help="导出负载/功率两组业务回放 JSON，仅 FF 与 GREEDY_MIN_NOISE")
    parser.add_argument("--powers", type=float, nargs='+', help="业务导出的功率扫描点，默认 7 8 9 10 10.5 dBm")
    parser.add_argument("--fixed-load", type=float, default=None, help="实验功率扫描的固定三芯组负载/Erlang（默认 10）")
    parser.add_argument("--fixed-power", type=float, default=10.5, help="实验负载扫描的固定每芯每信道功率/dBm")
    args = parser.parse_args(argv)
    if args.export_business:
        from traffic_scan import run_business_export
        try:
            return run_business_export(args, build_simulation, base)
        except ValueError as exc:
            parser.error(str(exc))
    if args.powers is not None or args.fixed_load is not None or args.fixed_power != 10.5:
        parser.error("--powers/--fixed-load/--fixed-power 需要 --export-business")
    if args.scan_load:
        from traffic_scan import run_load_scan
        try:
            return run_load_scan(args, build_simulation, base)
        except ValueError as exc:
            parser.error(str(exc))
    if any(value is not None for value in (args.loads, args.seeds, args.warmup,
                                          args.scan_scenarios, args.output_dir)) or args.save_samples:
        parser.error("业务扫描参数需要与 --scan-load 一起使用")
    algorithms = ALGORITHMS if args.algorithm == "ALL" else (args.algorithm,)
    for name in algorithms:
        print(f"开始运行算法：{name}，拓扑：{args.topology}")
        sim = build_simulation(
            base / "topologies" / f"{args.topology}.json", args.raman_file,
            algorithm=name, slots=args.slots, arrival_rate=args.arrival_rate,
            holding_time=args.holding_time, k=args.k, seed=args.seed,
            classical_channels=args.classical_channels, quantum_channels=args.quantum_channels,
            launch_power=1e-3 * 10 ** (args.launch_power_dbm / 10),
            link_length_km=args.link_length_km,
            core_layout=args.core_layout, skip_c33=not args.include_c33,
        )
        sim.run()


if __name__ == "__main__":
    main()
