# QKD 共纤资源分配仿真

仿真多芯光纤中经典通信与量子密钥分发（QKD）共存时的资源分配，比较 CQLI、CCA、FF、SCWA 和 GREEDY_MIN_NOISE 的秘密密钥率（SKR）、经典光信噪比（OSNR）、阻塞率，以及相对 first-fit 的有符号协同度。

## 快速开始

安装 Conda 后，在项目根目录执行（Python 3.12）：

```bash
conda env create -f environment.yml
conda activate mcf-resource-allocation
python main.py --algorithm ALL --slots 5 --arrival-rate 2 --seed 53
```

最后一条是短仿真，终端会打印五种算法的阻塞率、SKR、OSNR，以及相对 first-fit 的协同度。默认使用 topology7 的两节点链路（0–1），距离直接读取拓扑文件，每条业务独立等概率选择 0→1 或 1→0，每经典信道功率为10 dBm。正式运行可调整时长、负载和种子。

```bash
# 单个算法
python main.py --algorithm greedy_min_noise --slots 100 --seed 53

# 扫描负载并比较五种算法
python main.py --scan-load --algorithm ALL --loads 10 20 30 --seeds 53 54 55 --slots 100 --warmup 10

# 一条命令完成负载、功率、距离三组扫描
python main.py --scan-all --algorithm ALL --loads 10 20 30 --powers 0 5 10 --distances 1 5 10 20 --fixed-load 10 --launch-power-dbm 10 --seeds 53 54 55 --slots 100 --warmup 10

# 按拓扑文件的实际距离分配三芯业务，导出双向回放及指标
python main.py --export-business --classical-channels 10 --slots 100 --warmup 10 --seed 53

# 调整有限样本估算的总脉冲数和统计波动系数
python main.py --algorithm ALL --key-pulses 1e10 --key-gamma 5.3

# 查看所有参数
python main.py --help
```

扫描及回放自动保存到 `results/` 下的新目录，包含 JSON、Excel 和图表；可用 `--output-dir` 指定新的空目录。普通单次运行只打印结果。单独选择算法时自动补跑 FF 以计算协同度。五算法比较使用 `--scan-load --algorithm ALL`；三芯业务回放比较 FF、CCA 和 GREEDY_MIN_NOISE。普通扫描每组（负载扫描为每个场景）仅输出一张 SVG，按 2×2 展示协同度、OSNR、SKR 和阻塞率；Excel/JSON 保留统计数值与配置。扫描的负载单位为 Erlang，到达率 = 负载 / 平均保持时间；一个时隙是一单位仿真时间。

`--scan-all` 等价于同时指定 `--scan-load --scan-power --scan-distance`，依次运行三个独立扫描，输出到同一批次的 `load_scan/`、`power_scan/`、`distance_scan/`。也可只选其中一组或两组。功率/距离扫描固定 `--fixed-load`（默认10 Erlang）；负载/距离扫描固定 `--launch-power-dbm`。功率默认点为7、8、9、10、10.5 dBm；距离默认点为1、5、10、20、30、40、50 km，并显式覆盖全网边长。负载/功率扫描仍使用拓扑距离，或通过 `--link-length-km` 指定固定距离。各组分别改变一个变量，不运行三维交叉组合。

默认量子信道为 C35，10 个经典信道为 C34、C32、C31、C30、C29、C28、C27、C26、C25、C24；可用 `--classical-channels` 调整数量，三芯回放同样生效。CQLI（参考项目中的 my）、CCA、FF 按参考动态实现对齐；SCWA 保留参考奇偶分组，并修复小频点集合下的阈值与无回退问题。均保留本项目频点与量子预留配置。三芯回放是硬件实验扩展，不等同于参考单芯 FF；执行规则、SCWA 可变信道数适配及噪声含义见 `algorithm.py` 注释。单次和扫描共用单链路指标计算；SKR采用双诱骗态有限样本估算，逐量子信道截零后平均。greedy 的单芯近似同分容差由 `--greedy-noise-rtol` 调整（默认 0.01）。

## 阅读代码

| 文件 | 内容 |
| --- | --- |
| [main.py](main.py) | 参数、默认物理配置、业务生成与资源占用/释放 |
| [algorithm.py](algorithm.py) | 五种算法的选择规则 |
| [noise_calculation.py](noise_calculation.py) / [skr_calculation.py](skr_calculation.py) | 噪声公式、经典 OSNR、参数单位、SKR 计算 |
| [synergistic_calculation.py](synergistic_calculation.py) | 有符号协同度及归一化假设 |
| [traffic_scan.py](traffic_scan.py) | 扫描设置、统计窗口与指标定义 |
| [traffic_export.py](traffic_export.py) | 回放 JSON 格式、Excel 与图表输出 |
| [topology.py](topology.py) / [core_layout.py](core_layout.py) | 拓扑 JSON、候选路径与纤芯编号 |

运行需要 `topologies/` 和根目录的拉曼数据表 `Ramancrosssection25GHz（25GHz间隔）.xls`。`environment.yml` 是依赖安装清单，并非完整环境锁定文件。

`results/` 用于保存运行生成的实验配置、原始数据和图表，程序会自动创建输出目录。修改项目前请阅读唯一规范 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。项目代码采用 MIT 许可证，以 GitHub 仓库已有的 `LICENSE` 为准；第三方材料仍须遵守其原授权。

`--observe-link U V` 使用零起始节点编号，必须是一条真实边；不指定时取拓扑中的最短边；同长时按节点编号字典序选择。
默认 topology7 只有0–1一条链路。普通运行、负载扫描和业务导出均直接使用所选拓扑文件各边的 `length_km`，不缩放到固定距离。`--observe-link` 只选择观测边。`--link-length-km` 和 `--observe-link-length-km` 分别用于显式全边等长及单边覆盖实验，单边覆盖优先。

普通运行默认30 Erlang（到达率7.5、平均保持时间4），普通扫描省略 `--loads` 时也使用30；三芯回放默认遍历5、10、15、20、25、30、35、40 Erlang业务组，保持时间4，到达率为负载除以4；三芯回放的功率扫描固定负载默认10 Erlang业务组（到达率2.5）。两者均为双向合计，不是每方向的负载。三芯组显式负载不再除以三，可用 `--loads` 和 `--fixed-load` 覆盖。业务方向独立随机抽取，不自动生成反向配对业务；显式选择其他拓扑时仍在所选完整网络生成源宿和分配资源。统计定义见 `traffic_scan.py`。

SKR接入用户提供的 `SKR_new.py` 中双诱骗态有限样本估算（信号/诱骗强度0.6/0.2，概率14/16、1/16，余下为真空态）。默认总脉冲数1e10、波动系数5.3；探测器与损耗沿用本项目配置。每个资源快照按静态脉冲块估算，1 GHz下默认块长对应10秒，不是把变化的业务时隙直接合并为安全密钥块；该统计波动近似不声称完整可组合安全。参数与限制见 `skr_calculation.py`，结果元数据记录实际配置。

SKR、OSNR、协同度的参考来源和公式见对应计算脚本。协同度仅在 OSNR 和 SKR 都严格提高时为正，否则非零幅值取负，任一指标相等时为零；SKR 归一化上下限按实际距离和串扰条件计算，协同度绝对值不保证小于 1；结果另存 SKR/OSNR 相对 FF 的差值。历史批次不按新口径改写。

业务回放只包含经过观测链路的已接入业务和该链路两方向的真实资源状态，保留完整路径用于追溯。三芯绑定仍仅用于 `--export-business`。

CCA 三芯实验适配使用量子芯0、前向芯[1,2,3]、后向芯[4,5,6]，按实际频率升序分配；普通扫描仍使用六芯双向共享的 CCA。扫描 SVG 的 SKR 面板标注 greedy 相对 FF 的最大正提升百分比。
