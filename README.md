# QKD 共纤资源分配仿真

仿真多芯光纤中经典通信与量子密钥分发（QKD）共存时的资源分配，比较 CQLI、CCA、FF、SCWA 和 QCNM（Quantum channel noise mitigation）的秘密密钥率（SKR）、经典光信噪比（OSNR）、阻塞率，以及相对 first-fit 的有符号协同度。

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
python main.py --algorithm QCNM --slots 100 --seed 53

# 扫描负载：默认比较六个 QCNM 容差、CCA、FF
python main.py --scan-load --loads 10 20 30 --seeds 53 54 55 --slots 100 --warmup 10

# 一条命令完成负载、功率、距离三组扫描
python main.py --scan-all --loads 10 20 30 --powers 0 5 10 --distances 1 5 10 20 --fixed-load 10 --launch-power-dbm 10 --seeds 53 54 55 --slots 100 --warmup 10

# 按拓扑文件的实际距离分配三芯业务，导出双向回放及指标
python main.py --export-business --classical-channels 10 --slots 100 --warmup 10 --seed 53

# 自选比较算法和 QCNM 容忍系数；同样适用于 --export-business
python main.py --scan-load --algorithm QCNM CCA first-fit --qcnm-noise-rtol 0 0.2 0.5 --slots 100 --warmup 10

# 调整有限样本估算的总脉冲数和统计波动系数
python main.py --algorithm ALL --key-pulses 1e10 --key-gamma 5.3

# 查看所有参数
python main.py --help
```

扫描及回放自动保存到 `results/` 下的新目录，包含 JSON、Excel 和图表；可用 `--output-dir` 指定新的空目录。普通单次运行只打印结果。

扫描和三芯回放默认比较 QCNM 的六个容忍系数 `0 0.1 0.2 0.3 0.4 0.5`、CCA 和 FF。`--algorithm` 支持多选；普通扫描可选择全部五种算法，`ALL` 展开全部算法。三芯回放仅支持 QCNM、CCA、FF。未选 FF 时内部补跑基准，不额外导出 FF 曲线或回放文件。

`--qcnm-noise-rtol` 支持自选容忍系数列表；普通单次运行默认仅用 0.1。旧算法名 `greedy_min_noise` 和旧参数 `--greedy-noise-rtol` 仍可使用。容忍系数的定义及固定三芯组结果可能相同的原因见 [algorithm.py](algorithm.py)。

`--scan-all` 输出三个独立扫描到 `load_scan/`、`power_scan/`、`distance_scan/`；普通扫描图为四指标 SVG，业务回放图为 PNG/SVG。详细参数见 `python main.py --help`，统计与数据格式见下列脚本注释。

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
