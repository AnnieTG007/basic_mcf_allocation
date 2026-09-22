# QKD 共纤资源分配仿真

仿真多芯光纤中经典通信与量子密钥分发（QKD）共存时的资源分配，比较 CQLI、FF、SCWA 和 GREEDY_MIN_NOISE 的秘密密钥率（SKR）与经典业务阻塞率。

## 快速开始

安装 Conda 后，在项目根目录执行（Python 3.12）：

```bash
conda env create -f environment.yml
conda activate mcf-resource-allocation
python main.py --algorithm ALL --slots 5 --arrival-rate 2 --seed 53
```

最后一条是短仿真，终端会打印四种算法的阻塞率和 SKR。默认拓扑为两节点、10 km 链路，每经典信道功率为 10.5 dBm。正式运行可调整时长、负载和种子。

```bash
# 单个算法
python main.py --algorithm greedy_min_noise --slots 100 --seed 53

# 扫描负载并比较四种算法
python main.py --scan-load --algorithm ALL --loads 10 20 30 --seeds 53 54 55 --slots 100 --warmup 10

# 导出 FF 与 GREEDY_MIN_NOISE 的负载/功率实验和资源状态回放
python main.py --export-business --slots 100 --warmup 10 --seed 53

# 查看所有参数
python main.py --help
```

扫描及回放自动保存到 `results/` 下的新目录，包含 JSON、Excel 和图表；可用 `--output-dir` 指定新的空目录。普通单次运行只打印结果。扫描的负载单位为 Erlang，到达率 = 负载 / 平均保持时间；一个时隙是一单位仿真时间。

默认量子信道为 C35，经典信道为 C34、C32、C31、C30、C29、C28、C27。单次运行沿用历史 SKR 统计，扫描则逐量子信道将负值置零，两者不能混算增益。SCWA 是动态适配，模型范围和算法限制见脚本注释。

## 阅读代码

| 文件 | 内容 |
| --- | --- |
| [main.py](main.py) | 参数、默认物理配置、业务生成与资源占用/释放 |
| [algorithm.py](algorithm.py) | 四种算法的选择规则 |
| [noise_calculation.py](noise_calculation.py) / [skr_calculation.py](skr_calculation.py) | 噪声公式、参数单位、SKR 计算 |
| [traffic_scan.py](traffic_scan.py) | 扫描设置、统计窗口与指标定义 |
| [traffic_export.py](traffic_export.py) | 回放 JSON 格式、Excel 与图表输出 |
| [topology.py](topology.py) / [core_layout.py](core_layout.py) | 拓扑 JSON、候选路径与纤芯编号 |

运行需要 `topologies/` 和根目录的拉曼数据表 `Ramancrosssection25GHz（25GHz间隔）.xls`。`environment.yml` 是依赖安装清单，并非完整环境锁定文件。

`results/` 用于保存运行生成的实验配置、原始数据和图表，程序会自动创建输出目录。修改项目前请阅读唯一规范 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。项目代码采用 MIT 许可证，以 GitHub 仓库已有的 `LICENSE` 为准；第三方材料仍须遵守其原授权。
