# QKD 共纤资源分配仿真

仿真多芯光纤中经典通信与量子密钥分发（QKD）共存时的资源分配，比较 CQLI（Classical and quantum signal layered interleaved，经典量子信号分层交错资源分配）、CCA（Conventional channel allocation，传统信道分配方案）、FF（first-fit，首次适配）、SCWA（Synergistic core and wavelength allocation，协同纤芯波长分配方案）和 QCNM（Quantum channel noise mitigation，量子信道噪声抑制）的秘密密钥率（SKR）、经典光信噪比（OSNR）、阻塞率，以及相对 FF 的有符号协同度。

当前五种算法使用固定七芯配置，不支持任意芯数或布局覆盖。SCWA 使用参考的固定余量10切换规则，信道奇偶按原索引划分，失败时不回退。

## 快速开始

安装 Conda 后，在项目根目录执行（Python 3.12）：

```powershell
conda env create -f environment.yml
conda activate mcf_resource_allocation
python main.py --algorithm ALL --slots 5 --arrival-rate 2 --seed 42
```

最后一条是短仿真，终端会打印五种算法的阻塞率、SKR、OSNR，以及相对 FF 的协同度。默认使用 topology7 的两节点链路（0–1），距离直接读取拓扑文件，每条业务独立等概率选择 0→1 或 1→0，每经典信道功率为10 dBm。正式运行可调整时长、负载和种子。随机种子优先使用 `42`，多个种子使用 `42 43 44`。

以下命令使用 PowerShell；行末反引号用于续行，其后不要添加空格，可整段复制执行。默认采用有限长密钥估算，无需额外开启。

```powershell
# 单个算法
python main.py --algorithm QCNM --slots 100 --seed 42

# 扫描负载：默认比较六个 QCNM 容差、CCA、FF
python main.py --scan-load `
  --loads 10 20 30 `
  --seeds 42 43 44 `
  --slots 100 `
  --warmup 10

# 一条命令完成负载、功率、距离三组扫描
python main.py --scan-all `
  --loads 5 10 15 20 25 30 35 40 45 50 `
  --powers 0 2 4 6 8 `
  --distances 1 5 10 20 `
  --fixed-load 40 `
  --launch-power-dbm 8 `
  --seeds 42 `
  --slots 100 `
  --warmup 10

# 三芯业务仿真并直接映射为一小时实验回放（含预热）
# 导出 CCA、QCNM 的双向回放及指标
python main.py --export-business `
  --loads 3 6 9 12 15 18 `
  --powers 0 2 4 6 8 `
  --fixed-power 8 `
  --fixed-load 18 `
  --algorithm ALL `
  --topology topology7 `
  --slots 100 `
  --warmup 10 `
  --seed 42 `
  --experiment-duration-seconds 3600

# 自选普通扫描算法和 QCNM 容忍系数；实验导出只接受 CCA、QCNM
python main.py --scan-load `
  --algorithm QCNM CCA FF `
  --qcnm-noise-rtol 0 0.2 0.5 `
  --slots 100 `
  --warmup 10

# 查看所有参数
python main.py --help
```

`--export-business` 一次完成仿真、实验时间映射和导出，回放 JSON 可由实验端直接读取，无需额外转换。`--experiment-duration-seconds 3600` 指每份回放的总时长（含预热），可改为所需秒数；上述 100 个时隙映射为 3600 秒，其中 10 个预热时隙占 360 秒。省略该参数则保留仿真时间单位。每个批次目录含 `config.json`（本次实际生效的全部仿真参数）、`manifest.json`（文件摘要与汇总）和逐次回放 JSON；回放只按时刻列出前后向占用信道，实验端可用 `forward_cores`/`backward_cores` 核对自身设置，字段含义见 [traffic_export.py](traffic_export.py) 顶部说明。`--reexport-traffic` 可把已有回放按新的实验时长另存到指定目录。

扫描及回放自动保存到 `results/` 下的新目录；普通扫描仅导出 Excel 和 SVG 图，实验业务回放另外导出 JSON。可用 `--output-dir` 指定新的空目录。普通单次运行只打印结果。

普通扫描默认比较 QCNM 的容忍系数列表、CCA 和 FF；实验回放默认只导出 CCA 和单个 QCNM 容忍系数（当前为 0.3）。`--algorithm` 支持多选；普通扫描可选择全部五种算法，`ALL` 展开全部算法。实验回放仅允许选择 CCA、QCNM，其 `ALL` 也仅展开这两种算法。未选 FF 时内部补跑基准，不额外导出 FF 曲线或回放文件。

`--qcnm-noise-rtol` 支持自选容忍系数列表；普通单次运行与实验回放默认使用 0.3，扫描默认遍历 0 到 0.5、步长 0.1。默认值只在 [main.py](main.py) 顶部定义一处，本文不再重复具体数字；以 `python main.py --help` 的输出和导出目录中的实际参数为准。算法名称仅使用 `CQLI`、`CCA`、`FF`、`SCWA`、`QCNM`，不保留旧名称或旧参数别名。容忍系数的定义及固定三芯组结果可能相同的原因见 [algorithm.py](algorithm.py)。

`--scan-all` 输出三个独立扫描到 `load_scan/`、`power_scan/`、`distance_scan/`；普通扫描图为四指标 SVG，业务回放图为 PNG/SVG。详细参数见 `python main.py --help`，统计与数据格式见下列脚本注释。

## 阅读代码

| 文件 | 内容 |
| --- | --- |
| [main.py](main.py) | 参数及算法容差组合、默认物理配置、业务生成与资源占用/释放 |
| [algorithm.py](algorithm.py) | 一个 ResourceAllocator 类中的五种算法及共用搜索 |
| [noise_calculation.py](noise_calculation.py) / [skr_calculation.py](skr_calculation.py) | 噪声公式、经典 OSNR、参数单位、SKR 计算 |
| [synergistic_calculation.py](synergistic_calculation.py) | 有符号协同度及归一化假设 |
| [traffic_scan.py](traffic_scan.py) | 执行主控传入的算法组合、扫描设置、统计窗口与指标定义 |
| [traffic_export.py](traffic_export.py) | 回放 JSON 格式、Excel 与图表输出 |
| [topology.py](topology.py) / [core_layout.py](core_layout.py) | 拓扑 JSON、候选路径、固定七芯算法配置及纤芯编号 |

运行需要 `topologies/`。拉曼系数采用内置的 GNPy 3.0.1 默认谱及增益换算方式，不再读取 Excel，也无需安装 GNPy；来源、模型假设、单位和第三方授权直接标在 `noise_calculation.py` 的参数及公式注释中。`environment.yml` 是依赖安装清单，并非完整环境锁定文件。

`results/` 用于保存运行生成的实验配置、原始数据和图表，程序会自动创建输出目录。修改项目前请阅读唯一规范 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。项目代码采用 MIT 许可证，以 GitHub 仓库已有的 `LICENSE` 为准；第三方材料仍须遵守其原授权。
