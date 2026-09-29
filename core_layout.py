"""供 main 和 algorithm 导入固定七芯配置，并由 cores_code 将芯数、间距和耦合系数转换为邻芯表与矩阵。
几何函数支持 19/37 芯，不代表算法支持这些芯数。"""

import math

import numpy as np



# FF 为 first-fit（首次适配），QCNM 为 Quantum channel noise mitigation（量子信道噪声抑制）；算法缩写全称与规则见 algorithm。
# SEVEN_CORE_LAYOUTS 固定各算法的量子芯及方向芯配置，例如 FF 的量子芯为 (1,)。
# 表中元组顺序就是候选芯顺序，不根据芯数、列表首项或几何邻接推导算法分组。
SEVEN_CORE_LAYOUTS = {
    'CQLI': dict(
        classical_forward=(2, 4, 6),
        classical_backward=(3, 5, 7),
        quantum=(1,),
    ),
    'CCA': dict(
        classical_forward=(3, 4, 5, 6, 7, 1),
        classical_backward=(3, 4, 5, 6, 7, 1),
        quantum=(2,),
    ),
    'FF': dict(
        classical_forward=(2, 3, 4, 5, 6, 7),
        classical_backward=(2, 3, 4, 5, 6, 7),
        quantum=(1,),
    ),
    'SCWA': dict(
        # 方向芯顺序固定为奇数组在前、偶数组在后；奇偶指原信道索引。
        classical_forward=(6, 5, 7),
        classical_backward=(1, 2, 4),
        quantum=(3,),
        forward_odd=(6,),
        forward_even=(5, 7),
        backward_odd=(1,),
        backward_even=(2, 4),
    ),
    'QCNM': dict(
        classical_forward=(4, 5, 6),
        classical_backward=(3, 7, 1),
        quantum=(2,),
    ),
}

# SEVEN_CORE_EXPERIMENT_LAYOUTS 为调用方的固定实验布局，例如 FF 前向绑定 (2, 3, 4)。
# 哪些算法参与实验、是否绑定这些芯由 main 决定，算法模块不读取此表。
SEVEN_CORE_EXPERIMENT_LAYOUTS = {
    'FF': dict(
        classical_forward=(2, 3, 4),
        classical_backward=(5, 6, 7),
        quantum=(1,),
    ),
    'CCA': dict(
        classical_forward=(3, 4, 5),
        classical_backward=(6, 7, 1),
        quantum=(2,),
    ),
    'QCNM': dict(
        classical_forward=(4, 5, 6),
        classical_backward=(3, 7, 1),
        quantum=(2,),
    ),
}


def cores_code(cores_nums, *, core_spacing, first_coupling, secondary_coupling, farthest_coupling):
    """为完整六角形纤芯排布生成一阶、二阶邻芯字典和对称功率耦合系数矩阵。"""
    # cores_nums 为总芯数（如 7）；完整排布须满足 N = 1 + 3*r*(r+1)，r 为圈数。
    if isinstance(cores_nums, bool) or not isinstance(cores_nums, (int, np.integer)) or cores_nums < 7:
        raise ValueError("Core count must be a complete hexagonal layout (7, 19, 37, ...)")
    core_count = int(cores_nums)
    rings = (math.isqrt(12 * core_count - 3) - 3) // 6
    if 1 + 3 * rings * (rings + 1) != core_count:
        raise ValueError("Core count must be a complete hexagonal layout (7, 19, 37, ...)")
    # core_spacing 如 10，长度单位由调用方一致约定；本函数只校验正值，邻接计算与绝对间距无关。
    if not np.isfinite(core_spacing) or core_spacing <= 0:
        raise ValueError("Core spacing must be positive")

    # 在单位间距下生成坐标，避免实际间距较小时，坐标取整导致不同芯重合。
    # 每圈沿六条直边等分，不能把一圈内所有芯均匀放在圆周上。
    # 中心芯编号为 1，逐圈向外编号；每圈从正上方开始顺时针递增。
    vertices = [(math.sin(k * math.pi / 3), math.cos(k * math.pi / 3))
                for k in range(6)]
    coordinates = [(0.0, 0.0)]
    for ring in range(1, rings + 1):
        for side in range(6):
            x1, y1 = vertices[side]
            x2, y2 = vertices[(side + 1) % 6]
            for step in range(ring):
                # 包含边的起点、不包含终点，避免相邻边重复编号顶点。
                coordinates.append((ring * x1 + step * (x2 - x1),
                                    ring * y1 + step * (y2 - y1)))

    first_neighbors = {i: [] for i in range(1, core_count + 1)}
    secondary_neighbors = {i: [] for i in range(1, core_count + 1)}
    # 三档耦合分别对应 1、√3、2 倍芯间距，单位 m^-1，例如 1e-6、1e-7、10**(-7.5)。
    # 对角线及超过第三档的耦合为零；返回矩阵不负责替代噪声模型的耦合参数。
    # 矩阵直接以芯号索引，第 0 行和第 0 列留空，不代表纤芯。
    hij_matrix = np.zeros((core_count + 1, core_count + 1), dtype=np.float64)
    for i in range(1, core_count + 1):
        x1, y1 = coordinates[i - 1]
        for j in range(i + 1, core_count + 1):
            x2, y2 = coordinates[j - 1]
            # 单位芯间距下的无量纲距离平方；1.5/1.9/2.1 阈值分隔 1、√3、2 三档距离。
            distance_squared = (x2 - x1)**2 + (y2 - y1)**2
            if distance_squared < 1.5**2:        # 第一档邻芯：距离为 core_spacing。
                first_neighbors[i].append(j)
                first_neighbors[j].append(i)
                coupling = first_coupling
            elif distance_squared < 1.9**2:
                secondary_neighbors[i].append(j) # 第二档次邻芯：距离为 sqrt(3) 倍芯间距。
                secondary_neighbors[j].append(i)
                coupling = secondary_coupling
            elif distance_squared < 2.1**2:      # 第三档次次邻芯：距离为 2 倍芯间距。
                coupling = farthest_coupling
            else:
                continue
            # 每对芯只计算一次，同时填写对称元素；上述遍历自然保持邻芯升序。
            hij_matrix[i, j] = hij_matrix[j, i] = coupling

    return first_neighbors, secondary_neighbors, hij_matrix
