"""生成完整六角形多芯光纤的编号、邻芯列表与耦合矩阵，由 main 构建实例时调用。

输入芯数、芯间距及三档耦合系数，输出两个邻芯字典和对称耦合矩阵。
七芯时外圈为 0..5、中心为 6；编号顺序和距离分类见 cores_code。
正式入口只使用两个邻芯字典，忽略返回矩阵；物理噪声使用 main 中独立
配置的 hmn 耦合系数，修改这里的三档系数不会自动改变正式仿真的噪声。
"""

import math

import numpy as np


def cores_code(cores_nums, *, core_spacing, first_coupling, secondary_coupling, farthest_coupling):
    """返回一阶邻芯字典、二阶邻芯字典和对称功率耦合系数矩阵。

    支持中心一芯、外围完整 r 圈的排布：N = 1 + 3*r*(r+1)，r >= 1，
    即 7、19、37、61……芯。编号从 0 开始，按外圈到内圈排列；
    每圈从正上方顶点顺时针编号，中心芯编号为 N-1。邻芯列表按编号升序。

    三档耦合分别对应距离 R、sqrt(3)*R、2*R，R 为 core_spacing。
    farthest_coupling 保留原参数名，实际表示第三档（2*R），并非所有远芯；
    更远纤芯之间及矩阵对角线的耦合为零。耦合系数单位为 m^-1。
    core_spacing 须为有限正数；规则排布的整体缩放不改变邻接关系，
    耦合系数由调用方传入，本函数不根据间距推算。
    """
    if isinstance(cores_nums, bool) or not isinstance(cores_nums, (int, np.integer)) or cores_nums < 7:
        raise ValueError("Core count must be a complete hexagonal layout (7, 19, 37, ...)")
    core_count = int(cores_nums)
    rings = (math.isqrt(12 * core_count - 3) - 3) // 6
    if 1 + 3 * rings * (rings + 1) != core_count:
        raise ValueError("Core count must be a complete hexagonal layout (7, 19, 37, ...)")
    if not np.isfinite(core_spacing) or core_spacing <= 0:
        raise ValueError("Core spacing must be positive")

    # 在单位间距下生成坐标，避免实际间距较小时，坐标取整导致不同芯重合。
    # 每圈沿六条直边等分，不能把一圈内所有芯均匀放在圆周上。
    vertices = [(math.sin(k * math.pi / 3), math.cos(k * math.pi / 3))
                for k in range(6)]
    coordinates = []
    for ring in range(rings, 0, -1):
        for side in range(6):
            x1, y1 = vertices[side]
            x2, y2 = vertices[(side + 1) % 6]
            for step in range(ring):
                # 包含边的起点、不包含终点，避免相邻边重复编号顶点。
                coordinates.append((ring * x1 + step * (x2 - x1),
                                    ring * y1 + step * (y2 - y1)))
    coordinates.append((0.0, 0.0))

    first_neighbors = {i: [] for i in range(core_count)}
    secondary_neighbors = {i: [] for i in range(core_count)}
    hij_matrix = np.zeros((core_count, core_count), dtype=np.float64)
    for i in range(core_count):
        x1, y1 = coordinates[i]
        for j in range(i + 1, core_count):
            x2, y2 = coordinates[j]
            distance_squared = (x2 - x1)**2 + (y2 - y1)**2
            # 保留原模型的 1.5、1.9、2.1 倍间距分界；比较平方距离即可。
            # 规则格点的前三档距离为 1、sqrt(3)、2，不会落在这些分界上。
            if distance_squared < 1.5**2:
                first_neighbors[i].append(j)
                first_neighbors[j].append(i)
                coupling = first_coupling
            elif distance_squared < 1.9**2:
                secondary_neighbors[i].append(j)
                secondary_neighbors[j].append(i)
                coupling = secondary_coupling
            elif distance_squared < 2.1**2:
                coupling = farthest_coupling
            else:
                continue
            # 每对芯只计算一次，同时填写对称元素；上述遍历自然保持邻芯升序。
            hij_matrix[i, j] = hij_matrix[j, i] = coupling

    return first_neighbors, secondary_neighbors, hij_matrix
