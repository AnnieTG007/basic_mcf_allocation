"""由 main 构建光纤模型，供 skr_calculation 计算量子噪声、main 计算经典光信噪比（OSNR）。
输入频率（Hz）、距离（m）和功率（W），返回噪声功率或线性信噪比，不负责资源分配。"""
from itertools import permutations, combinations
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


# GNPy v3.0.1 的标准单模光纤（SSMF）默认拉曼谱，原始精度不变。
# 来源：https://github.com/Telecominfraproject/oopt-gnpy/blob/fd32bf544310ac76172224bf2c2054a7d4e6d87a/gnpy/core/parameters.py
RAMAN_MODEL = 'gnpy_ssmf_thermal_v1'
GNPY_RAMAN_COMMIT = 'fd32bf544310ac76172224bf2c2054a7d4e6d87a'
_RAMAN_REFERENCE_HZ = 206184634112792.0  # 原谱参考泵浦频率（1454 nm），不是本光纤面积的参考频率。
_RAMAN_OFFSET_HZ = np.array([  # 泵浦与目标的频差采样点，下面由 THz 转为 Hz。
    0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5,
    9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 12.75, 13., 13.25, 13.5, 14., 14.5,
    14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25, 18.5, 18.75, 19., 19.5, 20.,
    20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27.,
    27.5, 28., 28.5, 29., 29.5, 30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34.,
    34.5, 35., 35.5, 36., 36.5, 37., 37.5, 38., 38.5, 39., 39.5, 40., 40.5, 41., 41.5, 42.
]) * 1e12
_RAMAN_GAMMA_M_PER_W = np.array([  # 与频差逐项对应的拉曼增益谱，单位 m/W。
    0.0, 8.524419934705497e-16, 2.643567866245371e-15, 4.410548410941305e-15,
    6.153422961291078e-15, 7.484924703044943e-15, 8.452060808349209e-15, 9.101549322698156e-15,
    9.57837595158966e-15, 1.0008642675474562e-14, 1.0865773569905647e-14, 1.1300776305865833e-14,
    1.2143238647099625e-14, 1.3231065750676068e-14, 1.4624900971525384e-14, 1.6013330554840492e-14,
    1.7458119359310242e-14, 1.9320241330434762e-14, 2.1720395392873534e-14, 2.4137337406734775e-14,
    2.628163218460466e-14, 2.8041019963285974e-14, 2.9723155447089933e-14, 3.129353531005888e-14,
    3.251796163324624e-14, 3.3198839487612773e-14, 3.329527690685666e-14, 3.313155691238456e-14,
    3.289013852154548e-14, 3.2458917188506916e-14, 3.060684277937575e-14, 3.2660349473783173e-14,
    2.957419109657689e-14, 2.518894321396672e-14, 1.734560485857344e-14, 9.902860761605233e-15,
    7.219176385099358e-15, 6.079565990401311e-15, 5.828373065963427e-15, 7.20580801091692e-15,
    7.561924351387493e-15, 7.621152352332206e-15, 6.8859886780643254e-15, 5.629181047471162e-15,
    3.679727598966185e-15, 2.7555869742500355e-15, 2.4810133942597675e-15, 2.2160080532403624e-15,
    2.1440626024765557e-15, 2.33873070799544e-15, 2.557317929858713e-15, 3.039839048226572e-15,
    4.8337165515610065e-15, 5.4647431818257436e-15, 5.229187813711269e-15, 4.510768525811313e-15,
    3.3213473130607794e-15, 2.2602577027996455e-15, 1.969576495866441e-15, 1.5179853954188527e-15,
    1.2953988551200156e-15, 1.1304672156251838e-15, 9.10004390675213e-16, 8.432919922183503e-16,
    7.849224069008326e-16, 7.827568196032024e-16, 9.000514440646232e-16, 1.3025926460013665e-15,
    1.5444108938497558e-15, 1.8795594063060786e-15, 1.7796130169921014e-15, 1.5938159865046653e-15,
    1.1585522355108287e-15, 8.507044444633358e-16, 7.625404663756823e-16, 8.14510750925789e-16,
    9.047944693473188e-16, 9.636431901702084e-16, 9.298633899602105e-16, 8.349739503637023e-16,
    7.482901278066085e-16, 6.240794767134268e-16, 5.00652535687506e-16, 3.553373263685851e-16,
    2.0344217706119682e-16, 1.4267522642294203e-16, 8.980016576743517e-17, 2.9829068181832594e-17,
    1.4861959129014824e-17, 7.404482113326137e-18
])
_RAMAN_OFFSET_HZ.setflags(write=False)
_RAMAN_GAMMA_M_PER_W.setflags(write=False)


def raman_model_config():
    """导出拉曼谱来源与计算约定；实际温度、面积和滤波宽度随光纤参数保存。"""
    return dict(model=RAMAN_MODEL, source="GNPy 3.0.1 DEFAULT_RAMAN_COEFFICIENT",
                source_commit=GNPY_RAMAN_COMMIT,
                source_url=f"https://github.com/Telecominfraproject/oopt-gnpy/blob/{GNPY_RAMAN_COMMIT}/gnpy/core/parameters.py",
                interpolation="linear", spectrum_reference_frequency_hz=_RAMAN_REFERENCE_HZ,
                frequency_offset_range_hz=[0.0, float(_RAMAN_OFFSET_HZ[-1])],
                polarization_modes=2, coefficient_unit="m^-2",
                thermal_factors="Stokes: n+1; anti-Stokes: n; zero offset: analytic limit",
                propagation="existing inter-core forward/backward weak-scattering integrals")


@dataclass(frozen=True)
class MulticoreFiber:
    """保存一组光纤参数，计算四波混频（FWM）、自发拉曼散射（SpRS）和线性芯间串扰（ICXT）。
    FWM 与芯间传播公式沿用原实现，完整出处尚待确认。"""
    loss: float  # FWM 使用的通用衰减，m^-1。
    loss_c: float  # 经典光衰减，m^-1。
    loss_q: float  # 量子光衰减，m^-1。
    D_c: float  # 原公式的色散参数，单位约定待确认。
    D_s: float  # 原公式的色散斜率，单位约定待确认。
    A_eff: float  # 参考频率处的有效模场面积，m²，如 80e-12。
    reference_frequency: float  # 模场面积的参考光频，Hz，如 193.1e12。
    c: float  # 光速，m/s，如 3e8。
    e3: float  # 原 FWM 公式的三阶非线性参数，单位约定待确认。
    n: float  # 无量纲折射率，如 1.45。
    hmn: float  # 两芯间的功率耦合系数，m^-1，如 1e-9。
    recapture_factor_Rayleigh: float  # 瑞利散射光的俘获比例，无量纲，如 0.01。
    loss_Rayleigh: float  # 瑞利散射系数，m^-1。
    width: float  # 接收滤波器的波长带宽，m，如 0.1e-9。
    temperature: float  # 拉曼热占据数使用的温度，K，如 300。

    def __post_init__(self):
        """拒绝非有限参数，以及不满足正值或非负约束的物理参数。"""
        if not all(np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Fiber parameters must be finite")
        if any(getattr(self, name) <= 0 for name in
               ("loss", "loss_c", "loss_q", "A_eff", "reference_frequency", "c", "n", "width", "temperature")):
            raise ValueError("Fiber loss, area, frequency, speed, index, bandwidth and temperature must be positive")
        if any(getattr(self, name) < 0 for name in
               ("e3", "hmn", "recapture_factor_Rayleigh", "loss_Rayleigh")):
            raise ValueError("Nonlinearity and coupling/scattering coefficients must be nonnegative")

    # FWM（四波混频）：相位失配、双向功率与目标信道汇总。
    def get_phase_matching_factor(self, fi, fj, fk):
        """由泵浦频率 fi、fj、fk（Hz，如 193.1e12）计算相位失配 beta（m^-1）。
        返回失配量，不是无量纲转换效率。"""
        f = fi + fj - fk  # FWM 生成频率，Hz。
        w = self.c / f  # 生成光的波长，m。
        beta = 2 * np.pi * w ** 2 / self.c * np.abs(fi - fk) * np.abs(fj - fk) \
               * (self.D_c + w ** 2 / 2 / self.c * (np.abs(fi - fk) + np.abs(fj - fk)) * self.D_s)
        return beta

    def get_intercore_four_wave_mixing(self, fi, fj, fk, pi, pj, pk, beta, z):
        """返回同向泵浦经芯间耦合产生的（FWM 频率 Hz，功率数组 W）。
        fi/fj/fk 和 beta 同相位失配接口，pi/pj/pk 为泵浦功率（W，如 1e-3），z 为 NumPy 距离数组（m）。"""
        # z 示例：np.array([1000.0])；调用方保证距离、功率非负及输入形状匹配。
        # 简并因子 D：两泵浦频差小于 1 MHz 时取 3，否则取 6。
        if np.abs(fi - fj) < 10 ** 6:
            D = 3
        else:
            D = 6

        f_fwm = fi + fj - fk  # 四波混频频率

        # I 为含相位振荡的传播积分项，C 为积分常数；共同计入衰减和失配。
        I = np.exp(-self.loss * z) / (self.loss ** 2 + beta ** 2) \
            * (beta * np.sin(beta * z) - self.loss * np.cos(beta * z))
        C = (beta ** 2 - 3 * self.loss ** 2) / (2 * self.loss * (self.loss ** 2 + beta ** 2))

        p_fwm = self.hmn * np.exp(-self.loss * z) / (self.loss ** 2 + beta ** 2) \
                * 256 * np.pi ** 4 * (2 * np.pi * f_fwm) ** 2 / (self.n ** 4 * self.c ** 4) \
                * (D * self.e3) ** 2 * pi * pj * pk / self.A_eff ** 2 \
                * (-np.exp(-2 * self.loss * z) / 2 / self.loss - 2 * I + z + C)

        return f_fwm, p_fwm

    def get_backward_intercore_four_wave_mixing(self, fi, fj, fk, pi, pj, pk, beta, z):
        """返回反向泵浦经瑞利散射及芯间耦合产生的（频率 Hz，功率数组 W）。
        输入含义与单位同前向 FWM 接口。"""
        # 原反向公式的简并容差为 1e-5 Hz，与前向不同，保留该约定。
        if np.abs(fi - fj) < 1e-5:
            D = 3
        else:
            D = 6
        f = fi + fj - fk
        w = self.c / f
        # M 汇集泵浦功率与非线性系数，S 为反向传播积分中的衰减、相位项。
        M = 1024 * np.pi ** 6 / (self.n ** 4 * w ** 2 * self.c ** 2) * (
                D * self.e3) ** 2 / self.A_eff ** 2 * pi * pj * pk
        S = -(np.exp(-4 * self.loss * z)) / (4 * self.loss) - (2 * np.exp(-3 * self.loss * z)) / (
                9 * self.loss ** 2 + beta ** 2) * \
            (beta * np.sin(beta * z) - 3 * self.loss * np.cos(beta * z)) - np.exp(-2 * self.loss * z) / (2 * self.loss)
        p = self.recapture_factor_Rayleigh * self.loss_Rayleigh * self.hmn * M / (self.loss ** 2 + beta ** 2) \
            * (S * z - np.exp(-4 * self.loss * z) / (16 * self.loss ** 2) \
               + 2 * np.exp(-3 * self.loss * z) / (9 * self.loss ** 2 + beta ** 2) ** 2 * (
                       (9 * self.loss ** 2 - beta ** 2) * np.cos(beta * z) - 6 * self.loss * beta * np.sin(
                   beta * z)) \
               - np.exp(-2 * self.loss * z) / (4 * self.loss ** 2) + 1 / (16 * self.loss ** 2) - 2 * (
                       9 * self.loss ** 2 - beta ** 2) / \
               (9 * self.loss ** 2 + beta ** 2) ** 2 + 1 / (4 * self.loss ** 2))
        return f, p

    def _iter_fwm_inputs(self, frequencies, powers):
        """从等长的一维 frequencies（Hz）和 powers（W）中筛出非零泵浦，按原顺序生成频率、功率六元组。"""
        active = np.nonzero(powers)
        frequencies = frequencies[active]
        powers = powers[active]
        for i, j, k in self.get_fwm_permutations(len(frequencies)):
            yield (frequencies[i], frequencies[j], frequencies[k],
                   powers[i], powers[j], powers[k])

    def _evaluate_fwm(self, inputs, function, z):
        """由 inputs 六元组求相位失配，再调用 function 指定的前向或反向 FWM 公式。"""
        beta = self.get_phase_matching_factor(*inputs[:3])
        return function(*inputs, beta, z)

    def get_fwm_power(self, ls_f_class, ls_p_class, ls_f_quantum, function, z: np.ndarray):
        """汇总落入目标频点的 FWM，返回（目标频率副本，功率数组 W）。
        无有效组合时功率为零，一个生成频点匹配多个目标时报错。"""
        # ls_f_class/ls_p_class：等长一维泵浦频率 Hz/功率 W；ls_f_quantum：目标频率 Hz。
        # function 选择前/后向公式，z 为距离数组 m；输出按 [目标频点, 距离] 排列。
        out_f = ls_f_quantum.copy()
        out_p = np.zeros((len(out_f), len(z)), dtype=np.float32)  # 保留原累加精度。
        # 仅匹配离散生成频率，容差 100 kHz；不模拟展宽、漂移或跨芯泵浦混合。
        for inputs in self._iter_fwm_inputs(ls_f_class, ls_p_class):
            frequency = inputs[0] + inputs[1] - inputs[2]
            matches = np.where(np.abs(out_f - frequency) < 1e5)[0]
            if len(matches) > 1:
                raise RuntimeError('计算出的有多个匹配的值。')
            if len(matches) == 1:
                _, power = self._evaluate_fwm(inputs, function, z)
                out_p[matches, :] += [power]
        return out_f, out_p

    @staticmethod
    def get_fwm_permutations(n):
        """为 n 个泵浦（如 3）生成 fi+fj-fk 的索引组合，排除 k=i 或 k=j 的平凡项。"""
        index_list = np.arange(n)
        fwm_permutations = []  # 如 (0, 1, 2)，末项为被减频率的索引。
        # 先枚举三泵浦组合；交换 i/j 不产生新项，只保留一次。
        for c1 in combinations(index_list, 3):
            for c2 in combinations(c1, 2):
                h = []
                for v in c2:
                    h.append(v)
                for v2 in c1:
                    if v2 not in c2:
                        h.append(v2)
                        break
                fwm_permutations.append(tuple(h))

        # 再加入 i=j、k 不同的二泵浦组合，保持原累加顺序。
        for c1 in permutations(index_list, 2):
            h = [c1[0], c1[0], c1[1]]
            fwm_permutations.append(tuple(h))

        return fwm_permutations

    # SpRS（自发拉曼散射）：谱系数、双向功率与目标信道汇总。
    @lru_cache(maxsize=4096)
    def get_raman_eta(self, f_pump, f_signal):
        """由泵浦 f_pump 和目标 f_signal（标量 Hz，如 193.1e12）求自发拉曼波长谱系数 eta（m^-2）。
        采用 GNPy 通用单模光纤谱与热占据数近似，尚未经本多芯光纤实验标定。"""
        if not (math.isfinite(f_pump) and math.isfinite(f_signal)) or min(f_pump, f_signal) <= 0:
            raise ValueError("Raman frequencies must be finite and positive")
        offset = abs(f_pump - f_signal)  # 频差 Hz；超过内置谱的 42 THz 范围时拒绝外推。
        if offset > _RAMAN_OFFSET_HZ[-1]:
            raise ValueError("Frequency offset exceeds the GNPy Raman spectrum (42 THz)")
        # 沿用 GNPy 的 4.2 μm 芯半径，将其模场面积缩放公式化简为下式。
        core_area = math.pi * (4.2e-6) ** 2
        def mode_area(frequency):
            """返回光频 frequency（Hz）处的有效模场面积（m²），超出近似适用域时报错。"""
            denominator = core_area / self.A_eff + math.log(frequency / self.reference_frequency)
            if denominator <= 0:
                raise ValueError("Frequency is outside the GNPy mode-area approximation")
            return core_area / denominator
        overlap = (mode_area(f_pump) + mode_area(f_signal)) / 2  # 两光场的平均重叠面积，m²。
        # GNPy Fiber.cr 含反向能量转移的频率比，取幅值后两侧均对应较高光频。
        scale = max(f_pump, f_signal) / (_RAMAN_REFERENCE_HZ * overlap)
        h, k = 6.62607015e-34, 1.380649e-23  # 普朗克常数 J·s、玻尔兹曼常数 J/K。
        if offset == 0:
            # 零频差时增益趋零、热占据数发散，使用二者乘积的解析极限。
            gain_thermal = (_RAMAN_GAMMA_M_PER_W[1] / _RAMAN_OFFSET_HZ[1]
                            * scale * k * self.temperature / h)
        else:
            gamma = float(np.interp(offset, _RAMAN_OFFSET_HZ, _RAMAN_GAMMA_M_PER_W))  # 插值增益，m/W。
            x = h * offset / (k * self.temperature)  # 频差能量与热能之比，无量纲。
            # exp(-x) 避免低温时 exp(x) 上溢；expm1 保留近零频差精度。
            thermal = 1 / (-math.expm1(-x))
            if f_signal > f_pump:
                # 斯托克斯侧用 n+1，反斯托克斯侧用 n；后者是本项目补充。
                thermal *= math.exp(-x)
            gain_thermal = gamma * scale * thermal
        # 两偏振噪声从每 Hz 转为每 m 波长；传播公式再乘一次 width，采用窄带近似。
        return 2 * h * f_signal * gain_thermal * f_signal ** 2 / self.c

    def get_inter_forward_raman_scatter(self, p, z, eta):
        """返回芯间前向拉曼功率（W）；p 为泵浦功率（W），z 为距离（m），eta 为谱系数（m^-2）。"""
        # 原式要求 loss_q-loss_c 及 loss_q-loss_c-2*hmn 非零，未处理这两个奇点。
        return eta * p * np.exp(-self.loss_q * z) * (
                (np.exp((self.loss_q - self.loss_c) * z) - 1) / (self.loss_q - self.loss_c)
                - (np.exp((self.loss_q - self.loss_c - 2 * self.hmn) * z) - 1) / (
                        self.loss_q - self.loss_c - 2 * self.hmn)) * self.width

    def get_inter_backward_raman_scatter(self, p, z, eta):
        """返回芯间后向拉曼功率（W），输入含义与单位同前向拉曼接口。"""
        return eta * p * ((np.exp(-(self.loss_q + self.loss_c + 2 * self.hmn) * z) - 1) / (
                self.loss_q + self.loss_c + 2 * self.hmn)
                          - (np.exp(-(self.loss_q + self.loss_c) * z) - 1) / (self.loss_q + self.loss_c)) * self.width

    def get_raman_power(self, ls_f: np.array, ls_p: np.array, ls_quantum: np.ndarray,
                        function, z: np.ndarray):
        """按原泵浦顺序累加拉曼噪声，返回 [目标频点, 距离] 功率数组（W），无泵浦时为零。"""
        # ls_f/ls_p 为等长一维频率 Hz/功率 W，ls_quantum 为目标频率 Hz，z 为距离数组 m。
        # function 选择前/后向传播公式；每个非零泵浦均贡献噪声，不按 FWM 容差筛频。
        z = np.asarray(z)
        active = np.nonzero(ls_p)[0]
        out_p = np.zeros((len(ls_quantum), len(z)), dtype=float)
        for target_index, signal in enumerate(ls_quantum):
            for pump_index in active:
                eta = self.get_raman_eta(ls_f[pump_index], signal)
                out_p[target_index] += function(ls_p[pump_index], z, eta)
        return out_p

    # ICXT 原式的 km 与 km^-1 换算因子抵消，统一使用 m 与 m^-1。
    def get_forward_icxt_power(self, distance, power):
        """返回前向串扰功率（W）；distance 为标量距离（m，如 1000），power 为源芯同频功率（W，如 1e-3）。"""
        return (power * math.exp(-self.hmn * distance) * math.sinh(self.hmn * distance)
                * math.exp(-self.loss_c * distance))

    def get_backward_icxt_power(self, distance, power):
        """返回反向串扰功率（W），输入含义与单位同前向 ICXT 接口。"""
        # 保留原式：使用共享衰减、散射与耦合参数，不额外乘 FWM 的瑞利俘获比例。
        loss = self.loss_c
        return self.loss_Rayleigh * power * self.hmn * (
            (1 - math.exp(-2 * loss * distance)) / loss
            - 2 * distance * math.exp(-2 * loss * distance))

    def __str__(self):
        """返回包含芯间耦合系数的简短光纤说明。"""
        return 'Multicore, hmn={}/m'.format(self.hmn)


@dataclass(frozen=True)
class NoiseModel:
    """组合最近邻、次近邻两组光纤参数，共用内置 GNPy 拉曼谱，远芯不参与量子噪声汇总。"""
    first_fiber: MulticoreFiber  # 最近邻芯使用的光纤参数。
    secondary_fiber: MulticoreFiber  # 次近邻芯使用的光纤参数。


class ClassicalOSNRScorer:
    """计算单链路两方向经典占用信道的线性 OSNR，空闲时返回 None。
    噪声仅含同频 ICXT 和固定噪声底，不含 FWM 或拉曼。"""
    def __init__(self, noise_model):
        """接收 NoiseModel 中的两组邻芯光纤参数，并保存最近一次评分的统计量。"""
        self.noise_model = noise_model
        self.statistics = {}

    def __call__(self, resources, powers, distances, first, secondary, link):
        """汇总 link（如 (0, 1)）两方向的接收信号和噪声，返回线性功率比并更新 statistics。"""
        # resources/powers 按 [源, 目的, 芯, 信道] 排列；状态 2 表示经典占用，功率单位 W。
        # distances 为节点间距离矩阵 m；first/secondary 按芯索引列出最近邻/次近邻芯。
        # signal_sum/xt_sum 为占用格信号/串扰总功率 W，count/zero_count 为占用格/零串扰格数。
        signal_sum = xt_sum = 0.0
        count = zero_count = 0
        a, b = link
        length = float(distances[a, b])
        for i, j in ((a, b), (b, a)):
            for c, w in np.argwhere(resources[i, j] == 2):  # c 为芯编号，w 为信道索引。
                noise = 0.0
                for neighbors, fiber in ((first[c], self.noise_model.first_fiber),
                                         (secondary[c], self.noise_model.secondary_fiber)):
                    for neighbor in neighbors:
                        # 前向项同样由受扰芯本方向功率门控，再使用邻芯功率计算串扰。
                        if powers[i, j, c, w] != 0:
                            noise += fiber.get_forward_icxt_power(length, float(powers[i, j, neighbor, w]))
                        # 沿用参考规则：受扰芯反向同频有光时，才计入邻芯的反向串扰。
                        if powers[j, i, c, w] != 0:
                            noise += fiber.get_backward_icxt_power(length, float(powers[j, i, neighbor, w]))
                # 接收信号沿用固定 0.2 dB/km 损耗，length 的单位为 m。
                signal_sum += float(powers[i, j, c, w]) * 10 ** (-length * 0.2 * 1e-4)
                xt_sum += noise
                count += 1
                zero_count += int(noise == 0)
        noise_sum = xt_sum + count * 3.21e-9  # 每个占用格加入固定噪声底，W。
        # *_sum_w 为功率总和，noise_per_channel_w 为每格均值；零串扰不等于零总噪声。
        self.statistics = dict(
            signal_sum_w=signal_sum, noise_sum_w=noise_sum, xt_sum_w=xt_sum,
            floor_sum_w=count * 3.21e-9,
            noise_per_channel_w=noise_sum/count if count else None,
            occupied_channels=count, zero_noise_channels=zero_count)
        return signal_sum/noise_sum if count else None


def noise_power_to_counts(power, frequencies, detector):
    """将噪声功率换算为每探测门的期望噪声探测计数，计入探测效率与插损，不含暗计数。
    返回期望计数，不是至少一次点击的概率。"""
    # power/frequencies 为可广播的功率 W/频率 Hz 数组；detector 提供门宽 s、效率和插损 dB。
    return (np.asarray(power, dtype=float) * detector.gate_time * detector.efficiency
            * 10 ** (-0.1 * detector.insertion_loss_db)
            / (6.62607015e-34 * np.asarray(frequencies, dtype=float)))


def _neighbor_noise_components(fiber, neighbors, forward_powers, backward_powers,
                               frequencies, quantum_frequencies, z):
    """按 neighbors 中的芯编号汇总噪声，依次返回前/后向拉曼、前/后向 FWM 功率（W）。"""
    # fiber 为该组邻芯的光纤参数，双向功率按 [芯, 信道] 排列；频率 Hz、距离 z 为 m 数组。
    # components 按 [四种噪声分量, 量子频点, 距离] 排列，如 (4, 2, 1)。
    components = np.zeros((4, len(quantum_frequencies), len(z)), dtype=float)
    directions = (
        (forward_powers, fiber.get_inter_forward_raman_scatter,
         fiber.get_intercore_four_wave_mixing),
        (backward_powers, fiber.get_inter_backward_raman_scatter,
         fiber.get_backward_intercore_four_wave_mixing),
    )
    for core in neighbors:
        for direction, (powers, raman_function, fwm_function) in enumerate(directions):
            components[direction] += fiber.get_raman_power(
                frequencies, powers[core], quantum_frequencies, raman_function, z)
            components[direction + 2] += fiber.get_fwm_power(
                frequencies, powers[core], quantum_frequencies, fwm_function, z)[1]
    return components


def calculate_noise_core(i, j, c, m_resourceMap, P_link, m_dis,
                         available_channel, first_neighbor, secondary_neighbor, noise_model):
    """返回链路 i→j、纤芯 c 各量子信道的噪声功率一维数组（W），保持原信道顺序。
    仅汇总最近邻和次近邻的双向拉曼、FWM；无量子信道时返回空数组。"""
    # i/j/c 为源节点/目的节点/芯编号，如 0/1/6；资源与 P_link 按 [源, 目的, 芯, 信道] 排列。
    # P_link 为功率 W，m_dis 为距离矩阵 m，available_channel 为信道频率数组 Hz。
    # 两组 neighbor 按芯列出邻居，noise_model 提供对应光纤；ICXT 不加入实际量子噪声。
    frequencies = np.asarray(available_channel)
    quantum_frequencies = frequencies[m_resourceMap[i, j, c] == 3]
    # 状态 3 选量子频点，状态 2 选经典泵浦；屏蔽其他格内可能残留的功率。
    active_forward = np.where(m_resourceMap[i, j] == 2, P_link[i, j], 0.0)
    active_backward = np.where(m_resourceMap[j, i] == 2, P_link[j, i], 0.0)
    z = np.array([m_dis[i][j]])

    first = _neighbor_noise_components(
        noise_model.first_fiber, first_neighbor[c], active_forward, active_backward,
        frequencies, quantum_frequencies, z)
    secondary = _neighbor_noise_components(
        noise_model.secondary_fiber, secondary_neighbor[c], active_forward, active_backward,
        frequencies, quantum_frequencies, z)

    # 保持先拉曼、后 FWM 的浮点求和顺序，避免改变噪声及密钥率的数值结果。
    noise_sum = (first[0] + first[1] + secondary[0] + secondary[1]
                 + first[2] + first[3] + secondary[2] + secondary[3])
    noise_sum = np.asarray(noise_sum, dtype=float).reshape(-1)
    if noise_sum.size != len(quantum_frequencies):
        raise ValueError("Expected one noise value per quantum channel")
    return noise_sum
