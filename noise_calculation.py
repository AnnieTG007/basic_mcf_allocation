"""计算共纤噪声：量子端拉曼/四波混频（FWM），以及经典端参考 OSNR 所需的串扰与固定噪声底。

本模块由 main 构建模型，skr_calculation 调用量子噪声，main 调用经典 OSNR；
不读文件、不分配资源。
频率 Hz、距离 m、发射和噪声功率 W，距离可传数组以同时计算多个长度。
正式量子评估只汇总最近邻/次近邻经典芯的前后向拉曼与 FWM，远芯被截断为零。
FWM 仅计算离散频率 fi+fj-fk，未模拟带内展宽、频率漂移或跨芯泵浦混合。
经典 OSNR 只使用参考同频芯间串扰及 3.21e-9 W 噪声底；独立 ClassicalFWMScorer 可汇总候选频点功率，
正式分配入口不调用该独立接口。

MulticoreFiber_Multi_Att 和 wave 是正式入口不使用的独立接口。
XT（线性芯间串扰）用于经典 OSNR，不加入量子噪声汇总；hij 单位为 km^-1。
现有公式缺少完整可核查的出处记录，参数和近似沿用原实现，不能据此声称
已完成实验标定。下方注释标明已知限制，不借注释整理改变数值模型。
"""
from functools import lru_cache, partial
from itertools import permutations, combinations
from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class FiberParameters:
    """光纤物理参数；默认数值集中在 main.build_simulation。
    
    loss/loss_c/loss_q：通用/经典/量子光的指数功率衰减系数，m^-1；
    D_c：色散参数，s/m^2；D_s：色散斜率，s/m^3；A_eff：有效模场面积 m^2；
    FW：拉曼谱频率校正的参考频率 Hz；c：光速 m/s；n：折射率；
    hmn：芯间耦合系数 m^-1；recapture_factor_Rayleigh：瑞利后向俘获比例；
    loss_Rayleigh：瑞利散射系数 m^-1；width：接收滤波波长宽度 m（不是 Hz）。
    e3：原公式中的三阶非线性系数；其数值和单位制必须与 gamma 表达式配套，
    当前项目未提供原始单位制出处，不能直接替换为其他文献的 SI 数值。
    """
    loss: float
    loss_c: float
    loss_q: float
    D_c: float
    D_s: float
    A_eff: float
    FW: float
    c: float
    e3: float
    n: float
    hmn: float
    recapture_factor_Rayleigh: float
    loss_Rayleigh: float
    width: float

    def __post_init__(self):
        if not all(np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Fiber parameters must be finite")
        if any(getattr(self, name) <= 0 for name in
               ("loss", "loss_c", "loss_q", "A_eff", "FW", "c", "n", "width")):
            raise ValueError("Fiber loss, area, frequency, speed, index and bandwidth must be positive")
        if any(getattr(self, name) < 0 for name in
               ("e3", "hmn", "recapture_factor_Rayleigh", "loss_Rayleigh")):
            raise ValueError("Nonlinearity and coupling/scattering coefficients must be nonnegative")


@dataclass(frozen=True)
class MulticoreFiber:
    """使用一组固定参数计算芯内、芯间的拉曼与 FWM 功率。
    
    输入 fi/fj/fk 为三个泵浦频率 Hz，pi/pj/pk 为各自功率 W；z 为正距离
    一维数组 m。FWM 接口返回 (生成频率 Hz, 各距离处功率数组 W)，拉曼
    单项接口返回功率数组 W。forward/backward 均相对量子信号传播方向。
    后缀 all 表示累加各非零功率的经典泵浦，不代表同时计算所有物理效应。
    """

    params: FiberParameters

    def __getattr__(self, name):
        return getattr(self.params, name)

    # FWM计算相关
    def get_phase_matching_factor(self, fi, fj, fk):
        """由三个泵浦频率 Hz 计算相位失配量 beta（m^-1）。
        
        生成频率为 fi+fj-fk，先转成对应波长再计算色散造成的失配。
        函数名沿用旧实现；返回量不是后续公式中无量纲的转换效率 eta。
        """
        f = fi + fj - fk  # 四波混频频率
        w = self.c / f  # 转化为波长
        beta = 2 * np.pi * w ** 2 / self.c * np.abs(fi - fk) * np.abs(fj - fk) \
               * (self.D_c + w ** 2 / 2 / self.c * (np.abs(fi - fk) + np.abs(fj - fk)) * self.D_s)
        return beta

    def get_four_wave_mixing(self, fi, fj, fk, pi, pj, pk, beta, z: np.ndarray):
        """计算三个泵浦在同一芯产生的 FWM，返回 (生成频率 Hz, 功率数组 W)。
        
        beta 为相位失配量 m^-1，z 为正距离数组 m。D=3/6 分别用于两泵浦
        频率相同/不同，判断容差为 1 MHz。这里的 eta 将通常距离振荡项
        sin(beta*z/2)^2 固定为 1，是原实现保留的包络式近似，不是振荡平均值；
        不能将此结果当作含相位振荡的精确解。经典 OSNR 的芯内 FWM 使用此公式。
        """
        if np.abs(fi - fj) < 10 ** 6:
            D = 3  # 如果相等，D=3
        else:
            D = 6  # 如果不相等，D=6

        f_fwm = fi + fj - fk  # 四波混频频率
        w_fwm = self.c / f_fwm  # 四波混频波长
        # 保留包络式近似：将正弦平方项固定为 1，详见本函数说明。
        eta = self.loss ** 2 / (self.loss ** 2 + beta ** 2) \
              * (1 + (4 * np.exp(-self.loss * z) / (1 - np.exp(-self.loss * z)) ** 2))

        eff_distance = (1 - np.exp(-self.loss * z)) / self.loss
        gamma = (32 * np.pi ** 3 * self.e3) / (self.n ** 2 * w_fwm * self.c * self.A_eff)
        p_fwm = eta * (D * gamma) ** 2. * eff_distance ** 2. * pi * pj * pk * np.exp(-self.loss * z)

        return f_fwm, p_fwm

    def get_intercore_four_wave_mixing(self, fi, fj, fk, pi, pj, pk, beta, z):

        """返回同向经典泵浦产生并耦合到另一芯的 (FWM 频率 Hz, 功率数组 W)，z 为 m。"""
        if np.abs(fi - fj) < 10 ** 6:
            D = 3  # 如果相等，D=3
        else:
            D = 6  # 如果不相等，D=6

        f_fwm = fi + fj - fk  # 四波混频频率
        w_fwm = self.c / f_fwm  # 四波混频波长

        gamma = (32 * np.pi ** 3 * self.e3) / (self.n ** 2 * w_fwm * self.c * self.A_eff)
        I = np.exp(-self.loss * z) / (self.loss ** 2 + beta ** 2) \
            * (beta * np.sin(beta * z) - self.loss * np.cos(beta * z))
        C = (beta ** 2 - 3 * self.loss ** 2) / (2 * self.loss * (self.loss ** 2 + beta ** 2))

        p_fwm = self.hmn * np.exp(-self.loss * z) / (self.loss ** 2 + beta ** 2) \
                * 256 * np.pi ** 4 * (2 * np.pi * f_fwm) ** 2 / (self.n ** 4 * self.c ** 4) \
                * (D * self.e3) ** 2 * pi * pj * pk / self.A_eff ** 2 \
                * (-np.exp(-2 * self.loss * z) / 2 / self.loss - 2 * I + z + C)

        return f_fwm, p_fwm

    def get_backward_intercore_four_wave_mixing(self, fi, fj, fk, pi, pj, pk, beta, z):
        """返回反向经典泵浦经瑞利散射及芯间耦合产生的 (频率 Hz, 功率数组 W)。
        
        沿用原公式，两泵浦频率相同的判断容差为 1e-5 Hz，与前向接口不同。
        """
        if np.abs(fi - fj) < 1e-5:
            D = 3  # 如果相等，D=3
        else:
            D = 6  # 如果不相等，D=6
        f = fi + fj - fk
        w = self.c / f
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
        """按原组合顺序生成非零泵浦的 (fi, fj, fk, pi, pj, pk)。"""
        active = np.nonzero(powers)
        frequencies = frequencies[active]
        powers = powers[active]
        for i, j, k in self.get_fwm_permutations(len(frequencies)):
            yield (frequencies[i], frequencies[j], frequencies[k],
                   powers[i], powers[j], powers[k])

    def _evaluate_fwm(self, inputs, function, z):
        beta = self.get_phase_matching_factor(*inputs[:3])
        return function(*inputs, beta, z)

    def _sum_fwm_on_input_channels(self, ls_f, ls_p, function, z, *, include_generated):
        """共用输入信道匹配；可选择收集并排序网格外的新频点。"""
        out_f = ls_f.copy()
        out_p = np.zeros((len(out_f), len(z)), dtype=float)
        for inputs in self._iter_fwm_inputs(ls_f, ls_p):
            frequency, power = self._evaluate_fwm(inputs, function, z)
            matches = np.where(np.abs(out_f - frequency) < 1e8)[0]
            # 保留两个旧接口的历史匹配行为；严格量子接口另行校验。
            if len(matches) > 2:
                raise RuntimeError('计算出的有两个匹配的值。')
            if len(matches) == 1:
                out_p[matches, :] += [power]
            elif include_generated:
                out_p = np.vstack((out_p, power))
                out_f = np.append(out_f, frequency)
        if include_generated:
            order = np.argsort(out_f)
            return out_f[order], out_p[order]
        return out_f, out_p

    def get_fwm_power_all(self, ls_f: np.array, ls_p: np.array, function, z: np.ndarray):
        """返回输入及新生成频点的 FWM；按频率排序，功率形状为 (频点数, 距离数)。"""
        return self._sum_fwm_on_input_channels(
            ls_f, ls_p, function, z, include_generated=True)

    def get_fwm_power_all2(self, ls_f: np.array, ls_p: np.array, function, z: np.ndarray):
        """仅返回输入频点上的 FWM，保留输入顺序；匹配容差 100 MHz。"""
        return self._sum_fwm_on_input_channels(
            ls_f, ls_p, function, z, include_generated=False)

    def get_fwm_power_all3(self, ls_f_class, ls_p_class, ls_f_quantum, function, z: np.ndarray):
        """返回指定频点上的 FWM；容差 100 kHz，沿用 float32 累加精度。

        先匹配目标频点再计算功率，避免计算不会落入目标信道的组合。
        """
        out_f = ls_f_quantum.copy()
        out_p = np.zeros((len(out_f), len(z)), dtype=np.float32)
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
        """生成不重复的 (i,j,k) 索引组合，供 fi+fj-fk 计算。
        
        先枚举三者不同的组合（i/j 可交换，只保留一次），再枚举 i=j、k 不同的
        二泵浦组合；排除 k=i 或 k=j 的平凡项，保持原累加顺序。
        """
        index_list = np.arange(n)
        fwm_permutations = []  # 输出的数组
        for c1 in combinations(index_list, 3):
            # 有三个元素参与的组合
            for c2 in combinations(c1, 2):
                h = []
                for v in c2:
                    h.append(v)
                for v2 in c1:
                    if v2 not in c2:
                        h.append(v2)
                        break
                fwm_permutations.append(tuple(h))

        for c1 in permutations(index_list, 2):
            h = [c1[0], c1[0], c1[1]]
            fwm_permutations.append(tuple(h))

        return fwm_permutations

    # 拉曼散射计算相关
    def get_raman_eta(self, f_pump, f_signal, list_eta_raman, index_center, f_diff):
        """按泵浦与目标频率差查询拉曼谱系数，并做四次方频率校正。
        
        f_pump/f_signal/f_diff 为 Hz，index_center 为零频差下标。
        频差除以谱采样间隔后取最近整数（Python round），不插值；越界直接报错。
        返回系数与 z(m)、width(m) 配套使用，使 eta*z*width 为无量纲比例。
        """
        index = int(round((f_signal - f_pump) / f_diff))  # 取最近整数，恰在半整数时取偶数
        if not 0 <= index_center + index < len(list_eta_raman):
            raise ValueError("Frequency offset is outside the supplied Raman spectrum")
        eta = list_eta_raman[index_center + index] * (f_signal / (self.FW + index * f_diff)) ** 4
        return eta

    def get_forward_raman_scatter(self, p: np.array, z: np.array, eta):
        """
        计算前向拉曼散射功率
        :param p: 输入功率
        :param z: 传输距离
        :param eta: 拉曼散射系数
        :return: 输出功率
        """
        pout = eta * p * np.exp(-self.loss * z) * z * self.width
        return pout

    def get_forward_raman_scatter2(self, p: np.array, z: np.array, eta):
        """同芯前向拉曼功率 W，分别使用经典/量子衰减；两衰减相等时此旧公式有零分母。"""
        return p * np.exp(-self.loss_q * z) * (1 - np.exp(- (self.loss_c - self.loss_q) * z)) / (
                self.loss_c - self.loss_q) * eta * self.width

    def get_backward_raman_scatter(self, p: np.array, z: np.array, eta):
        """
        计算后向拉曼散射功率
        :param p: 输入功率
        :param z: 传输距离
        :param eta: 拉曼散射系数
        :return: 输出功率
        """
        pout = eta * p * (1 - np.exp(-2 * self.loss * z)) / (2 * self.loss) * self.width
        return pout

    def get_backward_raman_scatter2(self, p: np.array, z: np.array, eta):
        """同芯后向拉曼功率 W，分别使用经典/量子衰减，z 为 m。"""
        return p * (1 - np.exp(- (self.loss_c + self.loss_q) * z)) / (self.loss_c + self.loss_q) * eta * self.width

    def get_inter_forward_raman_scatter(self, p, z, eta):
        """芯间前向拉曼功率 W；eta 为谱系数，p 为经典功率 W，z 为距离 m。
        
        原表达式要求 loss_q-loss_c 及 loss_q-loss_c-2*hmn 非零；默认参数满足。
        """
        return eta * p * np.exp(-self.loss_q * z) * (
                (np.exp((self.loss_q - self.loss_c) * z) - 1) / (self.loss_q - self.loss_c)
                - (np.exp((self.loss_q - self.loss_c - 2 * self.hmn) * z) - 1) / (
                        self.loss_q - self.loss_c - 2 * self.hmn)) * self.width

    def get_inter_backward_raman_scatter(self, p, z, eta):
        """芯间后向拉曼功率 W，使用固定 loss_c/loss_q/hmn，p 为 W、z 为 m。"""
        return eta * p * ((np.exp(-(self.loss_q + self.loss_c + 2 * self.hmn) * z) - 1) / (
                self.loss_q + self.loss_c + 2 * self.hmn)
                          - (np.exp(-(self.loss_q + self.loss_c) * z) - 1) / (self.loss_q + self.loss_c)) * self.width

    def _raman_eta_lookup(self, list_eta_raman, index_center, f_diff):
        """将光谱参数绑定到系数查询，汇总层只需传入泵浦和目标频率。"""
        return partial(self.get_raman_eta, list_eta_raman=list_eta_raman,
                       index_center=index_center, f_diff=f_diff)

    def _sum_raman_power(self, ls_f, ls_p, target_f, function, z, eta_for_pair,
                         *, attenuations=None):
        """共用拉曼汇总；保留逐泵浦累加顺序及各泵浦独立功率。

        attenuations 为可选的 (泵浦衰减数组, 目标衰减数组)；
        是否屏蔽同频系数由 eta_for_pair 决定。
        """
        z = np.array(z)
        active = np.nonzero(ls_p)[0]
        out_p = np.zeros((len(target_f), len(z)), dtype=float)
        for target_index, signal in enumerate(target_f):
            for pump_index in active:
                eta = eta_for_pair(ls_f[pump_index], signal)
                extra_args = () if attenuations is None else (
                    attenuations[0][pump_index], attenuations[1][target_index])
                out_p[target_index] += function(ls_p[pump_index], z, eta, *extra_args)
        return out_p

    def get_raman_power_all(self, ls_f: np.array, ls_p: np.array, function, z: np.ndarray,
                            list_eta_raman, index_center, f_diff):
        """计算输入频点上的拉曼噪声，返回 (频点数, 距离数) 功率数组。"""
        return self._sum_raman_power(
            ls_f, ls_p, ls_f, function, z,
            self._raman_eta_lookup(list_eta_raman, index_center, f_diff))

    def get_raman_power_all2(self, ls_f: np.array, ls_p: np.array, ls_quantum: np.ndarray,
                             function, z: np.ndarray, list_eta_raman, index_center, f_diff):
        """计算指定目标频点上的拉曼噪声，保留目标频点顺序。"""
        return self._sum_raman_power(
            ls_f, ls_p, ls_quantum, function, z,
            self._raman_eta_lookup(list_eta_raman, index_center, f_diff))

    def get_raman_power_all_O_quantum_C_classical(self, ls_f, ls_p, ls_quantum, function, z, eta_raman_O):
        """O 波段量子信道、C 波段经典泵浦，使用传入的常量拉曼系数。"""
        return self._sum_raman_power(
            ls_f, ls_p, ls_quantum, function, z,
            lambda pump, signal: eta_raman_O)

    def __str__(self):
        return 'Multicore, hmn={}/m'.format(self.hmn)


class MulticoreFiber_Multi_Att(MulticoreFiber):
    """保留的逐信道衰减计算接口，正式 main 仿真未调用。
    
    att/att_c/att_q 为 m^-1。注意后向芯间拉曼函数虽然接收 att_c/att_q，
    实际仍使用实例固定 loss_c/loss_q；这条旧接口不具备完整的逐信道衰减能力。
    芯内 FWM 则保留正弦振荡项，与基类的包络式近似不同。
    """
    def __init__(self, params):
        super().__init__(params)

    def get_four_wave_mixing(self, fi, fj, fk, pi, pj, pk, att, beta, z):
        """以传入衰减 att（m^-1）计算含相位振荡的芯内 FWM；返回频率 Hz 和功率数组 W。"""
        if np.abs(fi - fj) < 10 ** 6:
            D = 3  # 如果相等，D=3
        else:
            D = 6  # 如果不相等，D=6

        f_fwm = fi + fj - fk  # 四波混频频率
        w_fwm = self.c / f_fwm  # 四波混频波长
        # 四波混频转化效率
        eta = att ** 2 / (att ** 2 + beta ** 2) \
              * (1 + (4 * np.exp(-att * z) * np.sin(beta * z / 2) ** 2 / (1 - np.exp(-att * z)) ** 2))

        eff_distance = (1 - np.exp(-att * z)) / att
        gamma = (32 * np.pi ** 3 * self.e3) / (self.n ** 2 * w_fwm * self.c * self.A_eff)
        p_fwm = eta * (D * gamma) ** 2. * eff_distance ** 2. * pi * pj * pk * np.exp(-att * z)

        return f_fwm, p_fwm

    # 需要输入衰减的拉曼散射
    def get_forward_raman_scatter(self, p, z, eta, att_c, att_q):
        """逐信道衰减的芯内前向拉曼功率 W；两衰减相等时使用其解析极限。"""
        if att_c == att_q:
            return p * np.exp(-att_c * z) * z * self.width * eta
        return p * np.exp(-att_q * z) * ((1 - np.exp(-(att_c - att_q) * z)) / (att_c - att_q)) * self.width * eta

    def get_backward_raman_scatter(self, p: np.array, z: np.array, eta, att_c, att_q):
        """逐信道衰减的芯内后向拉曼功率 W；两衰减相等时使用同衰减公式。"""
        if att_c == att_q:
            return p * (1 - np.exp(-2 * att_c * z)) / (2 * att_c) * self.width * eta
        return p * (1 - np.exp(-(att_c + att_q) * z)) / (att_c + att_q) * self.width * eta

    def get_inter_forward_raman_scatter(self, p, z, eta, att_c, att_q):
        """芯间前向拉曼功率 W；衰减相等时沿用微小扰动，未使用解析极限。"""
        if att_c == att_q:
            att_q = att_c * (1 + 1e-5)
        return p * np.exp(-att_q * z) * (
                (np.exp((att_q - att_c) * z) - 1) / (att_q - att_c)
                - (np.exp((att_q - att_c - 2 * self.hmn) * z) - 1) / (
                        att_q - att_c - 2 * self.hmn)) * self.width * eta

    def get_inter_backward_raman_scatter(self, p, z, eta, att_c, att_q):
        """旧接口限制：最终表达式使用实例 loss_c/loss_q，传入 att_c/att_q 不影响返回值。"""
        if att_c == att_q:
            att_q = att_c * (1 + 1e-5)
        return eta * p * ((np.exp(-(self.loss_q + self.loss_c + 2 * self.hmn) * z) - 1) / (
                self.loss_q + self.loss_c + 2 * self.hmn)
                          - (np.exp(-(self.loss_q + self.loss_c) * z) - 1) / (self.loss_q + self.loss_c)) * self.width

    def get_raman_power_all(self, ls_f: np.array, ls_p: np.array, ls_att: np.ndarray,
                            function, z: list, out_ls_f, out_ls_att,
                            list_eta_raman, index_center, f_diff):
        """传递各泵浦/目标的衰减给 function，返回 [目标频点, 距离] 功率数组 W。
        
        同频差小于 1 kHz 时拉曼系数置零。具体函数是否采用传入衰减以该函数为准；
        本类后向芯间接口仍使用实例固定衰减，不能据此方法名推断其已支持逐信道衰减。
        """
        lookup = self._raman_eta_lookup(list_eta_raman, index_center, f_diff)

        def eta_for_pair(pump, signal):
            eta = lookup(pump, signal)
            return 0 if np.abs(pump - signal) < 1e3 else eta

        return self._sum_raman_power(
            ls_f, ls_p, out_ls_f, function, z, eta_for_pair,
            attenuations=(ls_att, out_ls_att))


@dataclass(frozen=True)
class XTParameters:
    """独立线性串扰接口参数，正式量子噪声汇总不使用它。
    
    loss_per_km/rayleigh_per_km 为 km^-1；gate_time 为 s；efficiency 为比例；
    insertion_loss_db 为 dB；wavelength_m 为 m；planck_constant 为 J*s，
    light_speed 为 m/s。串扰计数接口额外保留原公式的 1/2 因子。
    """
    loss_per_km: float
    rayleigh_per_km: float
    gate_time: float
    efficiency: float
    insertion_loss_db: float
    wavelength_m: float
    planck_constant: float
    light_speed: float


def forward_P_XT(hij, L, P0, params):
    """前向串扰功率 W；hij 为 km^-1，L 为 m。"""
    length_km = L / 1000
    return (P0 * math.exp(-hij * length_km) * math.sinh(hij * length_km)
            * math.exp(-params.loss_per_km * length_km))


def backward_P_XT(hij, L, P0, params):
    """反向串扰功率 W；沿用原公式与单位。"""
    length_km = L / 1000
    loss = params.loss_per_km
    return params.rayleigh_per_km * P0 * hij * (
        (1 - math.exp(-2 * loss * length_km)) / loss
        - 2 * length_km * math.exp(-2 * loss * length_km))


def _xt_counts(power, params):
    """将线性串扰功率 W 转为每门探测计数，保留历史 1/2 因子；不含暗计数。"""
    return (power / 2 * params.gate_time * params.efficiency
            / params.planck_constant / params.light_speed * params.wavelength_m
            * 10 ** (-0.1 * params.insertion_loss_db))


def forward_SDM_XT(hij, L, P0, params):
    """前向线性芯间串扰的每门计数；hij 为 km^-1、L 为 m、P0 为 W。"""
    return _xt_counts(forward_P_XT(hij, L, P0, params), params)


def backward_SDM_XT(hij, L, P0, params):
    """反向线性芯间串扰的每门计数；hij 为 km^-1、L 为 m、P0 为 W。"""
    return _xt_counts(backward_P_XT(hij, L, P0, params), params)


def wave(num_of_channels, interval, *, center_frequency, min_frequency, max_frequency):
    """以传入中心频率生成频率网格；超出传入边界则报错。"""
    if type(num_of_channels) is not int or num_of_channels < 1:
        raise ValueError("Channel count must be a positive integer")
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Channel interval must be positive")
    frequencies = center_frequency + (np.arange(num_of_channels) - num_of_channels // 2) * interval
    if frequencies[0] < min_frequency or frequencies[-1] > max_frequency:
        raise ValueError("Channels exceed supplied frequency range")
    return frequencies


@dataclass(frozen=True)
class RamanSpectrum:
    """已完成单位换算的拉曼系数表，不负责文件读取。"""
    coefficients: tuple
    index_center: int
    frequency_step_hz: float

    def __post_init__(self):
        values = tuple(float(v) for v in self.coefficients)
        if not values or not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError("Raman coefficients must be finite and nonnegative")
        if type(self.index_center) is not int or not 0 <= self.index_center < len(values):
            raise ValueError("Invalid Raman center index")
        if not np.isfinite(self.frequency_step_hz) or self.frequency_step_hz <= 0:
            raise ValueError("Raman frequency step must be positive")
        object.__setattr__(self, "coefficients", values)


@dataclass(frozen=True)
class NoiseModel:
    """组合最近邻、次近邻两组光纤参数及同一张拉曼谱；远芯不参与正式量子噪声汇总。"""
    first_fiber: MulticoreFiber
    secondary_fiber: MulticoreFiber
    raman: RamanSpectrum


class ClassicalFWMScorer:
    """求已占用泵浦在全部固定经典候选频点产生的芯内 FWM 总功率。

    复用项目多芯光纤的三频组合及功率模型；模型/频率改变时应新建实例。
    缓存只依赖频率（实例固定）、功率向量与距离，不依赖可变资源矩阵。
    """
    def __init__(self, frequencies, classical_indices, fiber):
        self.indices = np.asarray(tuple(classical_indices), dtype=int)
        self.frequencies = np.asarray(frequencies, dtype=float)[self.indices].copy()
        self.fiber = fiber
        self._cached = lru_cache(maxsize=8192)(self._calculate)

    def _calculate(self, powers, distance):
        if sum(p > 0 for p in powers) < 2 or distance == 0:
            return 0.0
        _, noise = self.fiber.get_fwm_power_all3(
            self.frequencies, np.asarray(powers, dtype=float), self.frequencies,
            self.fiber.get_four_wave_mixing, np.asarray([distance], dtype=float),
        )
        score = float(np.sum(noise, dtype=np.float64))
        if not np.isfinite(score) or score < 0:
            raise ValueError("Invalid classical FWM score")
        return score

    def __call__(self, powers, distance):
        """输入包含量子及经典索引的全信道功率向量 W 和距离 m，返回所有经典候选上的 FWM 总功率 W。"""
        selected = np.asarray(powers, dtype=float)[self.indices]
        if not np.all(np.isfinite(selected)) or np.any(selected < 0):
            raise ValueError("Classical powers must be finite and nonnegative")
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Link distance must be finite and nonnegative")
        return self._cached(tuple(selected), float(distance))



# 经典串扰参数沿用参考 XT.py；功率接口仅使用 loss_per_km 与 rayleigh_per_km。
# 最近邻/次近邻耦合为 1e-6/1e-7 km^-1，独立于量子拉曼/FWM 的 hmn（m^-1）。
# 功率接口仅使用前两个参数；其余值保留参考 XT.py 的探测参数约定。
XT_PARAMS = XTParameters(0.2 / 4.343, 1e-3, 0.5e-9, 0.1, 8.0,
                         1550e-9, 6.62e-34, 3e8)

class ClassicalOSNRScorer:
    """只读计算所选链路的参考经典 OSNR，不参与路由或分配。

    参考业务入口先对占用信道的串扰求均值，再加固定噪声底 3.21e-9 W。
    单链路用实际长度、实际发射功率和一跳；两个经典方向的占用格共同平均。
    保留参考 calculate_noise_core 的判断：反向串扰仅在受扰芯的反向同频
    功率非零时加入，即使邻芯反向有光也不绕过此判断。FWM/拉曼不计入经典
    OSNR；量子 SKR 的拉曼/FWM 仍由量子评分器计算。空闲返回 None。
    """
    def __init__(self):
        self.statistics = {}

    def __call__(self, resources, powers, distances, first, secondary, link):
        """数组维度 [源, 目的, 芯, 信道]，距离 m，功率 W；返回线性比值。"""
        signal_sum = xt_sum = 0.0
        count = zero_count = 0
        a, b = link
        length = float(distances[a, b])
        for i, j in ((a, b), (b, a)):
            for c, w in np.argwhere(resources[i, j] == 2):
                noise = 0.0
                for neighbors, coupling in ((first[c], 1e-6), (secondary[c], 1e-7)):
                    for neighbor in neighbors:
                        if powers[i, j, c, w] != 0:
                            noise += forward_P_XT(coupling, length, float(powers[i, j, neighbor, w]), XT_PARAMS)
                        if powers[j, i, c, w] != 0:
                            noise += backward_P_XT(coupling, length, float(powers[j, i, neighbor, w]), XT_PARAMS)
                signal_sum += float(powers[i, j, c, w]) * 10 ** (-length * 0.2 * 1e-4)
                xt_sum += noise
                count += 1
                zero_count += int(noise == 0)
        noise_sum = xt_sum + count * 3.21e-9
        self.statistics = dict(
            signal_sum_w=signal_sum, noise_sum_w=noise_sum, xt_sum_w=xt_sum,
            floor_sum_w=count * 3.21e-9,
            noise_per_channel_w=noise_sum/count if count else None,
            occupied_channels=count, zero_noise_channels=zero_count)
        return signal_sum/noise_sum if count else None


def noise_power_to_counts(power, frequencies, detector):
    """将噪声功率(W)换算为每探测门噪声光子数（含探测效率，不含暗计数）。

    power 与 frequencies 使用可广播的同形数组；探测器参数由调用方提供。
    XT 的历史 SDM 接口保留其自身的门宽和常数。
    """
    return (np.asarray(power, dtype=float) * detector.gate_time * detector.efficiency
            * 10 ** (-0.1 * detector.insertion_loss_db)
            / (6.62607015e-34 * np.asarray(frequencies, dtype=float)))


def _neighbor_noise_components(fiber, neighbors, forward_powers, backward_powers,
                               frequencies, quantum_frequencies, z, raman):
    """汇总一组邻芯，依次返回前/后向拉曼、前/后向 FWM 功率。"""
    components = np.zeros((4, len(quantum_frequencies), len(z)), dtype=float)
    coefficients = np.asarray(raman.coefficients)
    directions = (
        (forward_powers, fiber.get_inter_forward_raman_scatter,
         fiber.get_intercore_four_wave_mixing),
        (backward_powers, fiber.get_inter_backward_raman_scatter,
         fiber.get_backward_intercore_four_wave_mixing),
    )
    for core in neighbors:
        for direction, (powers, raman_function, fwm_function) in enumerate(directions):
            components[direction] += fiber.get_raman_power_all2(
                frequencies, powers[core], quantum_frequencies, raman_function, z,
                coefficients, raman.index_center, raman.frequency_step_hz)
            components[direction + 2] += fiber.get_fwm_power_all3(
                frequencies, powers[core], quantum_frequencies, fwm_function, z)[1]
    return components


def calculate_noise_core(i, j, c, m_resourceMap, P_link, m_dis,
                         available_channel, first_neighbor, secondary_neighbor, noise_model):
    """返回指定链路/纤芯各量子信道的总噪声功率，一维数组，单位 W。

    按资源状态选择量子信道（3）与已占用的经典泵浦（2），
    仅统计邻芯/次邻芯双向拉曼和 FWM。
    """
    frequencies = np.asarray(available_channel)
    quantum_frequencies = frequencies[m_resourceMap[i, j, c] == 3]
    active_forward = np.where(m_resourceMap[i, j] == 2, P_link[i, j], 0.0)
    active_backward = np.where(m_resourceMap[j, i] == 2, P_link[j, i], 0.0)
    z = np.array([m_dis[i][j]])

    first = _neighbor_noise_components(
        noise_model.first_fiber, first_neighbor[c], active_forward, active_backward,
        frequencies, quantum_frequencies, z, noise_model.raman)
    secondary = _neighbor_noise_components(
        noise_model.secondary_fiber, secondary_neighbor[c], active_forward, active_backward,
        frequencies, quantum_frequencies, z, noise_model.raman)

    # 保留原先先拉曼、后 FWM 的浮点求和顺序，避免影响量子噪声/SKR 的浮点结果。
    noise_sum = (first[0] + first[1] + secondary[0] + secondary[1]
                 + first[2] + first[3] + secondary[2] + secondary[3])
    noise_sum = np.asarray(noise_sum, dtype=float).reshape(-1)
    if noise_sum.size != len(quantum_frequencies):
        raise ValueError("Expected one noise value per quantum channel")
    return noise_sum
