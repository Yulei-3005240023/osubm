import subsicence as su
import numpy as np

class ClassicTerzaghi(su.SubsidenceModel):
    """
    这个类代表经典太沙基理论，不用考虑变渗透系数，要保证经典的原汁原味
    """
    def __init__(self, cv=None, mv=1e-8, L=10,  N = 21, T = 10, dt = 0.5, u0 = 98100, K = 1):
        super().__init__(L = L, N = N, T = T, dt = dt, u0 = u0)
        self.name_chinese = '经典太沙基一维固结'
        self.cv = cv # 固结系数 (m²/day)
        self.mv = mv  # 压缩系数 (1/Pa)
        self.K = float(K) # 渗透系数 m/d 这个float不加上会出大问题

    def __calculate_cv(self):
        self.cv = self.K / (self.rw * self.mv)

    def __calculate_sub(self, u, u_prev):
        # 计算单次时间沉降量
        s = 0
        for i in range(0, self.N):
            s += -self.mv * (u[i] - u_prev[i]) * self.dz  # 以沉降（压缩）为正
        return s

    def solve_u(self, Hydraulic_Head=False):
        # 计算孔压 沉降量
        if self.cv is not None:
            pass
        else:
            self.__calculate_cv()
        s = 0
        # 初始化数组
        alpha = self.cv * self.dt / self.dz ** 2
        settlement_history = [0]  # 沉降历史记录
        u_all = np.zeros((self.M, self.N))  # 全部孔压储存
        u_all[0, :] = self.u
        h_all = np.zeros((self.M, self.N))  # 全部水头储存
        h_list = np.zeros(self.N)
        # 转换为水头输出-初始条件
        for k in range(0, self.N):
            h = self.u_to_h(self.u[k], -1 * self.z[k])
            h_list[k] = h
        h_all[0, :] = h_list

        self.delta_pwp_redefine()

        # 时间循环
        for m in range(1, self.M):
            # 分时刻计算
            u_prev = np.copy(self.u)
            # 构建系数矩阵（三对角矩阵）
            A = np.zeros((self.N, self.N))
            # 构建常数矩阵
            b = np.zeros(self.N)

            # 内部节点方程
            for i in range(1, self.N - 1):
                A[i, i - 1] = alpha
                A[i, i] = -1 - 2 * alpha
                A[i, i + 1] = alpha
                b[i] = -u_prev[i]

            # 边界条件赋值
            A, b = self.boundary_violation(A=A, b=b, m=m)

            # 求解线性方程组
            u = np.linalg.solve(A, b)
            # 计算沉降
            ss = self.__calculate_sub(u, u_prev)
            s += ss
            settlement_history.append(ss)
            u_all[m,:] = u

            # 转换为水头输出
            h_list = np.zeros(self.N)
            for k in range(0, self.N):
                h = self.u_to_h(u[k], -self.z[k])
                h_list[k] = h
            h_all[m, :] = h_list  # 记录结果
            u_all[m, :] = u  # 记录结果
            self.u = np.copy(u)

        if Hydraulic_Head:
            return h_all, settlement_history,s
        else:
            return u_all,settlement_history,s
