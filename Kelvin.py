import numpy as np
import subsicence as su

class Kelvin(su.SubsidenceModel):
    def __init__(self, L=10,  N = 11, T = 3, dt = 0.1, K = 0.1, w = 0):
        super().__init__(L = L, N = N, T = T, dt = dt)
        self.name_chinese = '凯文模型'
        # 参数设置
        self.E = 1e7 # 胡克弹簧弹性模量
        self.phi = 1e9  # 牛顿黏壶参数
        self.e0 = 1.5  # 初始孔隙比
        self.K = float(K)  # 初始渗透系数 m/d 这个float不加上会出大问题
        self.K_list = np.full(self.N, self.K)
        self.w = w  # 源汇项，默认为0
        self.e = np.full(self.N, self.e0)  # 孔隙比数组
        self.epsilon = np.zeros((self.M, self.N))  # 分时刻每一个位置的应变储存
        self.epsilon_max = np.zeros(self.N)  # 每一个位置的最大应变储存

    def solve(self, Hydraulic_Head=False):
        # 初始化数组
        sigma_v0 = self.sigma_total - self.u  # 加荷前初始有效应力=自重应力-初始孔隙水压力, 后续计算中代表前一时刻有效应力
        sigma_pred = self.sigma_total - self.u
        cum_settlement = 0.0  # 累计沉降量
        settlement_history = [0]  # 沉降历史记录
        u_all = np.zeros((self.M, self.N))  # 全部孔压储存
        u_all[0, :] = self.u
        h_all = np.zeros((self.M, self.N))  # 全部水头储存
        h_list = np.zeros(self.N)
        sigma_all = np.zeros((self.M, self.N))  # 全部有效应力储存

        # 转换为水头输出-初始条件
        for k in range(0, self.N):
            h = self.u_to_h(self.u[k], -1 * self.z[k])
            h_list[k] = h
        h_all[0, :] = h_list
        sigma_all[0, :] = sigma_v0

        self.delta_pwp_redefine() # 计算外部荷载


        # 隐式有限差分法迭代求解
        for m in range(1, self.M):
            iterations = 0
            # 分时刻计算
            u_prev = np.copy(self.u)  # 用于记录上一次迭代的结果
            # 构建系数矩阵
            A = np.zeros((self.N, self.N))
            # 构建常数矩阵
            b = np.zeros(self.N)
            # 保存迭代/时间步开始时初始状态
            e_initial = np.copy(self.e)
            epsilon_initial = np.copy(self.epsilon[m-1])
            epsilon_initial_new = np.copy(epsilon_initial)
            delta_epsilon_initial = np.zeros(self.N)
            K_list_initial = np.copy(self.K_list)  # 初始渗透系数
            sigma_initial = np.copy(sigma_v0)
            while True:
                # 开始矩阵赋值,迭代计算
                for i in range(1, self.N - 1):
                    ki0 = 0.5 * (K_list_initial[i + 1] + K_list_initial[i])  # 代表k+
                    ki1 = 0.5 * (K_list_initial[i] + K_list_initial[i - 1])  # 代表k-
                    # 系数矩阵赋值
                    A[i, i - 1] = ki1 / self.dz ** 2
                    A[i, i] = -(ki0 / self.dz ** 2 + ki1 / self.dz ** 2)
                    A[i, i + 1] = ki0 / self.dz ** 2
                    # 常数矩阵赋值
                    b[i] = self.rw * (self.E / self.phi * delta_epsilon_initial[i] - 1 / self.phi * (sigma_initial[i]-sigma_v0[i]) + ki0 / self.dz - ki1 / self.dz - self.w)  # 此处ki0与ki1需要根据Z轴方向调整，此处默认Z轴方向向下
                # 边界条件赋值
                A, b = self.boundary_violation(A=A, b=b, m=m)

                # 求解线性方程组
                u_new = np.linalg.solve(A, b)
                # 计算新的有效应力
                sigma_v_new = self.sigma_total - u_new

                # 沉降计算,更新渗透系数
                delta_h_total = 0  # 单一时刻

                for k in range(0, self.N):
                    # 压缩过程
                    aa= self.E / self.phi * delta_epsilon_initial[k]
                    bb= 1 / self.phi * (sigma_initial[k]-sigma_v0[k])
                    delta_epsilon_initial[k] =  (-aa + bb)*self.dt
                    epsilon_initial_new[k] = self.epsilon[m-1][k]+delta_epsilon_initial[k]
                    delta_e = delta_epsilon_initial[k] * (1 + self.e0)
                    e_initial[k] -= delta_e
                    delta_h = delta_epsilon_initial[k] * self.dz
                    delta_h_total += delta_h
                    # 根据孔隙比e的变化来更新k
                    K_list_initial[k] = self.K * pow(e_initial[k] / self.e0, 3) * pow((1 + self.e0) / (1 + e_initial[k]), 1)

                # 检查收敛性
                if np.allclose(u_new, u_prev, atol=1e-4) and np.allclose(epsilon_initial, epsilon_initial_new, atol=1e-4):
                    #print(f"时刻{m}迭代{iterations}次.")
                    break
                elif iterations > 1000:
                    print(f"时刻{m}迭代超过1000次, 可能不收敛.")
                    break
                else:
                    u_prev = np.copy(u_new)  # 重置迭代水头
                    iterations += 1  # 迭代次数加一
                    e_initial = np.copy(self.e)  # 重置e
                    sigma_initial = np.copy(sigma_v_new)
                    epsilon_initial = np.copy(epsilon_initial_new)

            # 更新状态变量
            self.e = e_initial
            self.K_list = np.copy(K_list_initial)
            self.u = np.copy(u_new)  # 更新u，孔压
            self.epsilon[m, :] = epsilon_initial_new
            for k_ in range(0, self.N):  # 更新最大应变
                if epsilon_initial[k_] > self.epsilon_max[k_]:
                    self.epsilon_max[k_] = epsilon_initial[k_]
            cum_settlement += delta_h_total
            settlement_history.append(delta_h_total)
            sigma_v0 = np.copy(sigma_v_new)  # 前一时刻有效应力


            # 转换为水头输出
            h_list = np.zeros(self.N)
            for k in range(0, self.N):
                h = self.u_to_h(self.u[k], -self.z[k])
                h_list[k] = h
            h_all[m, :] = h_list  # 记录结果
            u_all[m, :] = self.u  # 记录结果
            sigma_all[m, :] = sigma_v0
        if Hydraulic_Head:
            return h_all, settlement_history, cum_settlement
        else:
            return u_all, settlement_history, cum_settlement


class Kelvin2(su.SubsidenceModel):
    def __init__(self, L=10, N=11, T=3, dt=0.1, K=0.1, w=0):
        super().__init__(L=L, N=N, T=T, dt=dt)
        self.name_chinese = '凯文模型'
        # 参数设置
        self.E = 1e7  # 胡克弹簧弹性模量
        self.phi = 1e9  # 牛顿黏壶参数
        self.e0 = 1.5  # 初始孔隙比
        self.K = float(K)  # 初始渗透系数 m/d 这个float不加上会出大问题
        self.K_list = np.full(self.N, self.K)
        self.w = w  # 源汇项，默认为0
        self.e = np.full(self.N, self.e0)  # 孔隙比数组
        self.epsilon = np.zeros((self.M, self.N))  # 分时刻每一个位置的应变储存
        self.epsilon_max = np.zeros(self.N)  # 每一个位置的最大应变储存

    def solve(self, Hydraulic_Head=False):
        # 初始化数组
        sigma_v0 = self.sigma_total - self.u  # 加荷前初始有效应力=自重应力-初始孔隙水压力, 后续计算中代表前一时刻有效应力
        sigma_pred = self.sigma_total - self.u
        cum_settlement = 0.0  # 累计沉降量
        settlement_history = [0]  # 沉降历史记录
        u_all = np.zeros((self.M, self.N))  # 全部孔压储存
        u_all[0, :] = self.u
        h_all = np.zeros((self.M, self.N))  # 全部水头储存
        h_list = np.zeros(self.N)
        sigma_all = np.zeros((self.M, self.N))  # 全部有效应力储存

        # 转换为水头输出-初始条件
        for k in range(0, self.N):
            h = self.u_to_h(self.u[k], -1 * self.z[k])
            h_list[k] = h
        h_all[0, :] = h_list
        sigma_all[0, :] = sigma_v0

        self.delta_pwp_redefine()  # 计算外部荷载

        # 关于应力期的条件设置
        self.delta_pwp_redefine_stress_period()
        m_press_period = 0
        sigma_v0_press_period = np.copy(sigma_v0)
        u_press_period = np.copy(self.u)

        # 隐式有限差分法迭代求解
        for m in range(1, self.M):
            # 换算成当前所在的应力期
            m_press_period1 = int(m / self.M_press_period)
            if m_press_period1 != m_press_period:
                # 不相等说明进入了下一个应力期，更新应力状态
                m_press_period = m_press_period1
                sigma_v0_press_period = np.copy(sigma_v0)
                u_press_period = np.copy(self.u)
            iterations = 0
            # 分时刻计算
            u_prev = np.copy(self.u)  # 用于记录上一次迭代的结果
            # 构建系数矩阵
            A = np.zeros((self.N, self.N))
            # 构建常数矩阵
            b = np.zeros(self.N)
            # 保存迭代/时间步开始时初始状态
            e_initial = np.copy(self.e)
            epsilon_initial = np.copy(self.epsilon[m - 1])
            epsilon_initial_new = np.copy(epsilon_initial)
            delta_epsilon_initial = np.zeros(self.N)
            K_list_initial = np.copy(self.K_list)  # 初始渗透系数
            sigma_initial = np.copy(sigma_v0)
            while True:
                # 开始矩阵赋值,迭代计算
                for i in range(1, self.N - 1):
                    ki0 = 0.5 * (K_list_initial[i + 1] + K_list_initial[i])  # 代表k+
                    ki1 = 0.5 * (K_list_initial[i] + K_list_initial[i - 1])  # 代表k-
                    # 系数矩阵赋值
                    A[i, i - 1] = ki1 / self.dz ** 2
                    A[i, i] = -(ki0 / self.dz ** 2 + ki1 / self.dz ** 2)
                    A[i, i + 1] = ki0 / self.dz ** 2
                    # 常数矩阵赋值
                    b[i] = self.rw * (self.E / self.phi * delta_epsilon_initial[i] - 1 / self.phi * (
                                u_press_period[i] - u_prev[
                            i]) + ki0 / self.dz - ki1 / self.dz - self.w)  # 此处ki0与ki1需要根据Z轴方向调整，此处默认Z轴方向向下
                # 边界条件赋值
                A, b = self.boundary_violation_press_period(A=A, b=b, m=m_press_period)

                # 求解线性方程组
                u_new = np.linalg.solve(A, b)
                # 计算新的有效应力
                sigma_v_new = self.sigma_total - u_new

                # 沉降计算,更新渗透系数
                delta_h_total = 0  # 单一时刻

                for k in range(0, self.N):
                    # 压缩过程
                    aa = self.E / self.phi * delta_epsilon_initial[k]
                    bb = 1 / self.phi * (sigma_initial[k] - sigma_v0_press_period[k])
                    delta_epsilon_initial[k] = (-aa + bb) * self.dt
                    epsilon_initial_new[k] = self.epsilon[m - 1][k] + delta_epsilon_initial[k]
                    delta_e = delta_epsilon_initial[k] * (1 + self.e0)
                    e_initial[k] -= delta_e
                    delta_h = delta_epsilon_initial[k] * self.dz
                    delta_h_total += delta_h
                    # 根据孔隙比e的变化来更新k
                    K_list_initial[k] = self.K * pow(e_initial[k] / self.e0, 3) * pow(
                        (1 + self.e0) / (1 + e_initial[k]), 1)

                # 检查收敛性
                if np.allclose(u_new, u_prev, atol=1e-4) and np.allclose(epsilon_initial, epsilon_initial_new,
                                                                         atol=1e-4):
                    #print(f"时刻{m}迭代{iterations}次.")
                    break
                elif iterations > 1000:
                    print(f"时刻{m}迭代超过1000次, 可能不收敛.")
                    break
                else:
                    u_prev = np.copy(u_new)  # 重置迭代水头
                    iterations += 1  # 迭代次数加一
                    e_initial = np.copy(self.e)  # 重置e
                    sigma_initial = np.copy(sigma_v_new)
                    epsilon_initial = np.copy(epsilon_initial_new)

            # 更新状态变量
            self.e = e_initial
            self.K_list = np.copy(K_list_initial)
            self.u = np.copy(u_new)  # 更新u，孔压
            self.epsilon[m, :] = epsilon_initial_new
            for k_ in range(0, self.N):  # 更新最大应变
                if epsilon_initial[k_] > self.epsilon_max[k_]:
                    self.epsilon_max[k_] = epsilon_initial[k_]
            cum_settlement += delta_h_total
            settlement_history.append(delta_h_total)
            sigma_v0 = np.copy(sigma_v_new)  # 前一时刻有效应力

            # 转换为水头输出
            h_list = np.zeros(self.N)
            for k in range(0, self.N):
                h = self.u_to_h(self.u[k], -self.z[k])
                h_list[k] = h
            h_all[m, :] = h_list  # 记录结果
            u_all[m, :] = self.u  # 记录结果
            sigma_all[m, :] = sigma_v0
        if Hydraulic_Head:
            return h_all, settlement_history, cum_settlement
        else:
            return u_all, settlement_history, cum_settlement
