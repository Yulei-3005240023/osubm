import numpy as np
import subsicence as su

class Maxwell(su.SubsidenceModel):
    def __init__(self, L=10,  N = 11, T = 3, dt = 0.1, K = 0.01, w = 0):
        """
        有一个非常重要的假定条件就是：初始的应变设定为0，相对应的初始的有效应力应该为0
        但是实际上由于土体自重应力与孔隙水压力的影响，初始有效应力不为0
        在数值计算过程中的凡是涉及到使用有效应力计算应变或水流的应该使用总应力-孔隙水压力-初始有效应力
        """
        super().__init__(L = L, N = N, T = T, dt = dt)
        self.name_chinese = '马克斯威尔模型'
        # 参数设置
        self.E = 1e6 # 胡克弹簧弹性模量
        self.phi = 1e8  # 牛顿黏壶参数
        self.e0 = 1.5  # 初始孔隙比
        self.K = float(K)  # 初始渗透系数 m/d 这个float不加上会出大问题
        self.K_list = np.full(self.N, self.K)
        self.w = w  # 源汇项，默认为0
        self.e = None
        self.epsilon0 = 0  # 暂时设定初始应变为0
        self.epsilon = np.zeros((self.M, self.N))  # 分时刻每一个位置的应变储存

    def solve(self, Hydraulic_Head=False):
        # 初始化数组
        sigma_v0 = self.sigma_total-self.u  # 加荷前初始有效应力=自重应力-初始孔隙水压力, 后续计算中代表前一时刻有效应力
        sigma_pred = self.sigma_total - self.u
        self.e = np.full(self.N, self.e0)  # 孔隙比数组
        cum_settlement = 0.0  # 累计沉降量
        settlement_history = [0]  # 沉降历史记录
        u_all = np.zeros((self.M, self.N))  # 全部孔压储存
        u_all[0, :] = self.u
        h_all = np.zeros((self.M, self.N))  # 全部水头储存
        h_list = np.zeros(self.N)

        # 转换为水头输出-初始条件
        for k in range(0, self.N):
            h = self.u_to_h(self.u[k], -self.z[k])
            h_list[k] = h
        h_all[0, :] = h_list
        self.delta_pwp_redefine()  # 计算外部荷载

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
            K_list_initial = np.copy(self.K_list)
            epsilon_initial = np.copy(self.epsilon[m])

            while True:
                # 开始矩阵赋值,迭代计算
                for i in range(1, self.N - 1):
                    ki0 = 0.5 * (K_list_initial[i + 1] + K_list_initial[i])  # 代表k+
                    ki1 = 0.5 * (K_list_initial[i] + K_list_initial[i - 1])  # 代表k-
                    # 系数矩阵赋值
                    A[i, i - 1] = ki1 / self.dz ** 2
                    A[i, i] = -(ki0 / self.dz ** 2 + ki1 / self.dz ** 2 + self.rw / (self.E * self.dt))
                    A[i, i + 1] = ki0 / self.dz ** 2
                    # 常数矩阵赋值
                    b[i] = self.rw * (-1 / self.E * self.u[i] / self.dt - (self.sigma_total[i]- u_prev[i]-sigma_pred[i])/ self.phi + ki0 / self.dz - ki1 / self.dz - self.w)  # 此处ki0与ki1需要根据Z轴方向调整，此处默认Z轴方向向下

                # 边界条件赋值
                A, b = self.boundary_violation(A=A, b=b, m=m)

                # 求解线性方程组
                u_new = np.linalg.solve(A, b)
                # 计算新的有效应力
                sigma_v_new = self.sigma_total - u_new

                # 沉降计算,更新渗透系数
                delta_h_total = 0  # 单一时刻
                for k in range(0, self.N):
                    aa= (sigma_v_new[k]-sigma_pred[k])/ self.phi
                    bb= (sigma_v_new[k] - sigma_v0[k]) / (self.E * self.dt)
                    delta_epsilon =  (aa + bb) * self.dt
                    delta_e = delta_epsilon * (1 + self.e0)
                    epsilon_initial[k] = self.epsilon[m - 1][k] + delta_epsilon
                    e_initial[k] -= delta_e
                    delta_h = delta_epsilon * self.dz
                    delta_h_total +=delta_h
                    # 根据孔隙比e的变化来更新k
                    K_list_initial[k] = self.K * pow(e_initial[k] / self.e0, 3) * pow((1 + self.e0) / (1 + e_initial[k]), 1)

                # 检查收敛性
                if np.allclose(u_new, u_prev, atol=1e-6):
                    #print(f"时刻{m}迭代{iterations}次, 已收敛.")
                    break
                elif iterations > 1000:
                    print(f"时刻{m}迭代超过1000次, 可能不收敛.")
                    break
                else:
                    u_prev = np.copy(u_new)  # 重置迭代水头
                    iterations += 1  # 迭代次数加一
                    e_initial = np.copy(self.e)  # 重置e
                    #sigma_v0_initial = np.copy(sigma_v_new)  # 更新判断压缩或回弹与再压缩的有效应力

            # 更新状态变量
            self.e = e_initial
            self.K_list = np.copy(K_list_initial)
            self.u = np.copy(u_new)  # 更新u，孔压
            cum_settlement += delta_h_total
            settlement_history.append(delta_h_total)
            sigma_v0 = np.copy(sigma_v_new)  # 前一时刻有效应力
            self.epsilon[m, :] = epsilon_initial

            # 转换为水头输出
            h_list = np.zeros(self.N)
            for k in range(0, self.N):
                h = self.u_to_h(self.u[k], -self.z[k])
                h_list[k] = h
            h_all[m, :] = h_list  # 记录结果
            u_all[m, :] = self.u  # 记录结果
        if Hydraulic_Head:
            return h_all, settlement_history, cum_settlement
        else:
            return u_all, settlement_history, cum_settlement
