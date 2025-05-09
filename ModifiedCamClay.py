import numpy as np
import subsicence as su


class ModifiedCamClay(su.SubsidenceModel):
    def __init__(self, L=10, N=11, T=3, dt=0.1, K=0.001, w=0):
        super().__init__(L=L, N=N, T=T, dt=dt)
        self.name_chinese = '修正的剑桥模型'
        # 参数设置
        self.M_ = 1.2  # 临界状态线斜率
        self.lam = 0.01  # 压缩指数
        self.kappa = 0.0005  # 回弹指数    这两个指数根据e~lnp,e~logp,sigma~lnp,sigma~logp有关
        self.e0 = 1.5  # 初始孔隙比
        self.K = float(K)  # 初始渗透系数 m/d 这个float不加上会出大问题
        self.K_list = np.full(self.N, self.K)
        self.w = w  # 源汇项，默认为0
        self.e = np.full(self.N, self.e0)  # 孔隙比数组

    def solve(self, Hydraulic_Head=False):
        # 初始化数组
        sigma_v0 = self.sigma_total - self.u # 加荷前初始有效应力=自重应力-初始孔隙水压力, 后续计算中代表前一时刻有效应力
        p_c = np.copy(sigma_v0)  # 预固结应力
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

        # 隐式有限差分法迭代求解
        for m in range(1, self.M):
            iterations = 0
            # 保存迭代/时间步开始时初始状态
            sigma_v0_initial = np.copy(sigma_v0)
            p_c_initial = np.copy(p_c)
            e_initial = np.copy(self.e)
            K_list_initial = np.copy(self.K_list)
            # 分时刻计算
            u_prev = np.copy(self.u)  # 用于记录上一次迭代的结果
            # 构建系数矩阵
            A = np.zeros((self.N, self.N))
            # 构建常数矩阵
            b = np.zeros(self.N)
            while True:
                # 开始矩阵赋值,迭代计算
                for i in range(1, self.N - 1):
                    ki0 = 0.5 * (K_list_initial[i + 1] + K_list_initial[i])  # 代表k+
                    ki1 = 0.5 * (K_list_initial[i] + K_list_initial[i - 1])  # 代表k-
                    if sigma_v0_initial[i] > p_c_initial[i]:  # 判断是压缩过程还是回弹与在压缩过程
                        lamorkappa = self.lam  # 压缩过程
                    else:
                        lamorkappa = self.kappa  # 回弹与再压缩过程
                    # 系数矩阵赋值
                    A[i, i - 1] = ki1 / self.dz ** 2
                    A[i, i] = -(ki0 / self.dz ** 2 + ki1 / self.dz ** 2 + self.rw * lamorkappa / (
                            self.sigma_total[i] - u_prev[i]) / self.dt)
                    A[i, i + 1] = ki0 / self.dz ** 2
                    # 常数矩阵赋值
                    b[i] = self.rw * (-lamorkappa / (self.sigma_total[i] - u_prev[i]) * self.u[i] / self.dt + ki0 / self.dz
                                      - ki1 / self.dz - self.w)  # 此处ki0与ki1需要根据Z轴方向调整，此处默认Z轴方向向下

                # 边界条件赋值
                A, b = self.boundary_violation(A=A, b=b, m=m)

                # 求解线性方程组
                u_new = np.linalg.solve(A, b)
                # 计算新的有效应力
                sigma_v_new = self.sigma_total - u_new

                # 计算孔隙比变化及更新参数（临时变量）
                delta_h_total = 0  # 单一时刻
                for i in range(self.N):
                    if sigma_v_new[i] < p_c[i]:  # 单一时间步内预固结应力不变
                        if sigma_v0[i] < p_c[i]:
                            delta_e = (self.kappa / (1 + self.e0)) * np.log(sigma_v_new[i] / sigma_v0[i])  # 回弹与再压缩过程
                        else:
                            delta_e = (self.kappa / (1 + self.e0)) * np.log(sigma_v_new[i] / p_c[i])  # 回弹与再压缩过程
                    else:
                        delta_e_elastic = (self.kappa / (1 + self.e0)) * np.log(p_c[i] / sigma_v0[i])
                        delta_e_plastic = (self.lam / (1 + self.e0)) * np.log(sigma_v_new[i] / p_c[i])
                        delta_e = delta_e_elastic + delta_e_plastic
                        p_c_initial[i] = sigma_v_new[i]  # 暂时保存更新，待收敛后应用
                    e_initial[i] -= delta_e
                    delta_h = delta_e * self.dz / (1 + self.e0)  # 以沉降为正
                    delta_h_total += delta_h

                    # 根据孔隙比e的变化来更新k
                    K_list_initial[i] = self.K * pow(e_initial[i] / self.e0, 3) * pow(
                        (1 + self.e0) / (1 + e_initial[i]), 1)

                # 检查收敛性
                if np.allclose(u_new, u_prev, atol=1e-6):
                    # print(f"时刻{m}迭代{iterations}次, 已收敛.")
                    break
                elif iterations > 1000:
                    print(f"时刻{m}迭代超过1000次, 可能不收敛.")
                    break
                else:
                    u_prev = np.copy(u_new)  # 重置迭代水头
                    iterations += 1  # 迭代次数加一
                    e_initial = np.copy(self.e)  # 重置e
                    sigma_v0_initial = np.copy(sigma_v_new)  # 更新判断压缩或回弹与再压缩的有效应力

            # 更新状态变量
            self.e = np.copy(e_initial)
            self.K_list = np.copy(K_list_initial)
            p_c = np.copy(p_c_initial)
            self.u = np.copy(u_new)  # 更新u，孔压
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
