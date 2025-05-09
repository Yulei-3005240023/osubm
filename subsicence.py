import numpy as np
from  matplotlib import pyplot as plt

class SubsidenceModel:
    """
    关于一维沉降模型的编制需要注意几点
    1 假设总应力-地面荷载不变
    2 先假设坐标轴方向向下
    3 承压含水层，顶部水头标高大于位置水头
    4 层内水量注入或者是抽出通过源汇项w来表示
    5 一维模型不考虑偏应力作用(侧限)
    6 由于水位波动所以引起的土层沉降或者抬升
    """
    # 在这个父类中只存放与计算有关的边界条件与有限差分法设置，水文地质与土力学参数等全部放在子类中
    def __init__(self, L = 10.0, N = 10, T = 10, dt = 0.5, u0 = 0, delta_pwp = 98100, z0=40, external_load=0):
        self.name_chinese = "一维土柱沉降模拟"
        # 土水性质参数
        self.rw = 9810 # 水的容重(N/m^3)
        self.gamma_prime = 18e3 # 土的有效容重 (N/m³)

        # 计算用初始参数设置
        self.L = L  # 土层厚度 (m)
        self.N = N  # 空间节点数
        self.z0 = z0 # 初始深度，代表模拟土层的上边界（顶部）距离地表的位置
        self.dz = self.L / self.N  # 空间步长
        self.z = np.linspace(self.z0+self.dz/2, self.z0+self.L-self.dz/2, self.N) # 各层中点深度,涵盖了初始深度
        self.T = T  # 总时间 (天)
        self.dt = dt  # 时间步长 (天)
        self.M = int(self.T / self.dt) # 时间节点数

        # 边界条件预设为数组，第一项为水头或通量值，第二项为边界条件类型，1为第一类边界，2为第二类边界
        '''二类边界以后应当更改为三类边界'''
        self.boundary_top = [0, 1] # 顶部边界条件
        self.boundary_bottom = [9810, 2] # 底部边界条件-此预设对应水头的隔水边界

        # 应力/孔压pore water pressure相关变量
        self.Delta_pwp = np.zeros(self.M) #这个外加的应力会以孔压的形式全额添加到上边界条件上！！！！考虑了半天还是这个方法比较好
        self.u0 = u0 # 初始最高位置，L=0位置的孔隙水压力
        self.external_load = external_load # 初始加荷
        self.delta_pwp0 = delta_pwp
        self.Delta_pwp[0] = self.delta_pwp0 # 施加荷载 (Pa)
        self.sigma_total = self.gamma_prime * self.z + self.external_load#总应力等于自重应力加上外荷载

        self.u = np.full(self.N, self.Delta_pwp[0])        # 初始孔隙水压力 在后续计算中代表前一时刻孔隙水压力
        self.u_z = np.copy(self.u)                  # 自重应力孔压-静水压力
        for z1 in range(0, self.N):                 # 要加上与深度相关的压力
            self.u_z[z1] = self.z[z1] * self.rw                   # 自重应力孔压-静水压力
            self.u[z1] = self.u[z1] + self.u_z[z1] + self.boundary_top[0]

    def set_boundary_top (self, value, type_top):
        # 设置顶部边界条件
        self.boundary_top = [value, type_top]

    def set_boundary_bottom (self, value, type_bottom):
        # 设置底部边界条件
        self.boundary_bottom = [value, type_bottom]

    def u_to_h(self, u, z):
        # 孔压转换为水头
        h = z + u / self.rw
        return h

    def h_to_u(self, h, z):
        # 水头转化为孔压
        u = self.rw * (h - z)
        return u

    def delta_pwp_redefine(self):
        # 随便测试一下在T=0.5T时卸载压力会是什么样子
        for m in range(0, self.M):
            if m < int(self.M / 3):
                self.Delta_pwp[m] = self.delta_pwp0
            elif  int(self.M / 3)<= m < int(2*self.M / 3):
                self.Delta_pwp[m] = 0
            else:
                self.Delta_pwp[m] = self.delta_pwp0

    def boundary_violation(self, A, b, m):
        # 判断边界——顶部边界
        if self.boundary_top[1] == 1:  # 第一类边界
            A[0, 0] = 1.0
            b[0] = self.boundary_top[0] + (self.z0+self.dz/2) * self.rw + self.Delta_pwp[m]
        elif self.boundary_top[1] == 2:  # 第二类边界
            A[0, 1] = 1.0
            A[0, 0] = -1.0
            b[0] = self.boundary_top[0] * self.dz
        else:
            print('边界条件类型错误')
        # 判断边界——底部边界
        if self.boundary_bottom[1] == 1:  # 第一类边界
            A[-1, -1] = 1.0
            b[-1] = self.boundary_bottom[0] + (self.z0-self.dz / 2) * self.rw + self.Delta_pwp[m]
        elif self.boundary_bottom[1] == 2:  # 第二类边界
            A[-1, -2] = -1.0
            A[-1, -1] = 1.0
            b[-1] = self.boundary_bottom[0] * self.dz
        else:
            print('边界条件类型错误')
        return A, b

    def draw_3D_hydraulic_head(self, H_ALL): # 绘制3D图片，孔压/水位-位置-时间
        # X轴
        X = np.linspace(0, self.L, self.N)
        # 时间轴
        Y = np.linspace(0, self.T, self.M)
        # 定义初值
        X, Y = np.meshgrid(X, Y)
        # 可以plt绘图过程中中文无法显示的问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 解决负号为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, H_ALL, linewidth=0, antialiased=True, cmap=plt.get_cmap('rainbow'))
        plt.xlabel('长度')
        plt.ylabel('时间')
        plt.title(self.name_chinese + '数值解')
        plt.show()

    def draw_line(self, H_ALL: np.ndarray, time=0, title=''):  # 按给定的时刻绘制水头曲线
        # X轴
        X = np.linspace(0, self.L, self.M)
        # 可以plt绘图过程中中文无法显示的问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 解决负号为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot()
        ax.plot(X, H_ALL[time], linewidth=1, antialiased=True)

        def maxH_y(h_all):
            hy = 0
            for i in h_all:
                if i > hy:
                    hy = i
            return hy

        def minH_y(h_all):
            hy = 0
            for i in h_all:
                if i < hy:
                    hy = i
            return hy

        ax.set_ylim(minH_y(H_ALL[time]), maxH_y(H_ALL[time]))
        ax.set(ylabel='沉降量（m）', xlabel='X轴（m）')
        plt.suptitle(self.name_chinese)
        plt.title(title)
        plt.show()

    def draw(self, H_ALL: np.ndarray, title=''):  # 按给定的时刻绘制水头曲线
        # T轴
        X = np.linspace(0, self.T, self.M)
        # 可以plt绘图过程中中文无法显示的问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 解决负号为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot()
        ax.plot(X, H_ALL, linewidth=1, antialiased=True)
        ax.set(ylabel='沉降量（m）', xlabel='X轴（m）')
        plt.suptitle(self.name_chinese)
        plt.title(title)
        plt.show()
