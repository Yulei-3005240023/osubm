import ClassicTerzaghi as cT
import ModifiedCamClay as mC
import Merchant as me
import Maxwell as ma
import Kelvin as ke
import pandas as pd
import CamClayandKelvin as cak
import CCKsp as cak2

if __name__ == '__main__':
    model1 = cT.ClassicTerzaghi(L=20, N=40, T=30, dt=0.5, K=0.5)
    u1, ss, s = model1.solve_u(Hydraulic_Head=True)
    model1.draw_3D_hydraulic_head(u1)
    print('总沉降量：太沙基%f' % s)
    model1.draw(ss)
    # # #
    # model2 = mC.ModifiedCamClay(L=40, N=40, T=40, dt=0.4, K=10)
    # u2, s1, s2 = model2.solve(Hydraulic_Head=True) #False
    # print('总沉降量：修正剑桥%f' % s2)
    # model2.draw_3D_hydraulic_head(u2)
    # model2.draw(s1)
    # #
    # model5 = me.Merchant(L=20, N=40, T=10, dt=0.1, K=0.4)
    # u5, s7, s8 = model5.solve(Hydraulic_Head=True) #False
    # model5.draw_3D_hydraulic_head(u5)
    # print('总沉降量：麦坎特%f' % s8)
    # model5.draw(s7)

    # model4 = ma.Maxwell(L=20, N=40, T=50, dt=0.1, K=0.5)
    # u4, s5, s6 = model4.solve(Hydraulic_Head=True) #False
    # model4.draw_3D_hydraulic_head(u4)
    # print('总沉降量：马克斯维尔%f' % s6)
    # model4.draw(s5)

    model3 = ke.Kelvin(L=20, N=40, T=30, dt=0.5, K=0.5)
    u3, s3, s4 = model3.solve(Hydraulic_Head=True) #False
    print('总沉降量：开尔文%f' % s4)
    model3.draw_3D_hydraulic_head(u3)
    model3.draw(s3)

    model31 = ke.Kelvin(L=20, N=40, T=30, dt=0.5, K=0.5)
    u31, s31, s41 = model3.solve(Hydraulic_Head=True) #False
    print('总沉降量：开尔文-应力期编写方法%f' % s41)
    model3.draw_3D_hydraulic_head(u31)
    model3.draw(s31)


    # model6 = cak.CamClayandKelvin(L=40, N=40, T=40, dt=0.4, K=10)
    # u5, s7, s8 = model6.solve(Hydraulic_Head=True) #False
    # model6.draw_3D_hydraulic_head(u5)
    # print('总沉降量：融合模型%f' % s8)
    # model6.draw(s7)

    # model7 = cak2.CamClayandKelvin2(L=40, N=40, T=40, dt=0.4, K=10)
    # u7, s9, s10 = model7.solve(Hydraulic_Head=True) #False
    # model7.draw_3D_hydraulic_head(u7)
    # print('总沉降量：融合模型-应力期编写方法%f' % s10)
    # model7.draw(s9)

    # df = pd.DataFrame({
    #     '剑桥模型沉降量':[s114 for s114 in s1],
    #     '融合剑桥模型沉降量':[s514 for s514 in s7]
    # })
    #
    # df.to_excel('sbwymjj.xlsx', index=False, engine='openpyxl')
    # print('ok')