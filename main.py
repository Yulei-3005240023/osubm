import ClassicTerzaghi as cT
import ModifiedCamClay as mC
import Merchant as me

if __name__ == '__main__':
    model1 = cT.ClassicTerzaghi(L=20, N=40, T=12, dt=0.1, K=0.2)
    u1, ss, s = model1.solve_u(Hydraulic_Head=True)
    model1.draw_3D_hydraulic_head(u1)
    print('总沉降量：太沙基%f' % s)
    model1.draw(ss)

    # model2 = mC.ModifiedCamClay(L=20, N=40, T=12, dt=0.1, K=0.2)
    # u2, s1, s2 = model2.solve(Hydraulic_Head=True) #False
    # print('总沉降量：修正剑桥%f' % s2)
    # model2.draw_3D_hydraulic_head(u2)
    # model2.draw(s1)

    model5 = me.Merchant(L=20, N=40, T=12, dt=0.1, K=0.2)
    u5, s7, s8 = model5.solve(Hydraulic_Head=True) #False
    model5.draw_3D_hydraulic_head(u5)
    print('总沉降量：麦坎特%f' % s8)
    model5.draw(s7)