import ClassicTerzaghi as cT
import ModifiedCamClay as mC

if __name__ == '__main__':
    model1 = cT.ClassicTerzaghi(L=10, N=20, T=5, dt=0.1, u0=98100, K=0.05)
    #model1.set_boundary_bottom(0, 1)
    u1, ss, s = model1.solve_u(Hydraulic_Head=True)
    model1.draw_3D_hydraulic_head(u1)
    print('总沉降量：太沙基%f' % s)
    model1.draw(ss)

    # model2 = mC.ModifiedCamClay(L=10, N=40, T=1, dt=0.01, u0=98100, K=0.025)
    # #model2.set_boundary_bottom(0, 1)
    # u2, s1, s2 = model2.solve(Hydraulic_Head=True) #False
    # print('总沉降量：修正剑桥%f' % s2)
    # model2.draw_3D_hydraulic_head(u2)
    # model2.draw(s1)