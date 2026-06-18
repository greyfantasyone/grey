import matplotlib.pyplot as plt
import numpy as np

def plot_svpwm_sector2():
    # 时间轴
    t = np.linspace(0, 7, 700)
    
    # 定义状态 (Sector II: 000 -> 010 -> 110 -> 111 -> 110 -> 010 -> 000)
    # A相 (Phase U): 0 -> 0 -> 1 -> 1 -> 1 -> 0 -> 0
    # B相 (Phase V): 0 -> 1 -> 1 -> 1 -> 1 -> 1 -> 0
    # C相 (Phase W): 0 -> 0 -> 0 -> 1 -> 0 -> 0 -> 0
    
    # 构造波形数据
    Sa = np.zeros_like(t)
    Sb = np.zeros_like(t)
    Sc = np.zeros_like(t)
    
    # 设置时间段 (简化示意)
    # T0/2 | T2/2 | T6/2 | T7 | T6/2 | T2/2 | T0/2
    # 为了画图清晰，假设各段时间相等
    
    # A相: 在 V6(110) 和 V7(111) 时为高
    # 区间: [2, 5]
    Sa[200:500] = 1
    
    # B相: 在 V2(010), V6(110), V7(111) 时为高
    # 区间: [100:600]
    Sb[100:600] = 1
    
    # C相: 只有在 V7(111) 时为高
    # 区间: [300:400]
    Sc[300:400] = 1
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    ax[0].plot(t, Sa, 'r', lw=2)
    ax[0].set_ylabel('Sa (A)')
    ax[0].set_ylim(-0.2, 1.2)
    ax[0].set_title('SVPWM Timing Diagram (Sector II)')
    
    ax[1].plot(t, Sb, 'g', lw=2)
    ax[1].set_ylabel('Sb (B)')
    ax[1].set_ylim(-0.2, 1.2)
    
    ax[2].plot(t, Sc, 'b', lw=2)
    ax[2].set_ylabel('Sc (C)')
    ax[2].set_ylim(-0.2, 1.2)
    
    # 标注矢量区域
    intervals = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    labels = ['V0', 'V2', 'V6', 'V7', 'V6', 'V2', 'V0']
    for i, label in enumerate(labels):
        ax[2].text(i+0.5, -0.6, label, ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

plot_svpwm_sector2()