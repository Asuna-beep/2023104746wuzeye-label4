import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# -------------------------- 1. 定义变换函数（彻底修复形状问题） --------------------------
def dft_1d(x):
    """手动实现一维DFT，返回一维复数数组"""
    x = np.asarray(x).flatten()
    N = len(x)
    n = np.arange(N)
    k = n.reshape((-1, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return (W @ x.reshape(-1, 1)).flatten()

def idft_1d(X):
    """手动实现一维IDFT，返回一维实数数组"""
    X = np.asarray(X).flatten()
    N = len(X)
    n = np.arange(N)
    k = n.reshape((-1, 1))
    W = np.exp(1j * 2 * np.pi * k * n / N)
    return (W @ X.reshape(-1, 1)).flatten() / N

def dct2_1d(x):
    """
    手动实现一维DCT-II（JPEG标准归一化）
    输入: x: 长度为N的一维数组
    输出: X: 长度为N的一维数组，与scipy dct(x, norm='ortho')完全一致
    """
    x = np.asarray(x).flatten()
    N = len(x)
    n = np.arange(N)
    k = n.reshape((-1, 1))
    cos_mat = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    alpha = np.ones((N, 1))
    alpha[0, 0] = np.sqrt(1 / N)
    alpha[1:, 0] = np.sqrt(2 / N)
    X = (alpha * cos_mat @ x.reshape(-1, 1)).flatten()
    return X

# -------------------------- 2. 生成测试信号（严格一维） --------------------------
x = np.array([10, 20, 30, 40, 50, 60, 70, 80])  # 严格一维，长度8
print(f"原始信号x shape: {x.shape}, x = {x}")

# -------------------------- 3. 边界延拓 --------------------------
N = len(x)
x_extend_dft = np.tile(x, 3)
x_extend_dct = np.concatenate([x[::-1], x, x[::-1]])

# -------------------------- 4. 执行变换（严格校验形状） --------------------------
# DFT变换
X_dft = dft_1d(x)
X_dft_amp = np.abs(X_dft)
print(f"DFT系数X_dft shape: {X_dft.shape}, 幅度shape: {X_dft_amp.shape}")

# DCT-II变换
X_dct = dct2_1d(x)
X_dct_scipy = dct(x, norm='ortho')
# 强制校验形状
print(f"手动DCT X_dct shape: {X_dct.shape}, scipy DCT shape: {X_dct_scipy.shape}")
if X_dct.shape == X_dct_scipy.shape:
    print(f"手动DCT与scipy DCT最大误差: {np.max(np.abs(X_dct - X_dct_scipy)):.2e}")
else:
    print(f"形状不匹配！手动DCT: {X_dct.shape}, scipy DCT: {X_dct_scipy.shape}")

# -------------------------- 5. 能量计算（严格一维） --------------------------
energy_dft = X_dft_amp ** 2
energy_dct = X_dct ** 2
energy_dft_norm = energy_dft / np.sum(energy_dft)
energy_dct_norm = energy_dct / np.sum(energy_dct)
cum_energy_dft = np.cumsum(energy_dft_norm)
cum_energy_dct = np.cumsum(energy_dct_norm)

# -------------------------- 6. 绘图（彻底修复形状问题） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 12))

# 子图1: 原始信号
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(np.arange(N), x, 'o-', color='#1f77b4', linewidth=2, markersize=8)
ax1.set_title('原始一维测试信号', fontsize=14)
ax1.set_xlabel('信号索引n', fontsize=12)
ax1.set_ylabel('像素值x[n]', fontsize=12)
ax1.grid(alpha=0.3)
ax1.set_xticks(np.arange(N))

# 子图2: 两种延拓方式对比
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(np.arange(len(x_extend_dft)), x_extend_dft, label='DFT周期延拓', color='#ff7f0e', linewidth=2)
ax2.plot(np.arange(len(x_extend_dct)), x_extend_dct, label='DCT偶对称延拓', color='#2ca02c', linewidth=2, linestyle='--')
ax2.axvline(x=N-1, color='k', linestyle=':', label='原序列边界')
ax2.axvline(x=2*N-1, color='k', linestyle=':')
ax2.set_title('DFT周期延拓 vs DCT偶对称延拓', fontsize=14)
ax2.set_xlabel('延拓后序列索引', fontsize=12)
ax2.set_ylabel('像素值', fontsize=12)
ax2.legend(fontsize=12)
ax2.grid(alpha=0.3)

# 子图3: DFT与DCT系数对比（形状完全匹配）
ax3 = fig.add_subplot(2, 2, 3)
print(f"绘图前校验: X_dft_amp shape={X_dft_amp.shape}, X_dct shape={X_dct.shape}")
ax3.bar(np.arange(N)-0.2, X_dft_amp, width=0.4, label='DFT系数幅度', color='#1f77b4', alpha=0.7)
ax3.bar(np.arange(N)+0.2, X_dct, width=0.4, label='DCT系数幅度', color='#ff7f0e', alpha=0.7)
ax3.set_title('DFT与DCT系数对比', fontsize=14)
ax3.set_xlabel('频率索引k', fontsize=12)
ax3.set_ylabel('系数幅度', fontsize=12)
ax3.legend(fontsize=12)
ax3.grid(alpha=0.3)
ax3.set_xticks(np.arange(N))

# 子图4: 累积能量对比
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(np.arange(N), cum_energy_dft, label='DFT累积能量', color='#1f77b4', linewidth=2, marker='o')
ax4.plot(np.arange(N), cum_energy_dct, label='DCT累积能量', color='#ff7f0e', linewidth=2, marker='s')
ax4.set_title('DFT与DCT能量集中性对比', fontsize=14)
ax4.set_xlabel('前k个系数', fontsize=12)
ax4.set_ylabel('累积能量占比', fontsize=12)
ax4.legend(fontsize=12)
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 1.05)
ax4.set_xticks(np.arange(N))

plt.tight_layout()
plt.savefig('dft_dct_experiment_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 7. 打印实验结果 --------------------------
print("="*60)
print("DFT系数幅度|X[k]| =", np.round(X_dft_amp, 2))
print("DCT系数X[k] =", np.round(X_dct, 3))
print("="*60)
print("前k个系数能量占比：")
print(f"k=1: DFT={cum_energy_dft[0]*100:.2f}%, DCT={cum_energy_dct[0]*100:.2f}%")
print(f"k=2: DFT={cum_energy_dft[1]*100:.2f}%, DCT={cum_energy_dct[1]*100:.2f}%")
print(f"k=4: DFT={cum_energy_dft[3]*100:.2f}%, DCT={cum_energy_dct[3]*100:.2f}%")
print("="*60)