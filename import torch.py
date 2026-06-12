import torch
import math

# ===================== 任务1：正弦位置编码 Sinusoidal PE =====================
def sinusoidal_pos_encoding(seq_len: int, dim: int, device: torch.device = torch.device("cpu")):
    """
    标准正弦位置编码
    :param seq_len: 序列长度
    :param dim: 词嵌入维度（要求为偶数）
    :return: 位置编码矩阵 [seq_len, dim]
    """
    assert dim % 2 == 0, "维度必须为偶数"
    pos = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    # 计算频率项
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float, device=device) *
        (-math.log(10000.0) / dim)
    )
    pos_enc = torch.zeros((seq_len, dim), device=device)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    return pos_enc

# ===================== 任务2：二维向量旋转 =====================
def rotate_2d(x: torch.Tensor, theta: float) -> torch.Tensor:
    """
    二维向量旋转
    :param x: 输入二维向量 [2]
    :param theta: 旋转角度（弧度）
    :return: 旋转后的二维向量
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x0, x1 = x[0], x[1]
    rx0 = x0 * cos_t - x1 * sin_t
    rx1 = x0 * sin_t + x1 * cos_t
    return torch.tensor([rx0, rx1])

# ===================== 任务3：高维 RoPE 实现 =====================
def precompute_rope_cos_sin(seq_len: int, d_model: int, device: torch.device = torch.device("cpu")):
    """预计算所有位置、所有维度对应的cos、sin值"""
    assert d_model % 2 == 0, "特征维度必须为偶数"
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2, device=device).float() / d_model))
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    angle = pos * freqs
    return torch.cos(angle), torch.sin(angle)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对张量最后一维执行RoPE旋转"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # 二维旋转公式
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # 重组维度
    out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
    return out

def rope_qk(q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
    """对Q、K批量执行RoPE"""
    seq_len, d_model = q.shape
    cos, sin = precompute_rope_cos_sin(seq_len, d_model, q.device)
    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)
    return q_rot, k_rot

# ===================== 任务4：E+pos 与 RoPE 前向流程对比 =====================
def forward_with_additive_pe(emb: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor):
    """
    E+pos 流程：
    1. 词嵌入与位置编码相加，提前注入位置信息
    2. 由融合后特征生成 Q、K
    3. 计算QK点积
    特点：语义与位置特征直接加法混合，编码绝对位置
    """
    seq_len, d = emb.shape
    pe = sinusoidal_pos_encoding(seq_len, d, emb.device)
    x = emb + pe
    Q = x @ Wq
    K = x @ Wk
    return Q @ K.transpose(-1, -2)

def forward_with_rope(emb: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor):
    """
    RoPE 流程：
    1. 原始词嵌入不添加位置编码，直接生成 Q、K
    2. 对 Q、K 执行旋转变换，注入位置信息
    3. 计算QK点积
    特点：以旋转方式融入位置，内积依赖相对位置
    """
    seq_len, d = emb.shape
    Q = emb @ Wq
    K = emb @ Wk
    positions = torch.arange(seq_len, device=emb.device)
    Q_rot, K_rot = rope_qk(Q, K, positions)
    return Q_rot @ K_rot.transpose(-1, -2)

# ===================== 任务5：数值对比实验（验证相对位置特性） =====================
def exp_compare():
    torch.manual_seed(42)  # 固定随机种子，保证实验可复现
    seq_len = 8
    d_model = 8
    # 随机初始化词嵌入、Q/K权重
    emb = torch.randn(seq_len, d_model)
    Wq = torch.randn(d_model, d_model)
    Wk = torch.randn(d_model, d_model)

    # 计算两种方案的注意力矩阵
    attn_add = forward_with_additive_pe(emb, Wq, Wk)
    attn_rope = forward_with_rope(emb, Wq, Wk)

    # 选取测试位置对（相对距离均为4）
    test_pairs = [(1, 5), (3, 7), (0, 4)]
    print("===== E+pos 与 RoPE 注意力分值对比 =====")
    print("位置(pos_q, pos_k)\tE+pos点积\tRoPE点积")
    for q_pos, k_pos in test_pairs:
        val_add = attn_add[q_pos, k_pos].item()
        val_rope = attn_rope[q_pos, k_pos].item()
        print(f"({q_pos}, {k_pos})\t\t{val_add:.6f}\t{val_rope:.6f}")

# ===================== 主程序：统一运行所有测试 =====================
if __name__ == "__main__":
    # 1. 测试正弦位置编码
    print("===== 正弦位置编码测试 =====")
    pe = sinusoidal_pos_encoding(5, 8)
    print("位置编码矩阵形状：", pe.shape)
    print(pe.round(4), "\n")

    # 2. 测试二维向量旋转
    print("===== 二维向量旋转测试 =====")
    vec = torch.tensor([1.0, 0.0])
    rot0 = rotate_2d(vec, 0)
    rot90 = rotate_2d(vec, math.pi / 2)
    print(f"向量[1.0,0.0] 旋转0°：{rot0.tolist()}")
    print(f"向量[1.0,0.0] 旋转90°：{rot90.tolist()}\n")

    # 3. 对比E+pos与RoPE数值结果
    exp_compare()