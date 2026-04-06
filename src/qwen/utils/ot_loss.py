import torch
import torch.nn.functional as F

def compute_ot_loss_cosine(
    hidden_states_a,
    mask_a,
    hidden_states_b,
    mask_b,
    reg=0.1,
    num_iters=20,
    eps=1e-8
):
    B, T1, D = hidden_states_a.shape
    T2 = hidden_states_b.shape[1]

    # ---------------------------------------------------------
    # 1. TÍNH COSINE DISTANCE (MA TRẬN CHI PHÍ)
    # ---------------------------------------------------------
    # Chuẩn hóa độ dài các vector về 1 (L2 Normalize)
    norm_a = F.normalize(hidden_states_a, p=2, dim=-1)
    norm_b = F.normalize(hidden_states_b, p=2, dim=-1)
    
    # Tính Cosine Similarity bằng Batch Matrix Multiplication (BMM)
    # Shape: [B, T1, T2], Giá trị: [-1, 1]
    cos_sim = torch.bmm(norm_a, norm_b.transpose(1, 2))
    
    # Chuyển thành Cosine Distance (Khoảng cách = 1 - Similarity)
    # Giá trị: [0, 2] -> Đã tự nhiên được "chuẩn hóa", không lo số quá to!
    cost = 1.0 - cos_sim

    # ---------------------------------------------------------
    # 2. XỬ LÝ PADDING
    # ---------------------------------------------------------
    valid = mask_a.unsqueeze(2) * mask_b.unsqueeze(1)
    
    # MẸO QUAN TRỌNG: Đừng dùng 1e6 nữa. 
    # Vì cost lớn nhất chỉ là 2.0, ta chỉ cần gán padding = 10.0 là quá đủ.
    # Nếu dùng 1e6, exp(-1e6 / reg) vẫn sẽ về 0.0 và gây chia cho 0.
    cost_masked = cost.masked_fill(valid == 0, 10.0)

    # ---------------------------------------------------------
    # 3. THUẬT TOÁN SINKHORN
    # ---------------------------------------------------------
    a = mask_a.float()
    a = a / (a.sum(dim=1, keepdim=True) + eps)

    b = mask_b.float()
    b = b / (b.sum(dim=1, keepdim=True) + eps)

    # Tính K an toàn (số mũ cao nhất cũng chỉ là -10.0 / 0.1 = -100)
    K = torch.exp(-cost_masked / reg)

    u = torch.ones_like(a)

    for _ in range(num_iters):
        v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + eps)
        u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + eps)

    # Kế hoạch vận chuyển: [B, T1, T2]
    P = u.unsqueeze(2) * K * v.unsqueeze(1)

    # ---------------------------------------------------------
    # 4. TÍNH LOSS
    # ---------------------------------------------------------
    # Tính tổng chi phí trên các token hợp lệ (Dùng cost gốc [0, 2])
    ot_loss = (P * cost * valid).sum(dim=(1, 2)).mean()

    return ot_loss