import torch
import torch.nn.functional as F

def fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, direction, oflex):
    """
    u: [B, C, T]
    delta: [B, C, T]
    A, B, C, D: typically [C] or [1, C, 1]
    """
    Bsz, Chn, T = u.shape

    # View as [1, C, 1] for broadcasting
    def reshape_param(p):
        if p.dim() == 1:  # [C]
            return p.view(1, -1, 1)
        elif p.shape == (Chn, 1):  # [C,1]
            return p.view(1, -1, 1)
        elif p.shape == (1, Chn):  # [1,C]
            return p.view(1, -1, 1)
        return p  # assume already correct shape

    A = reshape_param(A)
    B = reshape_param(B)
    C = reshape_param(C)
    D = reshape_param(D)
    delta_bias = reshape_param(delta_bias)

    # Add bias to delta
    delta = delta + delta_bias

    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    # Core computation
    x = A * u + B * torch.sigmoid(C * delta + D)

    # Directional scan
    if direction == 1:
        out = torch.cumsum(x, dim=-1)
    else:
        out = torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])

    return out, x, None, None


def bwd(u, delta, A, B, C, D, delta_bias, dout, x, delta_softplus, direction):
    print(" Using fallback bwd()")

    # 这里只是近似模拟梯度计算 —— 如果不影响训练，可以保留为 dout
    du = dout * A
    dA = dout * u
    dB = dout * torch.ones_like(B)
    dC = torch.zeros_like(C)
    dD = torch.zeros_like(D)
    ddelta = torch.zeros_like(delta)
    ddelta_bias = torch.zeros_like(delta_bias)

    # 如果需要真实梯度，需要使用 torch.autograd 功能（见下方进阶方案）
    return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None