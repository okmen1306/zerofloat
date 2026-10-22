"""Aynı mimarinin standart PyTorch (autograd, AdamW) float karşılığı: hız ve bellek kıyası için."""
import torch, torch.nn as nn, torch.nn.functional as F
RELU_CLIP = 0                                                        # >0: ReLU [0, RELU_CLIP] (tam sayı modeldeki kırpılı ReLU ile aynı mimari)
class RMS(nn.Module):
    def __init__(s, d): super().__init__(); s.g = nn.Parameter(torch.zeros(d))
    def forward(s, x): return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-3).to(x.dtype) * (1 + s.g)
class Block(nn.Module):
    def __init__(s, D, H, DH):
        super().__init__(); s.H = H; s.n1 = RMS(D); s.n2 = RMS(D)
        s.q = nn.Linear(D, D, bias=False); s.k = nn.Linear(D, D, bias=False); s.v = nn.Linear(D, D, bias=False); s.o = nn.Linear(D, D, bias=False)
        s.f1 = nn.Linear(D, DH); s.f2 = nn.Linear(DH, D)
    def forward(s, x):
        B, T, D = x.shape; h = s.n1(x); sp = lambda a: a.view(B, T, s.H, D // s.H).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(s.q(h)), sp(s.k(h)), sp(s.v(h)), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.o(a); z = s.f1(s.n2(x)); return x + s.f2(z.clamp(0, RELU_CLIP) if RELU_CLIP else torch.relu(z))
class FloatLM(nn.Module):
    def __init__(s, V, D, L, H, DH, T):
        super().__init__(); s.E = nn.Embedding(V, D); s.Pe = nn.Parameter(torch.zeros(T, D)); nn.init.normal_(s.Pe, std=0.1)
        s.blocks = nn.ModuleList(Block(D, H, DH) for _ in range(L)); s.nf = RMS(D); s.head = nn.Linear(D, V)
    def forward(s, idx):
        x = s.E(idx) + s.Pe[: idx.shape[1]]
        for b in s.blocks: x = b(x)
        return s.head(s.nf(x))

def int_init(m, D, DH):
    """Tam sayı modelle AYNI başlangıç dağılımları (int_lm_torch.make_cfg STD): adil kıyas için."""
    with torch.no_grad():
        nn.init.normal_(m.E.weight, std=0.3); nn.init.normal_(m.head.weight, std=0.1); m.head.bias.zero_()
        for b in m.blocks:
            for l in (b.q, b.k, b.v): nn.init.normal_(l.weight, std=(1 / D) ** 0.5)
            nn.init.normal_(b.o.weight, std=0.5 * (1 / D) ** 0.5); nn.init.normal_(b.f1.weight, std=(2 / D) ** 0.5); b.f1.bias.zero_()
            nn.init.normal_(b.f2.weight, std=0.5 * (1 / DH) ** 0.5); b.f2.bias.zero_()
