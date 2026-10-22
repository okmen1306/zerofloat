"""Geri yayılım doğruluğu: AYNI ağırlık + AYNI veri ile tam sayı gradyanı (int_lm_torch.fwd_bwd) vs float64 ikiz modelin autograd gradyanı.
Float ikiz, tam sayı modelin matematiğini birebir taşır (2 tabanlı softmax = e tabanlı softmax(S*ln2), kırpılı ReLU, (1+g) kazanç), yalnız
yuvarlama/sabit nokta yoktur. Çıktı: tensör grubu başına kosinüs benzerliği ve işaret uyumu.
Kullanım: python grad_ikiz.py [--ckpt kayit.pt] [--cfg S] [--att_e 1] [--q8 1 (8 bit matmul girdileri, stokastik geri)]"""
import argparse, re, collections, numpy as np, torch
import int_lm_torch as T
from train_gpu_data import load
P = argparse.ArgumentParser(); P.add_argument("--ckpt", default=""); P.add_argument("--cfg", default="S"); P.add_argument("--att_e", type=int, default=1)
P.add_argument("--q8", type=int, default=0); P.add_argument("--seed", type=int, default=0); P.add_argument("--B", type=int, default=16); a = P.parse_args()
CFG = {"S": dict(L=4, D=128, H=4, DH=512), "M": dict(L=6, D=384, H=6, DH=1536)}[a.cfg]
TR, VA, V = load("shakespeare"); c = T.make_cfg(V=V, T=32, B=a.B, **CFG); T.NORM_EXACT = True; T.ATT_E = bool(a.att_e)
T.LOGW = 11; T.WSH.update(T.wscale_shifts(c, 11, 16))
tr = T.IntTrainer(c, a.seed, 3.6, "cpu", compact=True)
if a.ckpt: tr.Wd = torch.load(a.ckpt, weights_only=False)["Wd"]
if a.q8:                                                             # 8 bit matmul girdileri (hizli_act8 ile aynı): ileri RTN, geri stokastik
    g = torch.Generator().manual_seed(7); st = {"i": 0, "nf": 10**9}
    def bl(m): return torch.frexp(m.double())[1].long() * (m > 0)
    def q(x, ax, sr):
        m = x.abs().amax(ax, keepdim=True); s = (bl(m) - 7).clamp(min=0)
        if sr: r = torch.randint(0, 1 << 30, x.shape, generator=g) % torch.clamp(torch.ones_like(s) << s, min=1); y = (x + r) >> s
        else: y = T.rs_t(x, s) if hasattr(T, "rs_t") else (x + torch.where(s > 0, torch.ones_like(s) << (s - 1).clamp(min=0), torch.zeros_like(s))) >> s
        return y.clamp(-127, 127), s
    exact = T.mm
    def mmq(x, y):
        i = st["i"]; st["i"] += 1; sr = i >= st["nf"]; x8, sx = q(x, -1, sr); y8, sy = q(y, -2, sr); return exact(x8, y8) << (sx + sy)
rng = np.random.default_rng(100 + a.seed); s_ = rng.integers(0, len(TR) - 33, a.B); ix = torch.from_numpy(s_)[:, None] + torch.arange(32)[None]
idx, yy = TR[ix], TR[ix + 1]
if a.q8:
    T.mm = mmq; st["i"] = 0; T.fwd_bwd(c, tr.W, idx, None); st["nf"] = st["i"]; st["i"] = 0
G, _ = T.fwd_bwd(c, tr.W, idx, yy)
T.mm = T.mm if not a.q8 else exact

# ---- float64 ikiz ----
Wf = {n: (tr.W(n).double() / 2 ** 11).requires_grad_(True) for n in tr.Wd}
L_, D, H, dh = c["L"], c["D"], c["H"], c["dh"]; ln2 = np.log(2.0); att_sc = (1477 / 1024) if a.att_e else 1.0
def rms(x, gn): return x / torch.sqrt((x * x).mean(-1, keepdim=True)).clamp(min=8 / 256) * (1 + gn)
B_, T_ = idx.shape; x = Wf["E"][idx] + Wf["Pe"][:T_][None]; mask = torch.tril(torch.ones(T_, T_, dtype=torch.bool))
sp = lambda t: t.reshape(B_, T_, H, dh).transpose(1, 2); mg = lambda t: t.transpose(1, 2).reshape(B_, T_, D)
for l in range(L_):
    n1 = rms(x, Wf[f"g1{l}"]); Q, K, Vv = sp(n1 @ Wf[f"Wq{l}"].T), sp(n1 @ Wf[f"Wk{l}"].T), sp(n1 @ Wf[f"Wv{l}"].T)
    S = (Q @ K.transpose(-1, -2)) / (2 ** c["SH"]) * att_sc; S = S.masked_fill(~mask, -1e30); A = torch.softmax(S * ln2, -1)
    x1 = x + mg(A @ Vv) @ Wf[f"Wo{l}"].T; n2 = rms(x1, Wf[f"g2{l}"])
    h = (n2 @ Wf[f"W1{l}"].T + Wf[f"b1{l}"]).clamp(0, 2); x = x1 + h @ Wf[f"W2{l}"].T + Wf[f"b2{l}"]
lg = rms(x, Wf["gf"]) @ Wf["Wh"].T + Wf["bh"]
loss = torch.nn.functional.cross_entropy((lg * ln2).reshape(-1, V), yy.reshape(-1), reduction="sum"); loss.backward()

# ---- karşılaştırma ----
Gs = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])
for n in tr.Wd:
    gi = G[n].double().flatten(); gf = Wf[n].grad.flatten(); k = (re.match(r"(W1|W2|b1|b2|g1|g2|Wq|Wk|Wv|Wo)\d+$", n) or re.match(r"(.*)", n)).group(1); z = Gs[k]
    z[0] += float(gi @ gf); z[1] += float(gi @ gi); z[2] += float(gf @ gf); m = gf.abs() > gf.abs().median(); z[3] += int(((gi.sign() == gf.sign()) & m).sum()); z[4] += int(m.sum())
print(f"cfg={a.cfg} ckpt={a.ckpt or 'başlangıç'} att_e={a.att_e} q8={a.q8}  | grup: kosinüs(int, float) / işaret uyumu (büyük yarı)")
for k, z in sorted(Gs.items()): print(f"  {k:4s} kos {z[0] / (np.sqrt(z[1] * z[2]) + 1e-30):+.4f}  işaret %{100 * z[3] / max(1, z[4]):.1f}")
