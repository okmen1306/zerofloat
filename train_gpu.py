"""Kalite + hız + bellek: tam sayı (compact: int8 ağırlık, int16 ACC) ve float AdamW (fp32 / bf16) aynı modelde, aynı veri.
Örnek: python train_gpu.py --cfg M --steps 2000 --modes int,adam32,adam16 --int_hp 3 --adam_lr 0.001
cfg: S=0.8M, M≈10M, L≈85M (karakter-seviyesi tinyshakespeare; büyük modelde veri küçük kalır, KALİTE değil ÖLÇEK/HIZ/BELLEK kıyası içindir)."""
import argparse, time, json, numpy as np, torch, torch.nn.functional as F, os
import int_lm_torch as T
from float_baseline import FloatLM, int_init
import float_baseline
P = argparse.ArgumentParser(); P.add_argument("--cfg", default="S"); P.add_argument("--steps", type=int, default=500); P.add_argument("--modes", default="int,adam32")
P.add_argument("--int_hp", type=float, default=3); P.add_argument("--adam_lr", type=float, default=1e-3); P.add_argument("--B", type=int, default=16); P.add_argument("--Tlen", type=int, default=32)
P.add_argument("--eval_every", type=int, default=250); P.add_argument("--seed", type=int, default=0); P.add_argument("--out", default="sonuc.json"); P.add_argument("--data", default="shakespeare"); P.add_argument("--adam_cos", type=int, default=0); P.add_argument("--adam_beta2", type=float, default=0.999); P.add_argument("--adam_warm", type=int, default=0); P.add_argument("--clip", type=float, default=0.0); P.add_argument("--wq", type=float, default=0); P.add_argument("--opt", default="adamw"); P.add_argument("--sample_out", default=""); P.add_argument("--eval_final", type=int, default=0); P.add_argument("--relu_clip", type=float, default=0); P.add_argument("--int_init", type=int, default=0); a = P.parse_args()   # --opt: adamw (torch) | adam | rms | rowrms | rowrms_log2 (ablasyon: Adam -> int kuralı)   # --wq 1: Adam ağırlıkları int8 ızgarada (1/128, ±127; float master + STE). --wq K>1: tensör başına 2^k ölçek, başlangıç std ~K LSB
CFG = {"D64": dict(L=2, D=64, H=4, DH=128), "D96": dict(L=3, D=96, H=4, DH=384), "S": dict(L=4, D=128, H=4, DH=512), "M": dict(L=6, D=384, H=6, DH=1536), "L": dict(L=12, D=768, H=12, DH=3072), "XL": dict(L=24, D=1024, H=16, DH=4096), "G1": dict(L=20, D=2048, H=32, DH=8192), "tiny": dict(L=2, D=32, H=2, DH=64)}[a.cfg]
dev = "cuda" if torch.cuda.is_available() else "cpu"
from train_gpu_data import load
TR, VA, V = load(a.data)
def batch(arr, rng, B, Tl):
    st = rng.integers(0, len(arr) - Tl - 1, B); ix = torch.from_numpy(st)[:, None] + torch.arange(Tl)[None]; return arr[ix].to(dev), arr[ix + 1].to(dev)
def sync(): 
    if dev == "cuda": torch.cuda.synchronize()
class OptX:
    """Ablasyon: Adam'ı adım adım int kuralına yaklaştırır. adam: torch AdamW (wd 0) ile aynı | rms: momentum yok (b1=0) |
    rowrms: + v satır başına (int'teki gibi; 1-B tensörde tensör başına) | rowrms_log2: + gradyan büyüklüğü 2'nin kuvvetine yuvarlanır (int'teki gibi)."""
    def __init__(s, params, lr, kind, b2, eps=1e-8):
        s.P = list(params); s.b1 = 0.9 if kind == "adam" else 0.0; s.b2 = b2; s.row = kind.startswith("row"); s.lq = kind.endswith("log2"); s.eps = eps; s.t = 0
        s.param_groups = [dict(lr=lr)]
        s.m = [torch.zeros_like(p) for p in s.P] if s.b1 else None
        s.v = [torch.zeros(((p.shape[0],) + (1,) * (p.dim() - 1)) if p.dim() > 1 else (), device=p.device) if s.row else torch.zeros_like(p) for p in s.P]
    def zero_grad(s, set_to_none=True):
        for p in s.P: p.grad = None
    @torch.no_grad()
    def step(s):
        s.t += 1; lr = s.param_groups[0]["lr"]
        for i, p in enumerate(s.P):
            g = p.grad
            if g is None: continue
            if s.lq: g = torch.sign(g) * torch.exp2(torch.round(torch.log2(g.abs().clamp_min(1e-30))))
            g2 = g * g
            if s.row: g2 = g2.mean(dim=tuple(range(1, g.dim())), keepdim=True) if g.dim() > 1 else g2.mean()
            s.v[i].mul_(s.b2).add_(g2, alpha=1 - s.b2); vh = s.v[i] / (1 - s.b2 ** s.t)
            if s.b1: s.m[i].mul_(s.b1).add_(g, alpha=1 - s.b1); mh = s.m[i] / (1 - s.b1 ** s.t)
            else: mh = g
            p.add_(mh / (vh.sqrt() + s.eps), alpha=-lr)
res = {}
for mode in a.modes.split(","):
    rng = np.random.default_rng(100 + a.seed); erng = np.random.default_rng(12345); torch.manual_seed(a.seed)
    if dev == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if mode == "int":
        tr = T.IntTrainer(T.make_cfg(V=V, T=a.Tlen, B=a.B, **CFG), a.seed, a.int_hp, dev, compact=True); npar = sum(w.numel() for w in tr.Wd.values())
        state_b = sum(w.numel() * w.element_size() for w in tr.Wd.values()) + sum(w.numel() * w.element_size() for w in tr.Acc.values())
        def step(x, y): tr.step(x, y)
        def ev(x, y): return tr.eval_bits(x, y)
    else:
        float_baseline.RELU_CLIP = a.relu_clip; m = FloatLM(V, CFG["D"], CFG["L"], CFG["H"], CFG["DH"], a.Tlen).to(dev); (int_init(m, CFG["D"], CFG["DH"]) if a.int_init else None); npar = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=a.adam_lr, betas=(0.9, a.adam_beta2), weight_decay=0.0) if a.opt == "adamw" else OptX(m.parameters(), a.adam_lr, a.opt, a.adam_beta2); bf = mode == "adam16"
        PQ = list(m.parameters())
        if a.wq:                                                     # ızgara: int modeldeki depolama ile aynı (|w|<=127 adım); tensör başına ölçek (2^7..2^11) wq>1 ise
            SC = [128.0 if (a.wq == 1 or float(p.std() if p.numel() > 1 else 0) == 0) else float(2 ** np.clip(round(np.log2(a.wq / float(p.std()))), 7, 11)) for p in PQ]
            print(f"   wq {a.wq}: ölçekler {sorted(set(SC))}", flush=True)
        def step(x, y):
            if a.wq:
                MST = [p.data.clone() for p in PQ]
                for p, sc in zip(PQ, SC): p.data.copy_(torch.clamp(torch.round(p.data * sc), -127, 127) / sc)
            if a.adam_cos or a.adam_warm:
                w_ = min(1.0, (step.n + 1) / a.adam_warm) if a.adam_warm else 1.0
                c_ = 0.5 * (1 + np.cos(np.pi * min(1.0, step.n / a.steps))) if a.adam_cos else 1.0
                for g_ in opt.param_groups: g_['lr'] = a.adam_lr * w_ * c_
                step.n += 1
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=bf): lg = m(x)
            loss = F.cross_entropy(lg.float().reshape(-1, V), y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); (torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip) if a.clip else None)
            if a.wq:                                                 # gradyan nicemlenmiş noktada (STE), adım float master'a
                for p, w in zip(PQ, MST): p.data.copy_(w)
            opt.step()
        step.n = 0
        def ev(x, y):                                                # wq: değerlendirme de nicemlenmiş ağırlıkla
            if a.wq:
                MST = [p.data.clone() for p in PQ]
                for p, sc in zip(PQ, SC): p.data.copy_(torch.clamp(torch.round(p.data * sc), -127, 127) / sc)
            with torch.no_grad(): r = float(F.cross_entropy(m(x).float().reshape(-1, V), y.reshape(-1)) / np.log(2))
            if a.wq:
                for p, w in zip(PQ, MST): p.data.copy_(w)
            return r
        state_b = npar * 16 if bf else npar * 12                     # w+m+v (fp32), grad ayrı sayılmadı
    curve = []; t_acc = 0.0
    for s in range(1, a.steps + 1):
        x, y = batch(TR, rng, a.B, a.Tlen); sync(); t0 = time.time(); step(x, y); sync(); t_acc += time.time() - t0
        if s % a.eval_every == 0 or s == a.steps:
            vs = [ev(*batch(VA, erng, 32, a.Tlen)) for _ in range(3)]; curve.append((s, float(np.mean(vs)))); print(f"[{mode}] adım {s} val {curve[-1][1]:.3f} bit/kar  ({t_acc/s*1000:.0f} ms/adım)", flush=True)
    mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else float("nan")
    res[mode] = dict(params=npar, ms_per_step=t_acc / a.steps * 1000, peak_mem_MB=mem, state_bytes_per_param=state_b / npar, final_val=curve[-1][1], curve=curve)
    if a.eval_final:                                                 # BÜYÜK ölçüm: hizli_act8 --eval_final ile AYNI parçalar (tohum 777)
        frng = np.random.default_rng(777); vb = [ev(*batch(VA, frng, 32, a.Tlen)) for _ in range(a.eval_final)]
        print(f"[{mode}] BUYUK-EVAL n={a.eval_final * 32 * a.Tlen} kar: {np.mean(vb):.4f} (parça std {np.std(vb):.3f}, ort. hata {np.std(vb) / np.sqrt(len(vb)):.4f})", flush=True)
    if a.sample_out and mode != "int":                              # metin örneği (ornek.py ile aynı: "ROMEO:", temp 0.8, 2 x 400 karakter)
        ch = sorted(set(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakespeare.txt"), encoding="utf-8").read())); gs = torch.Generator(device="cpu").manual_seed(0)
        with open(a.sample_out, "a") as fo:
            fo.write(f"==== ÖRNEK Adam {a.cfg} lr {a.adam_lr} adım {a.steps} temp 0.8 (son val {curve[-1][1]:.3f})\n")
            for k in range(2):
                sq = [ch.index(c_) for c_ in "ROMEO:\n"]
                with torch.no_grad():
                    for _ in range(400):
                        l = m(torch.tensor([sq[-a.Tlen:]], device=dev))[0, -1].float().cpu(); sq.append(int(torch.multinomial(torch.softmax(l / 0.8, -1), 1, generator=gs)))
                fo.write(f"--- örnek {k + 1}\n" + "".join(ch[i] for i in sq) + "\n")
    print(f"== {mode}: {npar/1e6:.2f}M param, {res[mode]['ms_per_step']:.0f} ms/adım, tepe bellek {mem:.0f} MB, durum {state_b/npar:.2f} B/param, son val {curve[-1][1]:.3f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
