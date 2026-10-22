"""8-bit (nb) matmul ablasyonu + ölçek testi, CPU ve GPU. --phase f/b/all (hangi matmuller nicemlenir), --stoch 0 (RTN) / 1 (hepsi SR) / 2 (yalnız geri SR), --nb 0 = kırpmasız referans.
Kullanım: python hizli_act8.py --cfg M --nb 8 --phase all --stoch 2 --hp 8 --steps 800 --data shakespeare
Çıktı tek satır: (nbit-eval, tam-eval) eğrisi, ms/adım, tepe bellek."""
import argparse, time, os, hashlib, numpy as np, torch
import int_lm_torch as T
from train_gpu_data import load
P = argparse.ArgumentParser(); P.add_argument("--cfg", default="D64"); P.add_argument("--nb", type=int, default=8); P.add_argument("--phase", default="all")
P.add_argument("--stoch", type=int, default=0); P.add_argument("--steps", type=int, default=600); P.add_argument("--seed", type=int, default=0); P.add_argument("--hp", type=float, default=3)
P.add_argument("--blk", type=int, default=0); P.add_argument("--tag", default=""); P.add_argument("--data", default="shakespeare"); P.add_argument("--eval_every", type=int, default=0); P.add_argument("--B", type=int, default=16); P.add_argument("--compile", type=int, default=0); P.add_argument("--hp_end", type=float, default=0); P.add_argument("--elem_v", type=int, default=0); P.add_argument("--vshift", type=int, default=10); P.add_argument("--sched", default="lin"); P.add_argument("--upd_chunk", type=int, default=0); P.add_argument("--ckpt", default=""); P.add_argument("--ckpt_every", type=int, default=500); P.add_argument("--stop_at", type=int, default=0); P.add_argument("--wscale", type=float, default=0); P.add_argument("--fact", type=int, default=0); P.add_argument("--norm_exact", type=int, default=0); P.add_argument("--auto_rho", type=float, default=0); P.add_argument("--olc_rho", type=int, default=0); P.add_argument("--int8mm", type=int, default=0); P.add_argument("--acc32", type=int, default=0); P.add_argument("--eval_final", type=int, default=0); P.add_argument("--warm", type=int, default=0); P.add_argument("--mom", type=int, default=0); P.add_argument("--gfrac", type=int, default=0); P.add_argument("--vote", type=int, default=0); P.add_argument("--nbr", type=int, default=0); P.add_argument("--diag", type=int, default=0); P.add_argument("--autoshift", type=float, default=0); P.add_argument("--att_e", type=int, default=0); P.add_argument("--sr_upd", type=int, default=0); P.add_argument("--acc8", type=int, default=0); P.add_argument("--tail_from", type=int, default=0); P.add_argument("--tail_mult", type=float, default=1); P.add_argument("--keep", default=""); P.add_argument("--init_ckpt", default=""); P.add_argument("--nbrake", type=int, default=0); P.add_argument("--mom_g", type=int, default=0); P.add_argument("--mom_from", type=int, default=0); P.add_argument("--mom_hp", type=float, default=1); P.add_argument("--vshift2", type=int, default=0); P.add_argument("--thr_vec", type=int, default=0); P.add_argument("--thr_uni", type=int, default=0); P.add_argument("--sched_floor", type=float, default=0.005); a = P.parse_args()   # --auto_rho R: göreli adım hedefi R (hp yerine; hp programı hedefi ölçekler: R·hp0/hp_t)   # --wscale K: tensör başına ağırlık ölçeği, başlangıç std ~K LSB (0 = kapalı)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu": torch.set_num_threads(1)
CFG = {"tiny": dict(L=2, D=32, H=2, DH=64), "D64": dict(L=2, D=64, H=4, DH=128), "D96": dict(L=3, D=96, H=4, DH=384), "S": dict(L=4, D=128, H=4, DH=512),
       "M": dict(L=6, D=384, H=6, DH=1536), "L": dict(L=12, D=768, H=12, DH=3072), "XL": dict(L=24, D=1024, H=16, DH=4096), "G1": dict(L=20, D=2048, H=32, DH=8192)}[a.cfg]
exact = T.mm; st = dict(i=0, nf=10**9); g = None if a.compile else torch.Generator(device=dev).manual_seed(a.seed + 7)
def sync():
    if dev == "cuda": torch.cuda.synchronize()
def bl(m): return torch.frexp(m.double())[1].long() * (m > 0)
def shr(x, s):
    half = torch.where(s > 0, torch.ones_like(s) << (s - 1).clamp(min=0), torch.zeros_like(s)); return (x + half) >> s
qm = (1 << (max(a.nb, 1) - 1)) - 1
def q(x, ax):
    m = x.abs().amax(ax, keepdim=True); s = (bl(m) - (a.nb - 1)).clamp(min=0)
    if st.get('sr', a.stoch == 1):
        r = torch.randint(0, 1 << 30, x.shape, generator=g, device=dev) % torch.clamp(torch.ones_like(s) << s, min=1); y = (x + r) >> s
    else: y = shr(x, s)
    return y.clamp(-qm, qm), s
def mmq(x, y):
    i = st["i"]; st["i"] += 1
    use = a.phase == "all" or (a.phase == "f" and i < st["nf"]) or (a.phase == "b" and i >= st["nf"])
    if not use or a.nb == 0: return exact(x, y)
    st['sr'] = (a.stoch == 1) or (a.stoch == 2 and i >= st['nf'])
    if a.blk and x.shape[-1] % a.blk == 0 and x.shape[-1] > a.blk:                  # K ekseninde blok ölçek
        K = x.shape[-1]; out = 0
        for k0 in range(0, K, a.blk):
            x8, sx = q(x[..., k0:k0 + a.blk], -1); y8, sy = q(y[..., k0:k0 + a.blk, :], -2); out = out + (exact(x8, y8) << (sx + sy))
        return out
    x8, sx = q(x, -1); y8, sy = q(y, -2); return mm8(x8, y8) << (sx + sy)
def mm8(x8, y8):                                                    # --int8mm 1: GERÇEK int8 çekirdek (int8 x int8 -> int32). |x|,|y| <= 127, K <= 2^17 => taşma yok, sonuç exact ile BİT DÜZEYİNDE aynı
    K = x8.shape[-1]; M = x8.numel() // K; N = y8.shape[-1]
    if not a.int8mm or y8.ndim != 2 or M <= 16 or K % 8 or N % 8: return exact(x8, y8)   # dikkat matmulleri (toplu) ve V=65 başlık: eski yol
    r = torch._int_mm(x8.reshape(M, K).to(torch.int8), y8.to(torch.int8).t().contiguous().t())   # B sütun-ana: cuBLASLt int8 için hızlı yol
    return r.to(torch.int64).reshape(*x8.shape[:-1], N)
TR, VA, V = load(a.data)
def batch(arr, rng, B, Tl):
    st_ = rng.integers(0, len(arr) - Tl - 1, B); ix = torch.from_numpy(st_)[:, None] + torch.arange(Tl)[None]; return arr[ix].to(dev), arr[ix + 1].to(dev)
rng = np.random.default_rng(100 + a.seed); erng = np.random.default_rng(12345); torch.manual_seed(a.seed)
if dev == "cuda": torch.cuda.reset_peak_memory_stats()
c_ = T.make_cfg(V=V, T=32, B=a.B, **CFG); T.NORM_EXACT = bool(a.norm_exact); T.ATT_E = bool(a.att_e)
if a.wscale:
    T.LOGW = 11; T.WSH.update(T.wscale_shifts(c_, 11, a.wscale)); import collections
    print(f"   wscale {a.wscale}: kaydırma dağılımı {dict(collections.Counter(T.WSH.values()))}", flush=True)
tr = T.IntTrainer(c_, a.seed, a.hp, dev, compact=True); tr.elem = bool(a.elem_v); tr.fact = bool(a.fact); tr.auto = a.auto_rho > 0; tr.olc = bool(a.olc_rho); tr.rho_q = torch.full((), int(a.auto_rho * 65536), dtype=torch.int64, device=dev)
if tr.auto: tr.THR = torch.full((), tr.THR, dtype=torch.int64, device=dev)
tr.vshift = a.vshift; tr.chunk = a.upd_chunk; tr.mom = a.mom; tr.gfrac = bool(a.gfrac); tr.vote = a.vote; tr.nbr = a.nbr; tr.nbrake = a.nbrake; tr.mom_g = a.mom_g; tr.nbrake = a.nbrake; tr.sr_upd = bool(a.sr_upd); tr.sr_seed = a.seed; tr.acc8 = bool(a.acc8)
if a.acc8: tr.Acc = {n: v.to(torch.int8) for n, v in tr.Acc.items()}
if a.autoshift: tr.dsh = {n: torch.zeros((), dtype=torch.int64, device=dev) for n in tr.Wd}; assert a.acc32, '--autoshift --acc32 ister'
if a.acc32: tr.Acc = {n: v.to(torch.int32) for n, v in tr.Acc.items()}   # ACC int32: eşik 2^15'i aşınca int16 TAŞAR (hp > 32 => sessiz sarma)
elif max(a.hp, a.hp_end) * (1 << T.ACCF2) >= 32768 or a.auto_rho or a.warm: print(f"   UYARI: eşik int16 ACC'yi aşabilir (hp {max(a.hp, a.hp_end)}); --acc32 1 kullan", flush=True)
T.mm = mmq; ph = a.phase; a.phase = "none"; st["i"] = 0; x0, y0 = batch(TR, np.random.default_rng(1), a.B, 32); tr.eval_bits(x0, y0); st["nf"] = st["i"]; a.phase = ph   # ileri matmul sayısı
step_fn = tr.step
VEC = [n for n in tr.Wd if tr.Wd[n].ndim == 1 or n in ("E", "Pe", "Wh")]                                     # kazanç, bias, embedding, çıkış başı (basamağı model büyüdükçe küçülmeyenler)
if a.thr_vec: tr.thr_sh = {n: a.thr_vec for n in VEC}; print(f"   thr_vec: {len(VEC)} tensörde eşik x{2 ** a.thr_vec}", flush=True)   # --thr_vec k: bu tensörlerin eşiği x2^k (adım küçülür)
if a.thr_uni: ref = T.WSH.get("W10", 0); tr.thr_sh = {n: T.WSH.get(n, 0) - ref for n in tr.Wd}; print(f"   thr_uni: tek tip lr (Adam gibi), kaydırmalar {sorted(set(tr.thr_sh.values()))}", flush=True)   # gerçek adım her tensörde W1 ile aynı
if a.mom_from and a.vshift2: tr.vsh_t = torch.full((), a.vshift, dtype=torch.int64, device=dev)   # v hızı bayrağı (tensör): mom_from sonrası vshift2
if a.mom_from: tr.mom_on = torch.zeros((), dtype=torch.int64, device=dev)   # geç momentum bayrağı (tensör: compile bozulmasın)
if a.hp_end: thr_t = torch.full((), tr.THR, dtype=torch.int64, device=dev); tr.THR = thr_t      # THR tensör: compile'da her adım yeniden derleme olmasın
if a.compile: step_fn = torch.compile(tr.step)
ev = a.eval_every or max(1, a.steps // 4); t_acc = 0.0; out = []; WARM = 4 if a.compile else 0; s0 = 1
def rng_all():                                                      # tüm rastgelelik durumu: devam = kesintisiz koşu (bit düzeyinde)
    return dict(rng=rng.bit_generator.state, erng=erng.bit_generator.state, trng=torch.get_rng_state(), crng=torch.cuda.get_rng_state() if dev == "cuda" else None, g=g.get_state() if g is not None else None)
def save_ck(s):
    d = dict(s=s, Wd=tr.Wd, Acc=tr.Acc, V=tr.V, Vinit=tr.Vinit, VC=tr.VC, VCi=tr.VCi, MT=tr.MT, RHO=tr.RHO, Mm=tr.Mm, Cv=tr.Cv, NB=tr.NB, dsh=tr.dsh, out=out, t_acc=t_acc, rngs=rng_all(), args=vars(a))
    torch.save(d, a.ckpt + ".tmp"); os.replace(a.ckpt + ".tmp", a.ckpt)   # atomik: yarım dosya kalmaz
if a.init_ckpt and a.ckpt and not os.path.exists(a.ckpt):            # DAL: kayıtlı ara noktadan başla (aynı tohum/durum), sonrası yeni ayarlarla
    import shutil; shutil.copy(a.init_ckpt, a.ckpt); print(f"   dal: {a.init_ckpt} -> {a.ckpt}", flush=True)
if a.ckpt and os.path.exists(a.ckpt):
    d = torch.load(a.ckpt, map_location=dev, weights_only=False); tr.Wd, tr.Acc, tr.V, tr.Vinit, out, t_acc = d["Wd"], d["Acc"], d["V"], d["Vinit"], d["out"], d["t_acc"]; tr.VC, tr.VCi, tr.MT, tr.RHO, tr.Mm, tr.Cv = d.get("VC", {}), d.get("VCi", {}), d.get("MT", {}), d.get("RHO", {}), d.get("Mm", {}), d.get("Cv", {})
    R = d["rngs"]; rng.bit_generator.state = R["rng"]; erng.bit_generator.state = R["erng"]; torch.set_rng_state(R["trng"].cpu())
    if R["crng"] is not None: torch.cuda.set_rng_state(R["crng"].cpu())
    if g is not None and R["g"] is not None: g.set_state(R["g"].cpu())
    if d.get("dsh") is not None: tr.dsh = d["dsh"]
    s0 = d["s"] + 1; print(f"   {a.tag}checkpoint yüklendi: adım {d['s']}'den devam", flush=True)
if a.mom_from and s0 > a.mom_from + 1: tr.mom_on.fill_(1); (tr.vsh_t.fill_(a.vshift2) if tr.vsh_t is not None else None)            # kayıttan devam: momentum zaten açık
for s in range(s0, a.steps + 1):
    if a.hp_end:                                                    # hp programı: THR (≈ 1/öğrenme hızı) adım içinde değişir
        f = s / a.steps; f = f if a.sched == "lin" else 0.5 * (1 - np.cos(np.pi * f)); hp_t = a.hp + (a.hp_end - a.hp) * f
        if a.sched == "adamcos": hp_t = a.hp / max(0.5 * (1 + np.cos(np.pi * s / a.steps)), a.sched_floor)   # Adam'ın lr programıyla aynı gerçek adım: eşik = hp0 / kosinüs çarpanı (alt sınırla)
        if a.tail_from and s > a.tail_from: hp_t = hp_t * (1 + (a.tail_mult - 1) * (s - a.tail_from) / (a.steps - a.tail_from))   # son bölüm: eşik ek olarak tail_mult katına (öğr. hızı ~0'a)
        if a.mom_from and s > a.mom_from: hp_t = hp_t * a.mom_hp      # geç momentum: eşik çarpanı (momentum adımı küçültür)
        if a.warm and s < a.warm: hp_t = hp_t * a.warm / s          # ısınma: Adam'daki doğrusal lr ısınmasının karşılığı (eşik ~ 1/lr), ilk adımlar çok küçük
        if tr.auto: tr.rho_q.fill_(int(a.auto_rho * 65536 * a.hp / hp_t))   # otomatik: eşik sabit, hedef göreli adım programla azalır
        else: thr_t.fill_(int(hp_t * (1 << T.ACCF2)))
    if a.mom_from and s == a.mom_from + 1 and tr.vsh_t is not None: tr.vsh_t.fill_(a.vshift2); print(f"   v hızı: vshift {a.vshift2}", flush=True)
    if a.mom_from and s == a.mom_from + 1: tr.mom_on.fill_(1); print(f"   momentum açıldı: adım {s}", flush=True)   # geç momentum (M. Ökmen'in fikri): temel kural mom_from'a kadar, sonra ham gradyanda momentum
    tr.sr_step = s
    x, y = batch(TR, rng, a.B, 32); st["i"] = 0; sync(); t0 = time.time()
    try: step_fn(x, y)
    except Exception as e:
        if not a.compile: raise
        print(f"   compile HATA ({type(e).__name__}: {str(e)[:150]}) -> eager'e dönüldü", flush=True); a.compile = 0; step_fn = tr.step; step_fn(x, y); WARM = 0; t_acc = 0.0
    sync()
    if s > WARM: t_acc += time.time() - t0
    if a.autoshift and s % 100 == 0:                               # dinamik ölçek kontrolü (derlenmiş adımın dışında)
        thr_now = tr.THR if torch.is_tensor(tr.THR) else torch.tensor(tr.THR, device=dev)
        ks = T.rescale(tr, thr_now.to(torch.int64), a.autoshift)
        if ks: print(f"   KAYDIR adım {s}: " + ",".join(ks), flush=True)
    if s % ev == 0 or s == a.steps:
        bs = [batch(VA, erng, 32, 32) for _ in range(3)]; ph2 = a.phase; vq = []; vx = []
        for b_ in bs:
            st["i"] = 0; vq.append(tr.eval_bits(*b_)); a.phase = "none"; st["i"] = 0; vx.append(tr.eval_bits(*b_)); a.phase = ph2
        out.append((s, round(float(np.mean(vq)), 3), round(float(np.mean(vx)), 3)))
        if a.diag:                                                  # teşhis: grup başına ±127'de doyan ağırlık oranı, ort |W| (LSB), ort |ACC|/eşik
            import re, collections; Gs = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
            thr = float(tr.THR) if not torch.is_tensor(tr.THR) else float(tr.THR.item())
            for n_, w_ in tr.Wd.items():
                g_ = (re.match(r"(W1|W2|b1|b2|g1|g2|Wq|Wk|Wv|Wo)\d+$", n_) or re.match(r"(.*)", n_)).group(1); q_ = Gs[g_]
                q_[0] += int((w_.abs() >= 127).sum()); q_[1] += w_.numel(); q_[2] += float(w_.abs().float().sum()); q_[3] += float(tr.Acc[n_].abs().float().sum()) / thr
            print("   DIAG adım %d " % s + " ".join(f"{g_}:doy%{100 * q_[0] / q_[1]:.2f},|W|{q_[2] / q_[1]:.1f},A/T{q_[3] / q_[1]:.2f}" for g_, q_ in sorted(Gs.items())), flush=True)
        print(f"   {a.tag}adım {s}: nbit-eval {out[-1][1]} tam-eval {out[-1][2]} ({t_acc/max(1,s-WARM)*1000:.0f} ms/adım)", flush=True)
    if a.ckpt and (s % a.ckpt_every == 0 or s == a.stop_at): save_ck(s)
    if a.ckpt and a.keep and str(s) in a.keep.split(","):          # kalıcı ara kayıt (dallanma noktası)
        import shutil; save_ck(s); shutil.copy(a.ckpt, f"{a.ckpt}.s{s}"); print(f"   KAYIT {a.ckpt}.s{s}", flush=True)
    if a.stop_at and s == a.stop_at: print(f"   {a.tag}durduruldu: adım {s}", flush=True); raise SystemExit(0)
if a.eval_final:                                                   # BÜYÜK ölçüm: N x 32 x 32 karakter, sabit tohum (train_gpu ile AYNI parçalar), tam (nicemsiz) ileri geçiş
    frng = np.random.default_rng(777); ph2 = a.phase; a.phase = "none"; vb = []
    for _ in range(a.eval_final): st["i"] = 0; vb.append(tr.eval_bits(*batch(VA, frng, 32, 32)))
    a.phase = ph2; print(f"{a.tag}BUYUK-EVAL n={a.eval_final * 1024} kar: {np.mean(vb):.4f} (parça std {np.std(vb):.3f}, ort. hata {np.std(vb) / np.sqrt(len(vb)):.4f})", flush=True)
mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else float("nan")
print(f"{a.tag}SONUC cfg={a.cfg} data={a.data} nb={a.nb} phase={a.phase} stoch={a.stoch} hp={a.hp} seed={a.seed} steps={a.steps}: son nbit-eval {out[-1][1]} tam-eval {out[-1][2]}  {t_acc/max(1,a.steps-WARM)*1000:.0f} ms/adım  compile={a.compile} tepe bellek {mem:.0f} MB hash={hashlib.md5(b"".join(tr.Wd[n].cpu().numpy().tobytes() + tr.Acc[n].cpu().numpy().tobytes() for n in sorted(tr.Wd))).hexdigest()[:12]}", flush=True)
if tr.RHO:                                                          # göreli adım raporu (EMA, son adım): grup başına medyan ρ ve eşik çarpanı
    import re, collections; G = collections.defaultdict(list); Mg = collections.defaultdict(list)
    for n, r in tr.RHO.items(): g = (re.match(r"(W1|W2|Wq|Wk|Wv|Wo)\d+$", n) or re.match(r"(.*)", n)).group(1); G[g].append(float(r) ** 0.5 / 65536); Mg[g].append(float(tr.MT[n]) / 256 if n in tr.MT else 1.0)
    print("RHO " + " ".join(f"{g}:{np.median(v):.4f}(x{np.median(Mg[g]):.2f})" for g, v in sorted(G.items())), flush=True)
