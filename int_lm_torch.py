"""Tam sayı backprop dil modeli, PyTorch portu (numpy referansı lm.py ile BİT DÜZEYİNDE aynı olacak şekilde).
Yapılandırma sabit: 8 bit ağırlık, RMSNorm'lu transformer, RMS normalizasyonlu adım (v satır başına), çok adımlı eşik kuralı, kmax=2.
Matmul: tam sayı değerli float64 matmul (tüm aralıklar 2^53 altında, bu yüzden EXACT). Bu bir doğruluk/ölçek referansıdır,
GERÇEK int8 çekirdek değildir (bkz. bench_int8_matmul.py). compact=True: durum bellekte int8 (ağırlık) + int16 (ACC) + int32 (v) tutulur,
yani GERÇEK durum belleği ölçülebilir (hesap sırasında katman katman int64'e açılır)."""
import numpy as np, torch, time, os, sys
FA, FG, FP, ACCF2, RMS_MIN, KMAX = 8, 16, 12, 10, 8, 2
LOGW = 7                                                            # FW = bits-1 (8 bit). Hesap ızgarası; --wscale ile 11 olur
QW = 7                                                              # depolama: int8, |w| <= 2^QW - 1
NORM_EXACT = False                                                  # True: RMSNorm ortalaması D'ye tam bölme (D 2'nin kuvveti değilse 2^LOGD yerine)
ATT_E = False                                                       # True: dikkat logitleri x 1/ln2 (1477/1024): 2 tabanlı softmax = e tabanlı softmax (float ile aynı sıcaklık)
WSH = {}                                                            # tensör başına sol kaydırma: W = Wd << WSH[n]. Boşsa eski yol (bit düzeyinde aynı)
def wscale_shifts(c, logw, target):
    """Tensör başına 2'nin kuvveti ölçek: başlangıç std'si ~target LSB olsun. Depolama ölçeği 2^(logw-sh), en kaba hali eski 2^QW."""
    return {n: int(np.clip(round(logw - np.log2(target / s)), 0, logw - QW)) if s > 0 else logw - QW for n, s in c["STD"].items()}

def rs(a, n):
    return a if n <= 0 else (a + (1 << (n - 1))) >> n
def bitlen(v):                                                      # bit uzunluğu: tek çağrı (frexp, <2^53 için EXACT); eski 48 döngülü sürümle aynı (48'de kırpılır)
    return (torch.frexp(v.double())[1].long() * (v > 0)).clamp(max=48)
def isqrt(v):
    v = v.clamp(min=1); s = torch.ones_like(v) << ((bitlen(v) + 1) // 2)
    for _ in range(8): s = (s + torch.div(v, s, rounding_mode="floor")) >> 1
    return s.clamp(min=1)
def mm(a, b):                                                       # tam sayı matmul, float64 üstünden (exact)
    return (a.double() @ b.double()).round().long()
def fdiv(a, b): return torch.div(a, b, rounding_mode="floor")
def lgq(V):                                                          # log2(V) 4 kesir bitiyle (+112 ofset); v -> adım ölçeği için
    bl = bitlen(V.clamp(min=1)); mant = torch.where(bl >= 5, (V >> (bl - 5).clamp(min=0)) & 15, torch.zeros_like(bl)); return ((bl - 1) << 4) + mant + 112

def make_cfg(L=2, D=64, H=4, DH=128, T=32, B=32, V=65):
    sh = dict(E=(V, D), Pe=(T, D), Wh=(V, D), bh=(V,), gf=(D,)); fr = dict(E=1, Pe=1, Wh=0, bh=1, gf=0)
    sd = dict(E=0.3, Pe=0.1, Wh=0.1, bh=0.0, gf=0.0)
    for l in range(L):
        for n, s, f, d in (("Wq", (D, D), 0, np.sqrt(1 / D)), ("Wk", (D, D), 0, np.sqrt(1 / D)), ("Wv", (D, D), 0, np.sqrt(1 / D)),
                           ("Wo", (D, D), 0, 0.5 * np.sqrt(1 / D)), ("W1", (DH, D), 0, np.sqrt(2 / D)), ("b1", (DH,), 1, 0.0),
                           ("W2", (D, DH), 0, 0.5 * np.sqrt(1 / DH)), ("b2", (D,), 1, 0.0), ("g1", (D,), 0, 0.0), ("g2", (D,), 0, 0.0)):
            sh[f"{n}{l}"] = s; fr[f"{n}{l}"] = f; sd[f"{n}{l}"] = d
    return dict(L=L, D=D, H=H, DH=DH, T=T, B=B, V=V, SHAPES=sh, FRAC=fr, STD=sd, LOGD=int(np.log2(D)), dh=D // H, SH=int(round(np.log2(np.sqrt(D // H)))))

def init_q(c, seed, device, dtype=torch.int64):
    rng = np.random.default_rng(seed); qmax = (1 << QW) - 1
    out = {}
    for n, s in c["SHAPES"].items():
        w = rng.standard_normal(s) * c["STD"][n]
        out[n] = torch.from_numpy(np.clip(np.round(w * (1 << (LOGW - WSH.get(n, 0)))), -qmax, qmax).astype(np.int8 if dtype == torch.int8 else np.int64)).to(device)
    return out

def fwd_bwd(c, W, idx, y, sink=None):
    """W: n -> int64 tensor (çağrıda açılır). Döner gradyan sözlüğü (int64) ve logit."""
    B, T = idx.shape; D, H, dh, SH, LOGD = c["D"], c["H"], c["dh"], c["SH"], c["LOGD"]; ONE = 1 << FA
    shw = lambda a: rs(a, LOGW)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))[None, None]
    split = lambda a: a.reshape(B, T, H, dh).transpose(1, 2); merge = lambda a: a.transpose(1, 2).reshape(B, T, D)
    flat = lambda a: a.reshape(-1, a.shape[-1])
    def nfwd(x, q):
        sx = (x * x).sum(-1, keepdim=True); v = fdiv(sx + D // 2, D) if NORM_EXACT else rs(sx, LOGD); rms = isqrt(v).clamp(min=RMS_MIN)
        yy = fdiv((x << FA) + rms // 2, rms); return yy + shw(yy * q), (yy, rms)
    def nbwd(dout, cc, q):
        yy, rms = cc; dyp = dout + shw(dout * q); sm = (dyp * yy).sum(-1, keepdim=True)
        m = fdiv(sm + (D << FA) // 2, D << FA) if NORM_EXACT else rs(sm, FA + LOGD); t = dyp - rs(yy * m, FA)
        return fdiv((t << FA) + rms // 2, rms), (dout * yy).sum((0, 1))
    x = rs((W("E")[idx] + W("Pe")[:T][None]) * ONE, LOGW); caches = []
    for l in range(c["L"]):
        n1, c1 = nfwd(x, W(f"g1{l}"))
        Q = shw(mm(n1, W(f"Wq{l}").T)); K = shw(mm(n1, W(f"Wk{l}").T)); V_ = shw(mm(n1, W(f"Wv{l}").T))
        Qh, Kh, Vh = split(Q), split(K), split(V_)
        S = rs(rs(mm(Qh, Kh.transpose(-1, -2)), FA), SH)
        if ATT_E: S = rs(S * 1477, 10)
        Sm = torch.where(mask, S, torch.full_like(S, -(1 << 40))); u = Sm.amax(-1, keepdim=True) - Sm; k = u >> FA; r = u & ((1 << FA) - 1)
        mant = (1 << FP) - ((r << FP) >> (FA + 1)); ex = torch.where(k > FP + 2, torch.zeros_like(mant), mant >> k.clamp(max=62))
        A = fdiv(ex << FP, ex.sum(-1, keepdim=True).clamp(min=1))
        O = merge(rs(mm(A, Vh), FP)); x1 = x + shw(mm(O, W(f"Wo{l}").T))
        n2, c2 = nfwd(x1, W(f"g2{l}"))
        pre = shw(mm(n2, W(f"W1{l}").T) + (W(f"b1{l}") << FA)); hh = pre.clamp(0, 2 * ONE)
        x = x1 + shw(mm(hh, W(f"W2{l}").T) + (W(f"b2{l}") << FA))
        caches.append((c1, n1, Qh, Kh, Vh, A, O, c2, n2, pre, hh))
    xf, cf = nfwd(x, W("gf")); lg = shw(mm(xf, W("Wh").T) + (W("bh") << FA))
    if y is None: return None, lg
    N = B * T; yf = y.reshape(-1); o = lg.reshape(N, -1)
    u = o.amax(-1, keepdim=True) - o; k = u >> FA; r = u & ((1 << FA) - 1)
    mant = (1 << FG) - ((r << FG) >> (FA + 1)); ex = torch.where(k > FG + 2, torch.zeros_like(mant), mant >> k.clamp(max=62))
    g = fdiv(ex << FG, ex.sum(-1, keepdim=True).clamp(min=1)); g[torch.arange(N, device=g.device), yf] -= (1 << FG); g = g.reshape(B, T, -1)
    outg = {}; flush = sink if sink is not None else outg.update
    Gd = {"Wh": mm(flat(g).T, flat(xf)), "bh": g.sum((0, 1))}
    dx = shw(mm(g, W("Wh"))); dx, Gd["gf"] = nbwd(dx, cf, W("gf")); flush(Gd); Gd = {}
    for l in reversed(range(c["L"])):
        c1, n1, Qh, Kh, Vh, A, O, c2, n2, pre, hh = caches[l]
        Gd[f"W2{l}"] = mm(flat(dx).T, flat(hh)); Gd[f"b2{l}"] = dx.sum((0, 1))
        dpre = shw(mm(dx, W(f"W2{l}"))) * ((pre > 0) & (pre < 2 * ONE)); Gd[f"W1{l}"] = mm(flat(dpre).T, flat(n2)); Gd[f"b1{l}"] = dpre.sum((0, 1))
        dn2 = shw(mm(dpre, W(f"W1{l}"))); dn2, Gd[f"g2{l}"] = nbwd(dn2, c2, W(f"g2{l}"))
        dx1 = dx + dn2; Gd[f"Wo{l}"] = mm(flat(dx1).T, flat(O)); dO = split(shw(mm(dx1, W(f"Wo{l}"))))
        dA = rs(mm(dO, Vh.transpose(-1, -2)), FA); dVh = rs(mm(A.transpose(-1, -2), dO), FP)
        dS = rs(A * (dA - rs((dA * A).sum(-1, keepdim=True), FP)), FP)
        dS = dS if ATT_E else rs(dS * 11, 4)                          # ATT_E: ln2 (2 tabanı) x 1/ln2 (ölçek) = 1
        dQ = merge(rs(rs(mm(dS, Kh), FA), SH)); dK = merge(rs(rs(mm(dS.transpose(-1, -2), Qh), FA), SH)); dV = merge(dVh)
        Gd[f"Wq{l}"] = mm(flat(dQ).T, flat(n1)); Gd[f"Wk{l}"] = mm(flat(dK).T, flat(n1)); Gd[f"Wv{l}"] = mm(flat(dV).T, flat(n1))
        dn1 = shw(mm(dQ, W(f"Wq{l}"))) + shw(mm(dK, W(f"Wk{l}"))) + shw(mm(dV, W(f"Wv{l}"))); dn1, Gd[f"g1{l}"] = nbwd(dn1, c1, W(f"g1{l}"))
        dx = dx1 + dn1; flush(Gd); Gd = {}
    Gd["Pe"] = dx.sum(0); Gd["E"] = torch.zeros_like(W("E")).index_add_(0, idx.reshape(-1), dx.reshape(-1, D)); flush(Gd)
    return (outg if sink is None else None), lg

class IntTrainer:
    """Tam sayı RMS-normalizasyonlu adım (v satır başına), çok adımlı eşik. compact=True: durum int8/int16/int32 saklanır."""
    def __init__(self, c, seed, hp, device, compact=False):
        self.c, self.hp, self.compact, self.dev = c, hp, compact, device; self.THR = int(hp * (1 << ACCF2)); self.qmax = (1 << QW) - 1
        q = init_q(c, seed, device, torch.int8 if compact else torch.int64); self.Wd = q   # compact: tensör tensör doğrudan int8 (init tepe belleği ~8 B/param düşer)
        self.Acc = {n: torch.zeros(w.shape, dtype=torch.int16 if compact else torch.int64, device=device) for n, w in q.items()}
        self.V = {}; self.Vinit = {}; self.elem = False; self.vshift = 10; self.chunk = 0; self.fact = False; self.VC = {}; self.VCi = {}
        self.mom = 0; self.Mm = {}; self.gfrac = False; self.vote = 0; self.Cv = {}; self.nbr = 0; self.nbrake = 0; self.NB = {}; self.mom_g = 0; self.MG = {}; self.mom_on = None; self.vsh_t = None; self.thr_sh = None; self.nbrake = 0; self.NB = {}; self.dsh = None; self.sr_upd = False; self.acc8 = False; self.sr_seed = 0; self.sr_step = 0; self.auto = False; self.olc = False; self.rho_q = None; self.MT = {}; self.RHO = {}   # auto: tensör başına eşik çarpanı (Q8) göreli adımı rho_q/2^16 hedefinde tutar; RHO: ölçülen göreli adım (izleme)   # fact=True: satır+sütun v (Adafactor tarzı)   # elem=True: v ELEMAN başına (Adam gibi), vshift=EMA hızı
    def W(self, n):
        w = self.Wd[n].to(torch.int64)
        if self.dsh is not None: return w << (self.dsh[n] + WSH.get(n, 0))   # dinamik ölçek: kaydırma tensör (değişince yeniden derleme yok)
        s = WSH.get(n, 0); return w << s if s else w
    def step(self, idx, y):
        _, lg = fwd_bwd(self.c, self.W, idx, y, sink=self._upd)
        return lg
    def _upd(self, Gd):
        for n, gr in Gd.items():
            if n not in self.V:                                     # V/Vinit tüm tensör için bir kez açılır (ilk adım: V=g2s eşdeğeri, bkz. test)
                vs = tuple(gr.shape) if self.elem else ((gr.shape[0],) + (1,) * (gr.ndim - 1) if gr.ndim > 1 else (1,))
                self.V[n] = torch.zeros(vs, dtype=torch.int64, device=gr.device); self.Vinit[n] = torch.zeros(vs, dtype=torch.bool, device=gr.device)
            rows = gr.shape[0]; step = rows
            if self.chunk and gr.ndim > 1 and gr.numel() > self.chunk: step = max(1, self.chunk // (gr.numel() // rows))   # satır blokları: satır başına v olduğundan BİT DÜZEYİNDE aynı
            for i0 in range(0, rows, step):
                sl = slice(i0, i0 + step) if gr.ndim > 1 else slice(None)
                self._upd1(n, gr[sl], sl, step < rows)
    def _upd1(self, n, gr, sl, part):
        frc = self.c["FRAC"][n] and FG or FG + FA; a_ = gr.abs(); nz = a_ > 0; a1 = torch.where(nz, a_, torch.ones_like(a_))
        m = bitlen(a1) - 1; m = m + (a1 * a1 >= (torch.ones_like(m) << (2 * m + 1))).long()
        e = m - frc; ep = (e + 16).clamp(0, 29); g2 = torch.where(nz, torch.ones_like(ep) << (2 * ep), torch.zeros_like(ep))
        ax = tuple(range(1, gr.ndim)) if gr.ndim > 1 else tuple(range(gr.ndim))
        cnt = nz.sum(ax, keepdim=True).clamp(min=1); g2s = fdiv((g2 >> 7).sum(ax, keepdim=True), cnt); hasv = nz.any(dim=ax, keepdim=True)
        if self.elem: g2s = g2 >> 7; hasv = nz
        if self.elem and self.V[n].shape != gr.shape and not part:      # kayıttan (satır v) eleman başına v'ye geçiş: satır değerini elemanlara yay
            self.V[n] = self.V[n].expand(gr.shape).clone(); self.Vinit[n] = self.Vinit[n].expand(gr.shape).clone()
        init_ = self.Vinit[n][sl]; V_ = self.V[n][sl]; V_ = torch.where(hasv & ~init_, g2s, V_)
        V_ = torch.where(hasv & init_, V_ + ((g2s - V_) >> (self.vshift if self.vsh_t is None else self.vsh_t)), V_); self.V[n][sl] = V_; self.Vinit[n][sl] = init_ | hasv
        lg4 = lgq(V_)
        if self.fact and gr.ndim > 1 and not part:                      # Adafactor tarzı: v_ij ≈ satır_i · sütun_j / ort(satır), log2 alanında (taşma yok)
            if n not in self.VC:
                self.VC[n] = torch.zeros((1,) + tuple(gr.shape[1:]), dtype=torch.int64, device=gr.device); self.VCi[n] = torch.zeros_like(self.VC[n], dtype=torch.bool)
            cntc = nz.sum(0, keepdim=True).clamp(min=1); g2c = fdiv((g2 >> 7).sum(0, keepdim=True), cntc); hc = nz.any(dim=0, keepdim=True)
            C = self.VC[n]; Ci = self.VCi[n]; C = torch.where(hc & ~Ci, g2c, C); C = torch.where(hc & Ci, C + ((g2c - C) >> (self.vshift if self.vsh_t is None else self.vsh_t)), C)
            self.VC[n] = C; self.VCi[n] = Ci | hc
            Vm = fdiv(V_.sum(), V_.numel()).reshape((1,) * gr.ndim)
            lg4 = lg4 + lgq(C) - lgq(Vm)
        if self.gfrac:                                                  # gradyan büyüklüğü 4 kesir bitiyle (mantis + üs), 2'nin kuvveti yerine: hata %41 -> ~%3
            bl = bitlen(a1); mant = torch.where(bl >= 5, (a1 >> (bl - 5).clamp(min=0)) & 15, (a1 << (5 - bl).clamp(min=0)) & 15)
            g4 = ((bl - 1 - frc + 16) << 4) + mant; g4 = g4.clamp(0, 29 << 4)
        else: g4 = ep << 4
        sgn = torch.sign(gr) * nz.long()
        if self.mom_g and not part:                                     # MOMENTUM HAM GRADYANDA (Adam'ın m'si): m <- m + (g - m)/2^k (tam sayı); adım yönü ve büyüklüğü m'den, v ham g'den
            if n not in self.MG: self.MG[n] = torch.zeros(gr.shape, dtype=torch.int64, device=gr.device)
            Mg = self.MG[n] + rs(gr - self.MG[n], self.mom_g); self.MG[n] = Mg
            am = Mg.abs(); nzm = am > 0; am1 = torch.where(nzm, am, torch.ones_like(am)); mm_ = bitlen(am1) - 1; mm_ = mm_ + (am1 * am1 >= (torch.ones_like(mm_) << (2 * mm_ + 1))).long()
            g4m = ((mm_ - frc + 16).clamp(0, 29)) << 4; sgnm = torch.sign(Mg) * nzm.long()
            if self.mom_on is not None: on = self.mom_on > 0; g4 = torch.where(on, g4m, g4); sgn = torch.where(on, sgnm, sgn)   # geç momentum: bayrak TENSÖR -> açılınca yeniden derleme yok (m kapalıyken de güncellenir, açılışta ısınmış)
            else: g4 = g4m; sgn = sgnm
        d4 = g4 - (lg4 >> 1); k = (d4 >> 4).clamp(-ACCF2, KMAX); f = d4 & 15; val = ((16 + f) << (ACCF2 + k)) >> 4
        u = sgn * val
        if self.vote and not part:                                      # çoğunluk oylaması (yön medyanı): her ağırlıkta işaretlerin EMA'sı C in [-64, 64]; adım büyüklüğü bugünkü, yön/güç oylamadan
            if n not in self.Cv: self.Cv[n] = torch.zeros(gr.shape, dtype=torch.int16, device=gr.device)
            Cn = self.Cv[n].to(torch.int64); Cn = Cn + rs(64 * torch.sign(gr) * nz.long() - Cn, self.vote); self.Cv[n] = Cn.to(torch.int16)
            u = fdiv(val * Cn, 64)
        if self.nbr and gr.ndim > 1 and not part:                       # komşu çekimi (M. Ökmen'in fikri): adım, aynı satırdaki komşuların ortalama adımına doğru 2^-nbr oranında çekilir
            u = u + rs(fdiv(u.sum(1, keepdim=True), u.shape[1]) - u, self.nbr)
        if self.nbrake and not part:                                    # GÜRÜLTÜ FRENİ: işaret tutarlılığı C (EMA, [-64,64], int8 sığar). Yön bugünkü gradyandan,
            if n not in self.NB: self.NB[n] = torch.zeros(gr.shape, dtype=torch.int8, device=gr.device)   # büyüklük x (16 + 48|C|/64)/64: tutarlı -> 1, sallanan -> 1/4 (Adam'ın |m|/sqrt(v) freni)
            Cn = self.NB[n].to(torch.int64); Cn = Cn + rs(64 * torch.sign(gr) * nz.long() - Cn, self.nbrake); self.NB[n] = Cn.clamp(-64, 64).to(torch.int8)
            u = fdiv(u * (16 + fdiv(48 * Cn.abs(), 64)), 64)
        if self.nbrake and not part:                                    # GÜRÜLTÜ FRENİ: işaret tutarlılığı C (EMA, [-64,64], int8). Yön bugünkü gradyandan,
            if n not in self.NB: self.NB[n] = torch.zeros(gr.shape, dtype=torch.int8, device=gr.device)   # büyüklük x (16 + 48|C|/64)/64: tutarlı -> 1, sallanan -> 1/4
            Cn = self.NB[n].to(torch.int64); Cn = Cn + rs(64 * torch.sign(gr) * nz.long() - Cn, self.nbrake); self.NB[n] = Cn.clamp(-64, 64).to(torch.int8)
            u = fdiv(u * (16 + fdiv(48 * Cn.abs(), 64)), 64)
        if self.mom and not part:                                       # tam sayı momentum: normalize adımın EMA'sı, beta1 = 1 - 2^-mom (Adam'daki m'nin karşılığı)
            if n not in self.Mm: self.Mm[n] = torch.zeros(gr.shape, dtype=torch.int32, device=gr.device)
            Mv = self.Mm[n].to(torch.int64); Mv = Mv + rs(u - Mv, self.mom); self.Mm[n] = Mv.to(torch.int32); u = Mv
        if self.acc8: THRb = (self.THR << self.dsh[n]) if self.dsh is not None else self.THR; Acc = fdiv(self.Acc[n][sl].to(torch.int64) * THRb, 128) + u   # 1 baytlık kumbara: eşiğin 1/128'i çözünürlükte kesir
        else: Acc = self.Acc[n][sl].to(torch.int64) + u
        ctl = self.auto and gr.ndim > 1 and not part; olc = (self.auto or self.olc) and gr.ndim > 1 and not part                   # göreli adım denetleyicisi: yalnız matrisler, tam tensör
        if ctl and n not in self.MT: self.MT[n] = torch.full((), 256, dtype=torch.int64, device=gr.device)
        THR = (self.THR * self.MT[n]) >> 8 if ctl else self.THR
        if self.dsh is not None: THR = THR << self.dsh[n]
        if self.thr_sh is not None:                                     # tensör başına eşik kaydırması: gerçek (float birimli) öğrenme hızını tensörler arası ayarlar (Adam: hepsine aynı lr)
            ts = self.thr_sh.get(n, 0); THR = THR << ts if ts > 0 else (THR >> -ts if ts < 0 else THR)                # ağırlık 2x kaba ızgaraya geçince eşik 2x: gerçek adım büyüklüğü aynı kalır
        if self.sr_upd:                                                 # KUMBARASIZ: kesirli kısım saklanmaz; olasılıkla yuvarlanır. Rastgelelik (tohum, adım, tensör)'den
            gsr = torch.Generator(device="cpu").manual_seed((self.sr_seed * 1000003 + self.sr_step * 7919 + sum(map(ord, n))) % (1 << 62))   # yeniden üretilebilir, saklanmaz
            r = torch.randint(0, 1 << 30, u.shape, generator=gsr).to(u.device) % THR
            Acc = u + torch.sign(u) * r                                  # |u| + U[0,T) : floor((|u|+r)/T) = floor(|u|/T) + Bernoulli(frac)
        kk = Acc.abs() // THR; sg = torch.sign(Acc)
        Wn = (self.Wd[n][sl].to(torch.int64) - sg * kk).clamp(-self.qmax, self.qmax); Acc = Acc - sg * kk * THR
        if olc:
            sdw = (kk * kk).sum(); sw = (Wn * Wn).sum().clamp(min=1)
            self.RHO[n] = (self.RHO[n] - (self.RHO[n] >> 6) + (((sdw << 32) // sw) >> 6)) if n in self.RHO else (sdw << 32) // sw   # EMA: (göreli adım)^2 · 2^32
            if ctl:                                                     # sum(dW^2)/sum(W^2) > rho^2 ise eşik büyür (adım küçülür), değilse küçülür; adım %3
                Mn = self.MT[n]; d = (Mn >> 5).clamp(min=1)
                self.MT[n] = torch.where((sdw << 32) > sw * self.rho_q * self.rho_q, Mn + d, Mn - d).clamp(16, 1 << 16)
        self.Wd[n][sl] = Wn.to(self.Wd[n].dtype)
        if self.acc8:                                                   # saklarken stokastik yuvarla (beklenen değer korunur), int8'e sığar (|Acc| < THR -> |q| <= 127)
            gsr = torch.Generator(device="cpu").manual_seed((self.sr_seed * 1000003 + self.sr_step * 7919 + sum(map(ord, n))) % (1 << 62))
            r = torch.randint(0, 1 << 30, Acc.shape, generator=gsr).to(Acc.device) % THR
            self.Acc[n][sl] = fdiv(Acc * 128 + r, THR).clamp(-127, 127).to(self.Acc[n].dtype)
        elif not self.sr_upd: self.Acc[n][sl] = Acc.to(self.Acc[n].dtype)   # kumbarasız modda biriktirici YOK (durum yalnız int8 ağırlık + faktörlü v)
    def eval_bits(self, idx, y):
        _, lg = fwd_bwd(self.c, self.W, idx, None); l = lg.double() / (1 << FA); z = l - l.amax(-1, keepdim=True)
        lp = z - torch.log2(torch.exp2(z).sum(-1, keepdim=True)); return float(-lp.gather(-1, y[..., None]).mean())

def rescale(tr, thr, frac=0.002, smax=8):
    """Dinamik tensör ölçeği: |Wd| = 127'de doyan oran > frac ise Wd yarıya (en yakına), kaydırma +1, eşik 2x.
    Kalan yarım basamak biriktiriciye aktarılır (gerçek ağırlık + kesirli kısım korunur). Döner: kaydırılan tensör adları."""
    out = []
    for n, w in tr.Wd.items():
        if int(tr.dsh[n]) >= smax: continue                                          # 1-B dahil (RMSNorm kazancı g de doyuyordu)
        if float((w.abs() >= 127).float().mean()) <= frac: continue
        w64 = w.to(torch.int64); wn = rs(w64, 1); res = w64 - 2 * wn                  # res in {-1,0,1} eski basamak = yarım yeni basamak
        T_old = thr << tr.dsh[n]; tr.Acc[n] = (tr.Acc[n].to(torch.int64) + res * T_old).to(tr.Acc[n].dtype)
        tr.Wd[n] = wn.to(w.dtype); tr.dsh[n] += 1; out.append(n)
    return out
