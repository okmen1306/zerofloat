import os, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def load(data="shakespeare"):
    """data='shakespeare' (karakter, 65) veya 'wiki' (/content/wiki.txt veya ./wiki.txt, BAYT düzeyi V=256). Döner: train, val, V."""
    if data == "wiki":
        for p in ("/content/wiki.txt", os.path.join(HERE, "wiki.txt")):
            if os.path.exists(p): raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint8).astype(np.int64); break
        else: raise FileNotFoundError("wiki.txt not found: run  python get_wikitext.py  first")
        d = torch.from_numpy(raw.copy()); n = int(0.9 * len(d)); return d[:n], d[n:], 256
    txt = open(os.path.join(HERE, "shakespeare.txt"), encoding="utf-8").read(); chars = sorted(set(txt))
    d = torch.tensor([chars.index(ch) for ch in txt]); n = int(0.9 * len(d)); return d[:n], d[n:], len(chars)
