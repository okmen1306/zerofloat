"""Download WikiText-103 (raw) and write the first 150 MB as bytes to ./wiki.txt (the long-run data in the paper).
Usage: pip install datasets && python get_wikitext.py"""
import os
from datasets import load_dataset
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki.txt")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
b = "\n".join(ds["text"]).encode("utf-8")[:150_000_000]
open(out, "wb").write(b)
print(f"wrote {out} ({len(b) / 1e6:.0f} MB)")
