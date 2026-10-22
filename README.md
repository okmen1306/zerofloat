# ZeroFloat: fully integer training of language models

ZeroFloat trains transformer language models from scratch with **integer arithmetic only**: forward pass, backward pass and parameter update. There are no floating-point master weights and no floating-point optimizer state.

- Weights are int8 with a per-tensor power-of-two scale.
- The optimizer is an integer, Adafactor-like rule:
  - log2 gradient magnitudes,
  - a factored integer second moment,
  - an integer residual accumulator that moves a weight by whole int8 steps when it crosses a threshold.
- RMSNorm uses an integer square root. Attention uses an integer base-2 softmax. Backpropagation goes through both with exact integer Jacobians.

**Paper:** [paper/zerofloat.pdf](paper/zerofloat.pdf), Zenodo DOI [10.5281/zenodo.23450385](https://doi.org/10.5281/zenodo.23450385)
**Author:** Muhammed Ökmen ([ORCID 0009-0002-9709-966X](https://orcid.org/0009-0002-9709-966X))

## Results

Character-level language modeling, bits/char (lower is better), measured on 100 × 1024 fixed validation characters (the same pieces for both methods).

**Short runs** (2000 steps, Tiny Shakespeare, 3 seeds):

| Model | Params | ZeroFloat | AdamW | Gap |
|---|---|---|---|---|
| S | 0.8M | 2.627 | 2.656 | −0.029 |
| M | 10.7M | 2.557 | 2.602 | −0.045 |
| L | 85M | 2.522 | 2.542 | −0.020 |
| G1 | 1.0B | 2.610 | 2.535 | +0.075 |

**Long runs** (10k steps, WikiText-103 bytes):

| Model | ZeroFloat | AdamW | Gap |
|---|---|---|---|
| M (10.7M) | 2.154 (3 seeds) | 2.161 (best of 3 learning rates, 1 seed) | −0.007 |
| G1 (1.0B) | 2.151 (3 seeds) | 2.135 (2 seeds) | +0.016 |

At 1B, peak training memory is 11.2 GB for ZeroFloat vs. 20.2 GB for AdamW (fp32).

**Limitations:**
- ZeroFloat is **behind AdamW at 1B**, and the gap grows with model size. The paper (Sec. 6) lists what we ruled out as the cause and what did not close the gap.
- Training is short: the 1B model sees 5.1M tokens.
- No speed advantage is claimed.
- In the short runs, the AdamW baseline used standard ReLU and PyTorch's default initialization. In the long runs it used the same clipped ReLU and initialization as ZeroFloat.

## Quickstart

Requires Python ≥ 3.12.

```bash
git clone https://github.com/okmen1306/zerofloat && cd zerofloat
pip install torch numpy
bash scripts/quickstart_cpu.sh        # CPU, about 2 minutes: small model, ZeroFloat vs AdamW (1000 steps)
```

Check that integer backpropagation matches a float64 autograd twin with the same mathematics (cosine ≥ 0.99 per tensor group):

```bash
python grad_ikiz.py --cfg S
python grad_ikiz.py --cfg S --q8 1     # with 8-bit matrix inputs, as in training
```

## Reproducing the paper (GPU)

```bash
bash scripts/paper_short_runs.sh S 0   # Table 1: <S|M|L|G1> <seed>
pip install datasets && python get_wikitext.py
bash scripts/paper_long_runs.sh G1 0   # Table 2: <M|G1> <seed>
```

The 1B runs fit on one A100-40GB. For a free GPU, open [notebooks/zerofloat_colab.ipynb](notebooks/zerofloat_colab.ipynb) in Google Colab.

## Files

| File | What it is |
|---|---|
| `int_lm_torch.py` | The integer model: forward and backward pass, integer RMSNorm and softmax, and the integer optimizer (`IntTrainer`) |
| `hizli_act8.py` | ZeroFloat training script (8-bit matrix inputs, GPU/CPU, `torch.compile`) |
| `train_gpu.py` | AdamW baseline on the same architecture and data |
| `float_baseline.py` | The float model used by the baseline |
| `grad_ikiz.py` | Backpropagation check against a float64 autograd twin |
| `train_gpu_data.py`, `get_wikitext.py` | Data |

The code is exactly what produced the paper's numbers. Comments and printed labels are partly in Turkish:

| Printed label | Meaning |
|---|---|
| `adım` | step |
| `nbit-eval` | evaluation with 8-bit inputs |
| `tam-eval` | evaluation with exact inputs |
| `BUYUK-EVAL` | the large evaluation used in the paper |
| `ms/adım` | ms per step |
| `tepe bellek` | peak memory |
| `kos` / `işaret` | cosine similarity / sign agreement (`grad_ikiz.py`) |

Main ZeroFloat options in `hizli_act8.py`:

| Option | What it does |
|---|---|
| `--hp` / `--hp_end` / `--sched cos` | Threshold, the inverse learning rate, and its schedule |
| `--fact 1` | Factored second moment |
| `--vshift k` | Second-moment rate (β2 = 1 − 2^−k) |
| `--wscale 16` | Per-tensor weight scale |
| `--acc32 1` | 32-bit accumulator |
| `--att_e 1` | Softmax temperature fix |
| `--nbr 3` | Neighbor pull |
| `--tail_from` / `--tail_mult` | Tail brake |
| `--acc8 1` | 1-byte accumulator (tested only on the small model) |

## License

Free for personal, research and other non-commercial use under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). Commercial use requires a paid license: okmen1306@gmail.com. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{okmen2026zerofloat,
  author = {Ökmen, Muhammed},
  title  = {ZeroFloat: Fully Integer Training of Language Models up to 1B Parameters},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.23450385}
}
```

The residual accumulator comes from the author's earlier work on lattice-quantized multiplicative weights ([paper/kafes_mwu.pdf](paper/kafes_mwu.pdf), DOI [10.5281/zenodo.23568755](https://doi.org/10.5281/zenodo.23568755)). Provenance and timestamps: [PROVENANCE.md](PROVENANCE.md).
