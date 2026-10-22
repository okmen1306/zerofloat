# Provenance

ZeroFloat was developed by Muhammed Ökmen in a private research repository. That repository holds the full history of all experiments and is kept unchanged as a record. This public repository contains the core code from it, unmodified apart from one error message (`train_gpu_data.py`) and two attribution comments.

## Public records

| Date | Record | Identifier |
|---|---|---|
| 2026-10-18 | ZeroFloat, version 1 (Zenodo) | DOI [10.5281/zenodo.23450386](https://doi.org/10.5281/zenodo.23450386) |
| 2026-10-22 | ZeroFloat, version 2 (Zenodo) | DOI [10.5281/zenodo.23567374](https://doi.org/10.5281/zenodo.23567374) |
| 2026-10-22 | Lattice-quantized multiplicative weights, technical report (Zenodo), the origin of the residual accumulator | DOI [10.5281/zenodo.23568755](https://doi.org/10.5281/zenodo.23568755) |

All versions of ZeroFloat: DOI [10.5281/zenodo.23450385](https://doi.org/10.5281/zenodo.23450385).

## Bitcoin timestamps (OpenTimestamps)

`timestamps/` contains OpenTimestamps proofs (`.ots`) and SHA-256 hashes (`.sha256`) of `git archive` snapshots of the private repository, taken at tagged commits:

| Snapshot | Content | SHA-256 | Bitcoin block |
|---|---|---|---|
| `v2026-10-16-olcek` | scaling results | `0c4665aa…5c6d` | 971702 |
| `v2026-10-16-300m` | 0.8M–302M results | `b942f9e1…9b81` | 971709 |
| `v2026-10-18-1b` | first 1B results | `1726485f…7fba` | — |
| `v2026-10-18-1b-esit` | 1B results | `e73cffaa…53c6` | — |
| `zenodo_v1_zerofloat.pdf` | paper version 1 | see `zenodo_v1_pdf.sha256` | — |

A proof shows that a file with the given SHA-256 existed at the time of the Bitcoin block. To verify, you need the original archive, which the author can provide on request: `ots verify <file>.ots`.
