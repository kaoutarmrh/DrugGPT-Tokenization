# Molecular Representation and Tokenization for Transformer-Based de novo Drug Design

Code, executed notebooks, and results for a controlled comparison of **ten molecular
representation and tokenization strategies** within a fixed transformer-based generative
architecture (DrugGPT), trained on ChEMBL21.

---

## Overview

Ten representation and tokenization strategies are evaluated under an identical
architecture, corpus, optimization schedule, and generation protocol. They fall into
three conceptually distinct categories:

**Category I — SMILES-based segmentation and encoding**
| Notebook | Strategy | Vocab | Compression |
|---|---|---|---|
| `01_character_level.ipynb` | Character-level | 57 | 1.00× |
| `02_atom_level.ipynb` | Atom-level (regex) | 225 | 1.07× |
| `03_kmer.ipynb` | K-mer (k = 3, non-overlapping) | 4,648 | 2.73× |
| `04_bpe.ipynb` | Byte Pair Encoding (2,000 merges) | 2,059 | 4.54× |
| `05_spe.ipynb` | SMILES Pair Encoding (3,000 merges) | 3,226 | 4.88× |
| `06_atom_in_smiles.ipynb` | Atom-in-SMILES | 1,977 | 1.07× |

**Category II — Alternative molecular string representations**
| Notebook | Strategy | Vocab | Compression |
|---|---|---|---|
| `07_selfies.ipynb` | SELFIES | 303 | 1.11× |
| `08_deepsmiles.ipynb` | DeepSMILES | 264 | 1.04× |

**Category III — Chemistry-rule-based fragment substitution**
| Notebook | Strategy | Vocab | Compression |
|---|---|---|---|
| `09_brics.ipynb` | BRICS fragmentation | 38,284 | 7.18× |
| `10_functional_group.ipynb` | Functional-group substitution | 253 | 1.14× |

---

## Key results

All values are mean ± SD across **10 independent generation runs of 500 attempts each**
(5,000 attempts per condition, 50,000 total).

| Strategy | Validity (%) | Unique. (%) | Novelty (%) | IntDiv | Drug-like (%) | QED | SA | MW (Da) |
|---|---|---|---|---|---|---|---|---|
| Character-level | 92.62 ± 1.26 | 100.00 | 97.10 | 0.9102 | 75.94 | 0.577 | 2.779 | 387.6 |
| Atom-level | 93.16 ± 1.36 | 100.00 | 97.40 | 0.9090 | 74.98 | 0.570 | 2.811 | 396.8 |
| K-mer (k=3) | 90.62 ± 1.37 | 99.96 | 96.16 | 0.9077 | 76.79 | 0.583 | 2.730 | 387.6 |
| BPE | 93.96 ± 1.01 | 99.96 | 93.33 | 0.9090 | 82.78 | 0.613 | 2.679 | 368.3 |
| SPE | 93.80 ± 1.08 | 99.94 | 93.04 | 0.9094 | 80.67 | 0.596 | 2.731 | 375.9 |
| Atom-in-SMILES | 93.92 ± 0.93 | 100.00 | 99.19 | 0.9077 | 76.80 | 0.576 | 2.674 | 387.1 |
| **SELFIES** | **99.84 ± 0.20** | 100.00 | 99.36 | 0.9134 | 80.81 | 0.581 | 3.197 | 376.3 |
| DeepSMILES | 68.32 ± 1.67 | 99.94 | 99.03 | 0.9088 | 77.26 | 0.579 | 2.717 | 384.3 |
| **BRICS** | 87.56 ± 1.10 | 85.13 | 91.64 | **0.9232** | **88.21** | **0.659** | **2.098** | 247.1 |
| Functional group | 57.34 ± 1.99 | 100.00 | 99.12 | 0.9085 | 82.10 | 0.627 | 2.597 | 357.6 |

**Main findings**

1. **Validity and molecular quality are not aligned objectives.** SELFIES achieves the
   highest validity (99.84%) but the least favourable synthetic accessibility (3.197);
   BRICS and functional-group substitution achieve the best drug-likeness, QED and SA
   scores but the lowest validity among reconstruction-capable conditions.
2. **Molecular weight mediates the quality metrics.** Across the ten conditions, mean MW
   correlates with drug-likeness at r = −0.88 and with SA score at r = 0.80.
3. **Vocabulary granularity governs novelty and generalization.** BRICS, SPE and BPE show
   the lowest novelty and the widest train/validation loss gaps.
4. **For five representations, decoding/reconstruction is often the bottleneck**, with
   failure rates from 0.3% (SELFIES) to 42.7% (functional group).

---

## Repository structure

```
notebooks/     10 executed notebooks, one per condition (outputs included)
results/       per-condition loss curves (CSV) and benchmark summaries (JSON)
               RESULTS_SUMMARY.md — all tables in one place
figures/       multi-panel loss figure (PDF/PNG) and its underlying data
utils.py       shared evaluation helpers used by all notebooks
requirements.txt
```

Each notebook is self-contained: tokenizer definition → dataset encoding → model
definition → training → generation → evaluation. All ten share an identical model
architecture and training configuration; only the representation differs.

---

## Reproducing

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Obtain the data**

Preprocessed ChEMBL21 SMILES (1,498,669 molecules, 75,602,998 characters) must be
placed in the working directory as `data_for_generation_mol.txt`, one molecule per
entry delimited by `<` and `>`.

**3. Synthetic accessibility score**

The SA score requires RDKit's contrib implementation. The notebooks download it
automatically, but you can also fetch it manually to avoid rate limiting:

```bash
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
```

**4. Run a notebook**

Run all cells top to bottom. Each notebook writes:

- `drugGPT_<name>.pt` — model checkpoint
- `training_config_<name>.json` — full training configuration and final losses
- `loss_curve_data_<name>.csv` — raw loss trajectory
- `benchmark_results_<name>.json` — per-run and aggregated metrics

Training takes roughly 1.5–2.5 hours per condition on a single modern GPU.

---

## Experimental configuration

Identical across all ten conditions:

| Parameter | Value |
|---|---|
| Architecture | Decoder-only transformer, 12 layers, 12 heads, 384-dim |
| Feed-forward | 4× expansion (1,536), GELU, Pre-LN |
| Context length | 256 tokens |
| Dropout | 0.15 |
| Optimizer | AdamW, lr 1e-4, weight decay 0.01 |
| Schedule | Cosine annealing to 1e-5 |
| Training steps | 25,000 |
| Batch size | 32 |
| Random seed | 42 |
| Generation | 10 runs × 500 attempts, temperature U(0.7, 1.2), top-k U(30, 100) |

Total parameter count varies from 21.42M to 50.82M because the token embedding and
output projection layers scale with vocabulary size. The shared transformer backbone
is 21.38M parameters in every condition.

---

## Implementation notes

Two implementation details proved consequential and are documented here because they
materially affect benchmark conclusions:

**Molecule-boundary constraint in BPE/SPE.** Merge operations must be restricted to
candidate pairs that do not contain a sequence boundary marker as a substring.
Without this constraint, merges can produce composite tokens spanning the junction
between consecutive training molecules.

**Reconstruction-aware evaluation for fragment-based schemes.** BRICS and
functional-group sequences carry unfilled attachment points and are not molecules until
reconnected. Because cheminformatics parsers accept wildcard atoms and disconnected
components as syntactically valid, validity must be assessed *after* reconstruction
(`BRICSBuild` and `molzip` respectively), counting reconstruction failures as invalid.

---

## Citation

If you use this code, please cite the associated paper.

```bibtex
@article{mrhar_tokenization,
  title  = {Transformer-Based Generative Models for Drug Discovery:
            A Comparative Study of Molecular Representation and Tokenization Strategies},
  author = {M'Rhar, Kaoutar and Chadi, Mohamed-Amine and Mousannif, Hajar},
  note   = {Department of Computer Science, Cadi Ayyad University, Marrakesh, Morocco}
}
```

## License

MIT
