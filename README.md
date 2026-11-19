# Molecular Representation and Tokenization for Transformer-Based de novo Drug Design

Code, executed notebooks, and evaluation outputs for a controlled comparison of **ten
molecular representation and tokenization strategies** within a fixed transformer-based
generative architecture (DrugGPT), trained on ChEMBL21. All ten share an identical
architecture, corpus, optimization schedule, and generation protocol; only the
representation differs.

---

## Strategies

**Category I — SMILES-based segmentation and encoding**

| Notebook | Strategy | Vocabulary | Compression |
|---|---|---|---|
| `Character_level.ipynb` | Character-level | 57 | 1.00× |
| `Atom_level.ipynb` | Atom-level (regex) | 225 | 1.07× |
| `Kmer.ipynb` | K-mer (k = 3, non-overlapping) | 4,648 | 2.73× |
| `BPE.ipynb` | Byte Pair Encoding (2,000 merges) | 2,059 | 4.54× |
| `SPE.ipynb` | SMILES Pair Encoding (3,000 merges) | 3,226 | 4.88× |
| `AtomInSmiles.ipynb` | Atom-in-SMILES | 1,977 | 1.07× |

**Category II — Alternative molecular string representations**

| Notebook | Strategy | Vocabulary | Compression |
|---|---|---|---|
| `SELFIES.ipynb` | SELFIES | 303 | 1.11× |
| `DeepSMILES.ipynb` | DeepSMILES | 264 | 1.04× |

**Category III — Chemistry-rule-based fragment substitution**

| Notebook | Strategy | Vocabulary | Compression |
|---|---|---|---|
| `BRICS.ipynb` | BRICS fragmentation | 38,284 | 7.18× |
| `FunctionalGroup.ipynb` | Functional-group substitution | 253 | 1.14× |

Compression is the mean number of source SMILES characters per generated token.

---

## Contents

```
notebooks/        14 executed notebooks (outputs included)
                    10 main conditions, one per strategy
                    Atom_level_seed{1337,2024}.ipynb   training-seed replicates
                    SELFIES_seed{1337,2024}.ipynb      training-seed replicates
results/          per-run and aggregated evaluation metrics (JSON)
                    benchmark_results_<strategy>.json  main protocol, 10 runs each
                    benchmark_character_level.json     aggregated values only;
                                                       per-run output is in the notebook
                    topp_results.json, topp_brics.json nucleus-sampling runs (p = 0.95)
figures/          loss trajectories for all ten conditions (CSV)
utils.py          shared evaluation helpers used by all notebooks
requirements.txt
chembl21_preprocessed.rar   preprocessed training corpus
```

Each notebook is self-contained: tokenizer definition → dataset encoding → model
definition → training → generation → evaluation.

Each `benchmark_results_*.json` holds a `per_run_results` array with one record per
generation run, plus an aggregated `summary`.

---

## Running

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Extract the corpus**

Extract `chembl21_preprocessed.rar` and place `data_for_generation_mol.txt` in the
working directory. It contains 1,498,669 molecules, each delimited by `<` and `>`.

**3. Synthetic accessibility score**

The SA score requires RDKit's contrib implementation. The notebooks download it
automatically, but it can also be fetched manually:

```bash
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
```

**4. Run a notebook**

Run all cells top to bottom. Each notebook writes a model checkpoint
(`drugGPT_<name>.pt`), the training configuration (`training_config_<name>.json`), the
loss trajectory (`loss_curve_data_<name>.csv`), and the evaluation metrics
(`benchmark_results_<name>.json`).

Training takes roughly 1.5–2.5 hours per condition on a single modern GPU.

---

## Configuration

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
output projection layers scale with vocabulary size. The shared transformer backbone is
21.38M parameters in every condition.

---

## Implementation notes

**Molecule-boundary constraint in BPE/SPE.** Merge operations are restricted to
candidate pairs that do not contain a sequence boundary marker as a substring. Without
this constraint, merges can produce composite tokens spanning the junction between
consecutive training molecules.

**Reconstruction-aware evaluation for fragment-based schemes.** BRICS and
functional-group sequences carry unfilled attachment points and are not molecules until
reconnected. Because cheminformatics parsers accept wildcard atoms and disconnected
components as syntactically valid, validity is assessed *after* reconstruction
(`BRICSBuild` and `molzip` respectively), counting reconstruction failures as invalid.

---

## License

MIT
