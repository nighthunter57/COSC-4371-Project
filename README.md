# COSC 4371 Project 15: Prompt-injection detection across languages

Huynh Anh Hao Pham, COSC 4371 Security Analytics, Fall 2026.

**Question.** How well does a prompt-injection detector trained on English Safe-Guard
prompts detect injections written in German and French? Machine-translated German and
French versions of the Safe-Guard and Gandalf test injections are compared with deepset's
native German prompts, which check whether translation is a fair stand-in.

**Variable:** language of the test prompt (English, machine-translated German, native
German, machine-translated French). **Fixed:** training data (English Safe-Guard) and
classifier.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_baseline.py
```

The script downloads the data itself, so nothing needs to be copied in by hand. It takes
about 80 seconds on an Apple Silicon CPU. A second run writes byte-identical
`split.csv` and `scores.csv`.

Then download `check_checkpoint.py` from Canvas into this folder and run:

```bash
.venv/bin/python check_checkpoint.py --split split.csv --scores scores.csv --threshold 0.0416
```

## Data

Downloaded from Hugging Face at pinned commits into `data/` (not committed):

| Dataset | Revision |
|---|---|
| [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | a3a877d608f37b7d20d9945671902df895ecdb46 |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 4f61ecb038e9c3fb77e21034b22511b523772cdd |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | 04737b65e90a6794ec227012e4a255a7def6344b |

## Pipeline

1. Load all files of each dataset. Row ids are `dataset:original_split:index`.
2. Detect language with langdetect 1.0.9, seed 0.
3. Safe-Guard cleaning, in order: drop every row of a text group that has both labels;
   keep one copy of each exact duplicate; drop rows that match a deepset or Gandalf row
   exactly, after normalization, or at char 3-5 gram TF-IDF cosine >= 0.9.
4. Near-duplicate groups: two Safe-Guard prompts are linked when their normalized text
   (NFKC, casefold, punctuation and whitespace collapsed) is identical or their char 3-5
   gram TF-IDF cosine is >= 0.9. Linked prompts form connected components.
5. Split Safe-Guard 70/15/15 into train, validation and test with `StratifiedGroupKFold`
   (20 folds, seed 4371; folds 0-2 test, 3-5 validation), so no group crosses a boundary.
   All deepset and Gandalf rows are test rows. Gandalf rows are labeled 1.
6. Baseline: word 1-2 gram plus char_wb 3-5 gram TF-IDF, logistic regression with
   balanced class weights. C is chosen from {0.25, 1, 4, 16} by validation recall.
7. Threshold: the lowest 4-decimal value with validation false-positive rate <= 5%.
8. Test sets scored: Safe-Guard test, Gandalf, deepset English, deepset native German,
   deepset other languages.

## Outputs

| File | Contents |
|---|---|
| `split.csv` | Every kept row: row_id, dataset, split, label, lang, orig_split, near_dup_group |
| `scores.csv` | Every test row: row_id, test_set, label, score |
| `outputs/audit.json` | Audit numbers for the checkpoint audit table |
| `outputs/split_log.json` | Rows removed at each step, and leakage counts |
| `outputs/run_info.json` | Threshold, chosen C, validation grid, features, timing, versions |
| `outputs/checker_output.txt` | Output of the course checker on these files |
| `outputs/pr_curve_safeguard_test.png` | Precision-recall curve on the Safe-Guard test set |

## AI use

The code in this repository was written with help from Claude (Anthropic) and
reviewed by the author.
