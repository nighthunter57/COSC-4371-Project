# Project 15: Prompt-injection detection across datasets

Huynh Anh Hao Pham | COSC 4371 Security Analytics | Fall 2026

> **Status.** The project now follows the assigned language variant. The original
> proposal is kept below as a historical record. Its obfuscation and normalization plan
> and its 80/20 split were replaced; see [Current experiment](#current-experiment) for
> what the code actually does.

## Original proposal (historical, as submitted)

### 1. Security question

My question is: How well can TF-IDF with logistic regression and DistilBERT trained on xTRam1 detect prompt injections in deepset and Gandalf when the text is changed using zero-width characters, homoglyphs, or Base64 encoding, and does normalizing the text improve detection?

I will compare the two models with ProtectAI’s DeBERTa detector. I want to see which attacks they miss and whether cleaning the text helps without flagging more harmless prompts. A developer or security analyst could use the results to decide whether to screen prompts with one of these detectors. This project tests detector evasion; it does not test an LLM’s response.

### 2. Dataset and provenance

I downloaded all seven Parquet files from these Hugging Face datasets and ran the initial audit on October 6, 2026. The version IDs let me reuse the same files if a dataset changes later.

- xTRam1 Safe-Guard Prompt Injection [1], version a3a877d6: 10,296 rows in the training and test files (8,236 and 2,060 rows); 2,490,968 bytes in total. The authors used GPT-3.5-turbo to generate attacks in several categories, starting from prompts in public datasets. The documentation does not give collection dates or explain how each individual label was checked.
- deepset Prompt Injections [2], version 4f61ecb0: 662 rows in the training and test files (546 and 116 rows); 51,215 bytes. The documentation lists the columns and split sizes but does not explain where or when the prompts were collected, or how they were labeled.
- Lakera Gandalf Ignore Instructions [3], version 04737b65: 1,000 rows in the training, validation, and test files (777, 111, and 112 rows); 51,515 bytes. These prompts were submitted to the Gandalf game in July 2023. Lakera selected prompts similar in meaning to “Ignore all previous instructions,” removed possible personal information and near-duplicates, and sampled 1,000 prompts.

xTRam1 provides the largest set for training. The other datasets let me test different kinds of prompts, including real attempts from Gandalf. I plan to use Google Colab, a small GPU batch size for DistilBERT, and early stopping based on validation performance. I have not trained the models yet.

### 3. Observation unit

One row represents one text prompt. xTRam1 and deepset contain text and label; Gandalf contains text and similarity. None includes user, session, campaign, or source-document IDs, so I cannot count repeated users or campaigns. I will give each prompt an ID and keep its changed versions linked to it. This lets me compare each changed prompt with the original without counting the versions as independent examples.

### 4. Target

For xTRam1 and deepset, I will predict label: 0 means benign and 1 means injection. Neither dataset has missing or unknown label values. The xTRam1 documentation describes generated attacks and selected safe prompts; deepset does not explain how labels were assigned or checked. I will use the provided labels and report disagreements found in the audit.

Gandalf has no binary label column. Its prompts are intended to be injections, but Lakera warns that a few may have been included by mistake [3]. I will calculate recall assuming all 1,000 are attacks and state that limitation. There are no benign examples for measuring false positives. I will not use similarity as a label or an input to the models.

### 5. Initial audit and known limitations

I checked all 11,958 rows before cleaning the data. I looked for missing values in every column. Two rows counted as exact duplicates if their full text values matched. I did not change spaces, capitalization, or characters before this check. In the table, “extra rows” means the copies left over after keeping one row for each distinct text.

| Audit item | xTRam1 | deepset | Gandalf |
|---|---|---|---|
| Rows / columns | 10,296 / 2 | 662 / 2 | 1,000 / 2 |
| Benign target | 7,150 (69.44%) | 399 (60.27%) | Not available |
| Injection target | 3,146 (30.56%) | 263 (39.73%) | 1,000 intended attacks (100% assumed) |
| Missing fields | 0 in text or label | 0 in text or label | 0 in text or similarity; label not supplied |
| Empty text | 0 | 0 | 0 |
| Exact duplicates | 91 groups; 159 extra rows | 0 groups / extra rows | 0 groups / extra rows |
| Released split overlap | 35 exact texts in both train and test | 0 train-test overlaps | 0 overlaps across splits |
| Label conflicts | 2 text groups have both labels | 0 | Not applicable |
| Repeated entities | Unavailable: no IDs | Unavailable: no IDs | Unavailable: no IDs |
| Available time | No row timestamps; collection dates unavailable | No row timestamps; collection dates unavailable | July 2023 collection; no row timestamps |

Six exact prompts appear in both xTRam1 and deepset. Five are labeled benign in xTRam1 but injection in deepset. This could come from different labeling rules, missing context, or mistakes; the audit alone cannot tell which. I found no exact matches between Gandalf and either of the other datasets. I will remove any xTRam1 prompt that also appears in an external test set before training.

**Labels and missing IDs.** Without user or campaign IDs, I cannot check whether several prompts came from the same person or attack campaign. The limited labeling information also makes conflicts hard to resolve. Incorrect labels could make a model look better or worse than it is. I will report conflicting groups separately and will not use the model’s predictions to decide which label is correct.

**Different data sources.** A model could learn wording common in generated xTRam1 attacks and miss other kinds of prompts. Gandalf also contains a selected group of game submissions, so it does not represent every real attack. Detecting most Gandalf prompts would be useful, but it would not show how often the same model flags harmless prompts.

**Language.** A check with langdetect (version 1.0.9, seed 0) [6] classified 265 of 662 deepset prompts (40.0%) as German. It classified 98.8% of xTRam1 and 95.0% of Gandalf as English. My models will train on mostly English data, and ProtectAI also targets English [4]. I will report deepset results by language and manually review uncertain language labels.

**Similar prompts.** I have checked exact duplicates, but I have not finished checking prompts that differ by only a few characters or words. These could make test scores too high if similar examples appear in training. I will compare normalized text and character n-grams within and across datasets, then group similar prompts before splitting. I will set the matching rule before testing and report how many records it affects.

**ProtectAI’s training data.** ProtectAI used several public datasets but does not list every training record [4]. Some of my test prompts may therefore be familiar to that model. I will compare its results with mine, while making clear that I cannot verify whether its original training data overlap these tests.

**Text changes and long prompts.** The three text changes cover only part of what an attacker could try. Cleaning text can also change harmless Unicode or encoded content, so I will check false positives after cleaning. Long prompts may be cut off by a transformer’s input limit. I will record how often this happens before and after changing the text, since it could also explain missed attacks.

### 6. Proposed split

**Preparing the split.** I will combine xTRam1’s original training and test files and use this dataset as the only source for training. Cleaning starts by removing both groups with conflicting labels, then keeping one copy of each remaining exact duplicate. Next, xTRam1 prompts that exactly or closely match deepset or Gandalf are removed. I will split the remaining data into about 80% training and 20% validation, keeping similar prompts together and class proportions as close as possible. The split uses seed 4371 so it can be repeated. Since there are no timestamps or user IDs, the results will describe performance across these datasets, not performance on future traffic or new users.

**Training and validation.** The first model will use word and character TF-IDF features with logistic regression. The second will be a fine-tuned DistilBERT model [5]. The TF-IDF vocabulary and weights come from training data only. Validation data guide the choice of model settings and when to stop training. For each detector, I will choose the threshold with the highest validation recall while keeping the validation false-positive rate at or below 5%. That threshold stays fixed for all tests; the false-positive rate may still be higher on another dataset. I will use ProtectAI’s model without retraining its weights [4].

**Testing the text changes.** All 662 deepset rows and 1,000 Gandalf rows will be used for external testing. Their text is checked for duplicates, but I will not use their test scores to tune the models, thresholds, or cleaning rules. Each test injection has three separate versions: one inserts the invisible U+200B character between letters within words, one replaces selected Latin letters with look-alike Unicode characters using a fixed map, and one encodes the complete UTF-8 prompt in Base64. Every model will receive the same versions. Benign prompts will not receive these attack changes.

Normalization removes U+200B, maps the look-alike characters back to Latin letters, and decodes a whole prompt only when it is valid Base64 and produces valid UTF-8 text. A final check confirms that re-encoding the decoded text returns the same Base64 string. I will set these rules before testing. The comparison covers the original prompts, changed prompts, and normalized versions of both. Applying the same cleaning to benign prompts will show whether it introduces more false alarms.

Results for deepset will be reported overall and separately for English and German, including group sizes and class counts. Language is assigned from each original prompt and stays fixed for its changed and normalized versions. Other or uncertain cases will be reported separately.

**Results and next step.** The deepset metrics are precision, recall, F1, area under the precision-recall curve, and false-positive rate. Gandalf results report recall with the label assumption explained in Section 4. Detector evasion rate is calculated for each text change: out of the attacks caught in their original form, how many are missed after the change? Recall drop and recovery are reported in percentage points. My next step is to finish the near-duplicate checks and run the TF-IDF baseline before training DistilBERT. I will save the dataset versions, split indices, and model settings with a notebook, tables, and plots so the experiment can be repeated.

### Sources

[1] xTRam1. Safe-Guard Prompt Injection. Dataset card and data files, version a3a877d6. Hugging Face dataset snapshot.

[2] deepset. Prompt Injections. Dataset card and data files, version 4f61ecb0. Hugging Face dataset snapshot.

[3] Lakera. Gandalf Ignore Instructions. Dataset card and data files, version 04737b65. Hugging Face dataset snapshot.

[4] Protect AI. DeBERTa-v3-base-prompt-injection-v2. Hugging Face model card.

[5] Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.

[6] langdetect. Python language detection library, version 1.0.9. Project documentation.

## Assigned variant

Project assignment: your project is approved as a variant of Project 15. What varies: the language of the test prompt. Machine-translate the Safe-Guard and Gandalf test injections into German and French, and use deepset's native German prompts (the 40% your audit found) as the check on translation. What stays fixed: you train on English Safe-Guard. Your main table shows the score by language, with translated German next to native German.

## Current experiment

**Question.** How well does a prompt-injection detector trained on English Safe-Guard
prompts detect injections written in German and French, and how do results on
machine-translated German compare with results on native German prompts?

**Variable:** the language of the test prompt (English, machine-translated German,
native German, machine-translated French). **Fixed:** the English Safe-Guard training
split, the classifier, and the threshold rule.

**Language review.** langdetect mislabels many short English prompts, so it is not used
to choose rows. `language_review.py` flags every prompt that might not be English, and
each flagged prompt was checked by hand. Deepset prompts are labeled English, German,
mixed (English and German in one prompt), or other. The labels and the reason for every
manual decision are in `language_review.csv`.

**Split.**
1. Safe-Guard, the only training source: drop the 11 prompts that are not English, then
   the 5 rows in label-conflict groups, then 156 extra copies of exact duplicates, then
   13 prompts that match a deepset or Gandalf prompt (exact, normalized, or character
   n-gram cosine >= 0.9). That leaves 10,111 rows.
2. Group Safe-Guard near-duplicates (same normalized text, or character n-gram cosine
   >= 0.9) and split 70/15/15 into train, validation and test with
   `StratifiedGroupKFold` (20 folds, seed 4371; folds 0-2 test, 3-5 validation). No
   group crosses a split.
3. Test sets: Safe-Guard test (English), the 994 English Gandalf prompts, and deepset
   split into English, native German, mixed, and other-language prompts. All deepset
   rows are test rows.

The threshold is chosen on Safe-Guard validation only: the lowest 4-decimal threshold
with a false-positive rate of at most 5%.

## Running the code

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_baseline.py
```

`language_review.py` rebuilds `language_review.csv` from the flags and the recorded
manual decisions; you only need it if you change the review. `run_baseline.py` downloads the three datasets from Hugging Face at the pinned revisions below into `data/`, which is not committed. It then audits the data, builds the split, trains the TF-IDF with logistic regression baseline, and writes `split.csv` and `scores.csv`. A second run writes byte-identical files.

| Dataset | Revision |
|---|---|
| [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | a3a877d608f37b7d20d9945671902df895ecdb46 |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 4f61ecb038e9c3fb77e21034b22511b523772cdd |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | 04737b65e90a6794ec227012e4a255a7def6344b |

To check the outputs, download `check_checkpoint.py` from Canvas into this folder and run:

```bash
.venv/bin/python check_checkpoint.py --split split.csv --scores scores.csv --threshold 0.0384
```

| Output | Contents |
|---|---|
| `language_review.csv` | Reviewed language for every non-English Safe-Guard or Gandalf prompt and every deepset prompt, with the reason for manual decisions |
| `split.csv` | Every kept row: row_id, dataset, split, label, lang (reviewed), langdetect, orig_split, near_dup_group |
| `scores.csv` | Every test row: row_id, test_set, label, score |
| `outputs/audit.json` | Audit numbers |
| `outputs/split_log.json` | Rows removed at each step, and leakage counts |
| `outputs/run_info.json` | Threshold, chosen C, validation grid, features, timing, versions |
| `outputs/checker_output.txt` | Output of the course checker |
| `outputs/pr_curve_safeguard_test.png` | Precision-recall curve on the Safe-Guard test set |
