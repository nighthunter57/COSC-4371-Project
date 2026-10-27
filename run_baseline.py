"""COSC 4371 Project 15 (language variant): midsemester baseline.

One command reproduces everything:

    python run_baseline.py

It downloads the three datasets at pinned versions, audits them, builds the split,
trains TF-IDF + logistic regression on English Safe-Guard, picks the threshold on
validation, scores every test row, and writes:

Language labels come from language_review.csv (see language_review.py). Safe-Guard and
Gandalf prompts are English unless that file says otherwise.


    split.csv, scores.csv                 (checked by check_checkpoint.py)
    outputs/audit.json                    (audit table numbers, section 2(b))
    outputs/split_log.json                (removals and leakage counts, 2(c)-(d))
    outputs/run_info.json                 (threshold, settings, timing, versions, 3(a)-(b))
    outputs/pr_curve_safeguard_test.png   (figure, section 4)
"""

import json
import platform
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

SEED = 4371
MAX_VAL_FPR = 0.05
NEAR_DUP_COSINE = 0.90           # char 3-5 gram TF-IDF cosine at or above this = near duplicate
C_GRID = [0.25, 1.0, 4.0, 16.0]  # logistic regression C, chosen on validation
N_FOLDS = 20                     # 20 grouped folds: 3 test, 3 validation, 14 train = 15/15/70
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}

DATASETS = {
    "safeguard": ("xTRam1/safe-guard-prompt-injection", "a3a877d608f37b7d20d9945671902df895ecdb46"),
    "deepset": ("deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd"),
    "gandalf": ("Lakera/gandalf_ignore_instructions", "04737b65e90a6794ec227012e4a255a7def6344b"),
}

ROOT = Path(__file__).resolve().parent
LANG_REVIEW = "language_review.csv"
DATA = ROOT / "data"
OUT = ROOT / "outputs"


# ---------------------------------------------------------------- loading

def load(name):
    repo, rev = DATASETS[name]
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                   if f.endswith(".parquet"))
    parts = []
    for f in files:
        path = hf_hub_download(repo, f, repo_type="dataset", revision=rev, cache_dir=DATA)
        part = pd.read_parquet(path)
        part["orig_split"] = f.split("/")[-1].split("-")[0]
        part["orig_index"] = np.arange(len(part))
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = name + ":" + df["orig_split"] + ":" + df["orig_index"].astype(str)
    df["dataset"] = name
    if name == "gandalf":
        df["label"] = 1  # every Gandalf prompt is an intended injection (proposal section 4)
    df["label"] = df["label"].astype(int)
    return df


def norm_key(text):
    """Normalized text: NFKC, casefold, punctuation and whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", t).split())


def reviewed_language(df):
    """Reviewed language per row: the label in language_review.csv, else English."""
    review = pd.read_csv(ROOT / LANG_REVIEW).set_index("row_id")["language"]
    lang = df["row_id"].map(review)
    if df["dataset"].iloc[0] == "deepset":
        assert lang.notna().all(), "every deepset row needs a reviewed language"
    return lang.fillna("en")


def detect_lang(texts):
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0

    def one(t):
        try:
            return detect(t)
        except Exception:  # noqa: BLE001  (empty or symbol-only text)
            return "unknown"
    return [one(t) for t in texts]


# ---------------------------------------------------------------- audit

def audit(df, primary, near_groups=None):
    is_gandalf = df["dataset"].iloc[0] == "gandalf"
    out = {"rows": len(df),
           "columns": ["text", "similarity"] if is_gandalf else ["text", "label"]}
    if is_gandalf:
        out["classes"] = "no label column; all rows treated as injections"
        out["missing_target"] = "label not supplied"
        out["missing_similarity"] = int(df["similarity"].isna().sum())
    else:
        vc = df["label"].value_counts()
        out["classes"] = {str(k): {"count": int(vc.get(k, 0)),
                                   "pct": round(100 * vc.get(k, 0) / len(df), 2)} for k in (0, 1)}
        out["missing_target"] = int(df["label"].isna().sum())
        out["label_conflict_groups"] = int((df.groupby("text")["label"].nunique() > 1).sum())
    out["missing_text"] = int(df["text"].isna().sum())
    out["empty_text"] = int((df["text"].str.strip() == "").sum())
    dup = df.groupby("text").size()
    out["exact_duplicates"] = {"rule": "identical full text field",
                               "groups": int((dup > 1).sum()),
                               "extra_rows": int((dup - 1).sum())}
    out["language_counts_reviewed"] = {k: int(v) for k, v in df["lang"].value_counts().items()}
    out["langdetect_non_english_tags"] = int((df["langdetect"] != "en").sum())
    if primary:
        sizes = pd.Series(near_groups).value_counts()
        out["near_duplicates"] = {
            "rule": f"same normalized text (NFKC, casefold, punctuation and whitespace collapsed) "
                    f"or char_wb 3-5 gram TF-IDF cosine >= {NEAR_DUP_COSINE}, linked into "
                    f"connected components (computed after exact-duplicate removal)",
            "groups_with_2plus_rows": int((sizes > 1).sum()),
            "rows_in_those_groups": int(sizes[sizes > 1].sum()),
            "extra_rows": int((sizes - 1).sum()),
            "largest_group": int(sizes.max()),
        }
        out["repeated_entities"] = "unavailable: no user, session or campaign ID field"
        out["time_range"] = "unavailable: no timestamp field"
    return out


def near_dup_groups(texts):
    """Connected components linking two prompts when their normalized text is identical
    or their char_wb 3-5 gram TF-IDF cosine is at least NEAR_DUP_COSINE."""
    texts = list(texts)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    S = (X @ X.T).tocoo()
    keep = S.data >= NEAR_DUP_COSINE
    rows, cols = list(S.row[keep]), list(S.col[keep])
    first = {}
    for i, k in enumerate(map(norm_key, texts)):  # same normalized text: link to first seen
        j = first.setdefault(k, i)
        if j != i:
            rows.append(i)
            cols.append(j)
    n = len(texts)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(A, directed=False)[1]


def cross_near_matches(sg_texts, ext_texts):
    """For each Safe-Guard text, True if it is a near duplicate of any external test text."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    vec.fit(list(sg_texts) + list(ext_texts))
    S = vec.transform(sg_texts) @ vec.transform(ext_texts).T
    return np.asarray(S.max(axis=1).todense()).ravel() >= NEAR_DUP_COSINE


# ---------------------------------------------------------------- split

def build_split(sg, deepset, gandalf):
    log = {"safeguard_rows_loaded": len(sg)}

    # 0. training is English only: drop Safe-Guard prompts the language review marks
    #    as non-English or mixed, and keep only English Gandalf prompts for testing
    non_en = sg["lang"] != "en"
    log["removed_non_english_safeguard_rows"] = int(non_en.sum())
    sg = sg[~non_en]
    g_non_en = gandalf["lang"] != "en"
    log["removed_non_english_gandalf_rows"] = int(g_non_en.sum())
    ext_all = pd.concat([deepset, gandalf])   # every external row, for match removal
    gandalf = gandalf[~g_non_en]

    # 1. drop every row of a text group that carries both labels
    conflict = sg.groupby("text")["label"].transform("nunique") > 1
    log["removed_label_conflict_rows"] = int(conflict.sum())
    sg = sg[~conflict]

    # 2. keep one copy of each exact duplicate (first by original file order)
    dup = sg.duplicated("text", keep="first")
    log["removed_exact_duplicate_rows"] = int(dup.sum())
    sg = sg[~dup].reset_index(drop=True)

    # 3. remove Safe-Guard rows that match any external row (exact, normalized or near)
    ext = ext_all
    exact = sg["text"].isin(set(ext["text"]))
    normed = sg["key"].isin(set(ext["key"])) & ~exact
    near = cross_near_matches(sg["text"], ext["text"]) & ~exact & ~normed
    log["removed_matching_external_test"] = {"exact": int(exact.sum()),
                                             "normalized_only": int(normed.sum()),
                                             "near_only": int(near.sum())}
    sg = sg[~(exact | normed | near)].reset_index(drop=True)

    # 4. near-duplicate groups, then a grouped, stratified 70/15/15 split
    sg["group"] = near_dup_groups(sg["text"])
    folds = np.empty(len(sg), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(sgkf.split(sg, sg["label"], sg["group"])):
        folds[idx] = k
    sg["split"] = np.where(np.isin(folds, list(TEST_FOLDS)), "test",
                           np.where(np.isin(folds, list(VAL_FOLDS)), "validation", "train"))
    log["near_dup_groups_crossing_splits"] = int(sg.groupby("group")["split"].nunique().gt(1).sum())
    sizes = sg["group"].value_counts()
    log["near_dup_rows_kept_together"] = int(sizes[sizes > 1].sum())
    log["safeguard_rows_kept"] = len(sg)

    deepset = deepset.assign(split="test", group=-1)
    gandalf = gandalf.assign(split="test", group=-1)
    return sg, deepset, gandalf, log


def leakage_counts(sg, ext):
    train = sg[sg["split"] == "train"]
    tests = pd.concat([sg[sg["split"] == "test"], ext])
    val = sg[sg["split"] == "validation"]
    return {
        "train_rows_exactly_duplicating_any_test_row": int(train["text"].isin(set(tests["text"])).sum()),
        "train_rows_exactly_duplicating_a_validation_row": int(train["text"].isin(set(val["text"])).sum()),
        "train_rows_matching_any_test_row_after_normalization": int(train["key"].isin(set(tests["key"])).sum()),
        "train_rows_matching_a_validation_row_after_normalization": int(train["key"].isin(set(val["key"])).sum()),
    }


# ---------------------------------------------------------------- model

def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, lowercase=True)),
    ])


def pick_threshold(y_val, s_val):
    """Lowest 4-decimal threshold whose validation FPR (score >= t) is at most MAX_VAL_FPR.
    The lowest such threshold gives the highest validation recall under the FPR cap."""
    grid = np.round(np.arange(0, 10001) / 10000, 4)
    neg = np.sort(s_val[y_val == 0])
    fp = len(neg) - np.searchsorted(neg, grid, side="left")  # negatives with score >= t
    ok = np.where(fp / len(neg) <= MAX_VAL_FPR)[0]
    t = float(grid[ok[0]])
    pred = s_val >= t
    return t, float(pred[y_val == 1].mean()), float(pred[y_val == 0].mean())


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    sg, deepset, gandalf = (load(n) for n in ("safeguard", "deepset", "gandalf"))
    for df in (sg, deepset, gandalf):
        df["key"] = df["text"].map(norm_key)
        df["langdetect"] = detect_lang(df["text"])
        df["lang"] = reviewed_language(df)

    audits = {
        "safeguard": audit(sg, primary=True,
                           near_groups=near_dup_groups(sg.drop_duplicates("text")["text"])),
        "deepset": audit(deepset, primary=False),
        "gandalf": audit(gandalf, primary=False),
    }
    sg_u = sg.drop_duplicates("text")
    audits["cross_dataset_exact_overlap"] = {
        "safeguard_deepset_texts": len(set(sg["text"]) & set(deepset["text"])),
        "safeguard_deepset_label_disagreements": int(
            sg_u.merge(deepset, on="text").query("label_x != label_y").shape[0]),
        "safeguard_gandalf_texts": len(set(sg["text"]) & set(gandalf["text"])),
        "deepset_gandalf_texts": len(set(deepset["text"]) & set(gandalf["text"])),
    }
    audits["deepset_by_language"] = {
        g: {"rows": len(d), "injection": int(d["label"].sum()), "benign": int((d["label"] == 0).sum())}
        for g, d in deepset.groupby("lang")}

    sg_split, deepset, gandalf, log = build_split(sg, deepset, gandalf)
    log["leakage"] = leakage_counts(sg_split, pd.concat([deepset, gandalf]))

    # ---- train, choosing C on validation
    train = sg_split[sg_split["split"] == "train"]
    val = sg_split[sg_split["split"] == "validation"]
    t_fit = time.time()
    feats = make_features().fit(train["text"])
    Xtr, Xva = feats.transform(train["text"]), feats.transform(val["text"])
    yva = val["label"].to_numpy()
    tried = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
        clf.fit(Xtr, train["label"])
        s = clf.predict_proba(Xva)[:, 1]
        thr, rec, fpr = pick_threshold(yva, s)
        tried.append({"C": C, "threshold": thr, "val_recall": rec, "val_fpr": fpr,
                      "val_auroc": float(roc_auc_score(yva, s)), "model": clf})
    best = max(tried, key=lambda r: (r["val_recall"], r["val_auroc"]))
    clf, threshold = best["model"], best["threshold"]
    fit_seconds = time.time() - t_fit

    # ---- score every test row once
    test_sets = [
        ("safeguard_test_en", sg_split[sg_split["split"] == "test"]),
        ("gandalf_en", gandalf),
        ("deepset_en", deepset[deepset["lang"] == "en"]),
        ("deepset_de_native", deepset[deepset["lang"] == "de"]),
        ("deepset_mixed_en_de", deepset[deepset["lang"] == "mixed"]),
        ("deepset_other_lang", deepset[deepset["lang"] == "other"]),
    ]
    scores = pd.concat([
        pd.DataFrame({"row_id": d["row_id"], "test_set": name, "label": d["label"],
                      "score": clf.predict_proba(feats.transform(d["text"]))[:, 1]})
        for name, d in test_sets], ignore_index=True)

    cols = ["row_id", "dataset", "split", "label", "lang", "langdetect", "orig_split", "group"]
    split = pd.concat([sg_split[cols], deepset[cols], gandalf[cols]], ignore_index=True)
    split.rename(columns={"group": "near_dup_group"}).to_csv(ROOT / "split.csv", index=False)
    scores.to_csv(ROOT / "scores.csv", index=False, float_format="%.6f")

    # ---- figure: precision-recall on the Safe-Guard test set
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = scores[scores["test_set"] == "safeguard_test_en"]
    p, r, _ = precision_recall_curve(st["label"], st["score"])
    pred = st["score"] >= threshold
    op_r = float(pred[st["label"] == 1].mean())
    op_p = float(st["label"][pred].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, color="#2563eb", lw=2, label="TF-IDF + logistic regression")
    ax.axhline(st["label"].mean(), color="#9ca3af", ls="--", lw=1, label="random scorer")
    ax.scatter([op_r], [op_p], color="#dc2626", zorder=3, label=f"threshold {threshold:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_title(f"Safe-Guard test (English): {len(st)} rows, "
                 f"{100 * st['label'].mean():.1f}% injections", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve_safeguard_test.png", dpi=200)

    import sklearn
    info = {
        "threshold": threshold,
        "threshold_rule": f"lowest 4-decimal threshold with validation FPR <= {MAX_VAL_FPR} "
                          f"(score >= threshold is positive)",
        "chosen_C": best["C"],
        "validation_grid": [{k: v for k, v in r.items() if k != "model"} for r in tried],
        "features": {"word": "TF-IDF word 1-2 grams, min_df 2, sublinear tf, lowercase",
                     "char": "TF-IDF char_wb 3-5 grams, min_df 2, sublinear tf, lowercase",
                     "n_features": int(Xtr.shape[1])},
        "classifier": "LogisticRegression(class_weight='balanced', max_iter=5000, lbfgs)",
        "fit_seconds_all_C": round(fit_seconds, 1),
        "total_seconds": round(time.time() - t0, 1),
        "hardware": f"{platform.machine()} CPU, {platform.platform()}",
        "python": platform.python_version(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
        "seed": SEED,
        "dataset_revisions": {k: v[1] for k, v in DATASETS.items()},
        "figure_operating_point": {"recall": op_r, "precision": op_p},
    }
    (OUT / "audit.json").write_text(json.dumps(audits, indent=2, default=int))
    (OUT / "split_log.json").write_text(json.dumps(log, indent=2, default=int))
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps({"split_log": log, "threshold": threshold, "C": best["C"],
                      "fit_seconds": info["fit_seconds_all_C"]}, indent=2, default=int))
    print(f"\nNow run:\n  python check_checkpoint.py --split split.csv --scores scores.csv "
          f"--threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
