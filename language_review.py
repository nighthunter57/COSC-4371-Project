"""Language review: writes language_review.csv, which run_baseline.py reads.

Every prompt gets one language label: en, de, mixed (English and German together), or
other (any other language, alone or combined with English or German).

How the labels were set
-----------------------
Safe-Guard and Gandalf are English datasets, so a prompt is English unless the manual
review below says otherwise. The review covered every candidate from these flags:
  1. langdetect (seed 0) tags the whole prompt as something other than English;
  2. a sentence or line of at least 6 words is detected as non-English with p >= 0.95;
  3. at least 20% of its letters are outside the Latin script;
  4. it contains 3 or more accented Latin letters.
langdetect alone is not used, because it mislabels many short English prompts.

Deepset mixes English and German, often inside one prompt. Each deepset prompt is first
labeled by a word rule (English and German function words, plus umlauts), then every
prompt where the rule is unsure, finds both languages, or disagrees with langdetect was
reviewed by hand. MANUAL holds those decisions.

Run:  python language_review.py   (also writes language_candidates.csv, not committed,
                                   which includes prompt text for reviewing)
"""
import re
import unicodedata

import pandas as pd
from langdetect import DetectorFactory, detect_langs
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import run_baseline as rb

DetectorFactory.seed = 0

# ---------------------------------------------------------------- manual decisions

SG_NON_ENGLISH = {
    "safeguard:train:7900": ("de", "jailbreak written entirely in German"),
    "safeguard:train:2893": ("other", "English jailbreak with several Italian sentences"),
    "safeguard:train:3028": ("other", "English jailbreak with several Italian sentences"),
    "safeguard:train:3626": ("other", "product review written in Spanish"),
    "safeguard:train:1964": ("other", "asks to translate a French tweet; contains the French text"),
    "safeguard:train:4009": ("other", "asks to translate a French headline; contains the French text"),
    "safeguard:train:4232": ("other", "asks to translate an Arabic headline; contains the Arabic text"),
    "safeguard:train:2619": ("other", "translation task containing a Russian sentence"),
    "safeguard:train:2835": ("other", "rewrite task containing Russian text"),
    "safeguard:train:3185": ("other", "language-identification task containing Chinese text"),
    "safeguard:test:1087": ("other", "translation task containing Chinese text"),
}

GANDALF_NON_ENGLISH = {
    "gandalf:test:83": ("other", "written in Russian"),
    "gandalf:train:95": ("de", "written in German"),
    "gandalf:train:359": ("de", "written in German"),
    "gandalf:train:452": ("de", "written in German"),
    "gandalf:validation:48": ("de", "written in German"),
    "gandalf:train:682": ("other", "English sentence followed by a Spanish sentence"),
}

_EN = ("test:2 test:9 test:10 test:18 test:19 test:21 test:25 test:26 test:28 test:75 "
       "train:0 train:9 train:12 train:13 train:15 train:16 train:23 train:25 train:28 "
       "train:30 train:34 train:40 train:46 train:50 train:60 train:65 train:67 train:73 "
       "train:75 train:79 train:81 train:90 train:91 train:92 train:95 train:96 train:99 "
       "train:100 train:104 train:106 train:107 train:108 train:113 train:115 train:131 "
       "train:136 train:139 train:151 train:157 train:161 train:163 train:171 train:368 "
       "train:388 train:408 train:413 train:421 train:426 train:456 train:476 train:484")
_DE = ("test:39 test:46 test:47 test:48 test:55 test:56 test:58 test:62 test:63 "
       "train:189 train:192 train:193 train:195 train:196 train:203 train:205 train:208 "
       "train:210 train:214 train:220 train:226 train:230 train:240 train:243 train:245 "
       "train:252 train:253 train:259 train:261 train:263 train:270 train:272 train:275 "
       "train:279 train:280 train:282 train:286 train:288 train:292 train:293 train:294 "
       "train:295 train:311 train:313 train:316 train:319 train:327 train:331 train:337 "
       "train:339 train:340 train:341 train:343 train:351 train:356")
_OTHER = {
    "test:80": "Spanish", "test:89": "Spanish", "train:365": "Spanish",
    "train:389": "Spanish", "train:414": "Spanish", "train:477": "Spanish",
    "train:406": "Croatian", "train:453": "Croatian", "train:427": "French",
    "train:433": "single word, not English or German", "train:466": "Russian with English",
    "train:479": "Khmer", "train:446": "romanized Hindi",
    "train:504": "German question with a Croatian injection",
    "train:509": "English question with a French injection",
}
_MIXED = {
    "test:107": "German keywords followed by English", "train:491": "German keywords followed by English",
    "train:496": "English keywords followed by German", "train:506": "German keywords followed by English",
    "train:511": "English keywords followed by German", "train:522": "German keywords followed by English",
    "train:525": "English fragment followed by German", "train:536": "English keywords followed by German",
    "train:377": "English with a German sentence in the payload",
    "train:488": "English with a German sentence in the payload",
}
_FIXES = {  # rule or langdetect got these wrong
    "test:1": ("en", "English; langdetect said Italian"), "test:76": ("en", "English; langdetect said Dutch"),
    "test:87": ("en", "English with one German word"), "train:422": ("en", "English with spaced-out letters"),
    "train:464": ("en", "English; langdetect said Tagalog"), "train:173": ("en", "English; only a German surname"),
    "train:276": ("de", "German; langdetect said Dutch"), "train:312": ("de", "German; langdetect said Afrikaans"),
    "train:317": ("de", "German; langdetect said Afrikaans"), "train:474": ("de", "German; langdetect said Afrikaans"),
}

DEEPSET_MANUAL = {}
for k in _EN.split():
    DEEPSET_MANUAL["deepset:" + k] = ("en", "short English prompt; rule unsure")
for k in _DE.split():
    DEEPSET_MANUAL["deepset:" + k] = ("de", "short German prompt; rule unsure")
for k, note in _OTHER.items():
    DEEPSET_MANUAL["deepset:" + k] = ("other", note)
for k, note in _MIXED.items():
    DEEPSET_MANUAL["deepset:" + k] = ("mixed", note)
for k, v in _FIXES.items():
    DEEPSET_MANUAL["deepset:" + k] = v

# ---------------------------------------------------------------- deepset word rule

_AMBIGUOUS = {"was", "in", "an", "so", "also", "will", "am", "man", "hat", "bin", "um", "war",
              "die", "de", "me", "mine", "hers", "on"}
EN_WORDS = set(ENGLISH_STOP_WORDS) - _AMBIGUOUS
DE_WORDS = set("""der die das und ist nicht ich du sie wir ihr ein eine einen einem einer zu mit
von für auf den dem des sich wie wer warum wo bitte alle alles sind wird werden kann können auch
oder aber nach bei über aus noch nur mein dein gibt sehr hat haben bin mir mich dir dich im vom
zum zur uns welche welcher welches vergiss schreibe sage sag gesagte jetzt heute gerne würde
eines diese dieser man wurde sein ihre ihren ob wenn dass bist habe viel viele""".split())


def word_rule(text):
    words = re.findall(r"[a-zA-ZäöüÄÖÜß']+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    de = sum(w in DE_WORDS for w in words) + len(re.findall(r"[äöüß]", text.lower()))
    if en >= 2 and de >= 2:
        return "mixed"
    if en >= 2:
        return "en"
    if de >= 2:
        return "de"
    return "unsure"


# ---------------------------------------------------------------- candidate flags

def _top(text):
    try:
        best = detect_langs(text)[0]
        return best.lang, best.prob
    except Exception:  # noqa: BLE001
        return "unknown", 0.0


def _non_latin_share(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("LATIN" not in unicodedata.name(c, "") for c in letters) / len(letters)


def _foreign_segment(text):
    for seg in re.split(r"[\n.!?]+", text):
        if len(seg.split()) >= 6:
            lang, prob = _top(seg)
            if lang != "en" and prob >= 0.95:
                return f"{lang}:{seg.strip()[:60]}"
    return ""


def main():
    candidates, review = [], []
    for name in ("safeguard", "deepset", "gandalf"):
        df = rb.load(name)
        manual = {"safeguard": SG_NON_ENGLISH, "gandalf": GANDALF_NON_ENGLISH,
                  "deepset": DEEPSET_MANUAL}[name]
        missing = set(manual) - set(df["row_id"])
        assert not missing, f"unknown row ids in {name}: {sorted(missing)[:5]}"
        for rid, text in zip(df["row_id"], df["text"]):
            lang, prob = _top(text)
            seg, nl = _foreign_segment(text), _non_latin_share(text)
            accents = len(re.findall(r"[à-ÿÀ-ßœŒ]", text))
            if lang != "en" or seg or nl >= 0.2 or accents >= 3:
                candidates.append({"row_id": rid, "dataset": name, "langdetect": lang,
                                   "prob": round(prob, 3), "foreign_segment": seg,
                                   "non_latin_share": round(nl, 2), "accents": accents,
                                   "text": text})
            if rid in manual:
                review.append({"row_id": rid, "dataset": name, "language": manual[rid][0],
                               "method": "manual", "note": manual[rid][1]})
            elif name == "deepset":
                rule = word_rule(text)
                assert rule != "unsure", f"{rid} needs a manual label"
                review.append({"row_id": rid, "dataset": name, "language": rule,
                               "method": "rule", "note": ""})
    pd.DataFrame(candidates).to_csv("language_candidates.csv", index=False)
    out = pd.DataFrame(review)
    out.to_csv("language_review.csv", index=False)
    print(out.groupby(["dataset", "language", "method"]).size().to_string())


if __name__ == "__main__":
    main()
