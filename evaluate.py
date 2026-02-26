"""
evaluate.py — Score system outputs using F1, EM, and ROUGE-L.
Mirrors the metrics described in the assignment spec (SQuAD-style).

Usage:
    python src/evaluate.py --pred system_outputs/system_output_1.json \
                           --ref data/train/reference_answers.json
"""

import json
import re
import string
import argparse
from collections import Counter
from rouge_score import rouge_scorer


def normalize(text: str) -> str:
    text = text.lower()
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def best_score(pred: str, golds: list[str], metric_fn) -> float:
    return max(metric_fn(pred, g) for g in golds)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, gold: str) -> float:
    return _rouge.score(gold, pred)["rougeL"].fmeasure


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(pred_path: str, ref_path: str):
    with open(pred_path) as f:
        predictions: dict[str, str] = json.load(f)

    with open(ref_path) as f:
        references: dict[str, list[str]] = json.load(f)
        # Support both {"1": "answer"} and {"1": ["answer1", "answer2"]}
        references = {
            k: v if isinstance(v, list) else [v]
            for k, v in references.items()
        }

    f1s, ems, rls = [], [], []
    for qid, pred in predictions.items():
        if qid not in references:
            print(f"[WARN] qid {qid} not in references, skipping.")
            continue
        golds = references[qid]
        f1s.append(best_score(pred, golds, f1_score))
        ems.append(float(best_score(pred, golds, lambda p, g: float(exact_match(p, g)))))
        rls.append(best_score(pred, golds, rouge_l))

    print(f"\nEvaluation over {len(f1s)} questions:")
    print(f"  F1:      {100 * sum(f1s) / len(f1s):.2f}")
    print(f"  EM:      {100 * sum(ems) / len(ems):.2f}")
    print(f"  ROUGE-L: {100 * sum(rls) / len(rls):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to predictions JSON")
    parser.add_argument("--ref", required=True, help="Path to reference answers JSON")
    args = parser.parse_args()
    evaluate(args.pred, args.ref)
