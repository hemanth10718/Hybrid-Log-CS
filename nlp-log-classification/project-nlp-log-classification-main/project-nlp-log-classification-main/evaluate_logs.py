"""Evaluate the hybrid log classifier.

Run from the project folder (the one containing classify.py):
    python evaluate_logs.py
    python evaluate_logs.py --with-llm      # also sends the 7 LegacyCRM rows to the LLM

What it does
  1. Rebuilds the notebook's held-out split (30%, random_state=42) over the rows the
     classifier was trained on (complexity == "bert") and scores the saved model on it.
  2. Runs the FULL pipeline (regex, then classifier with the 50% rule) on the same rows.
  3. Runs the pipeline on the regex-routed rows.
  4. Optionally runs the LegacyCRM rows through the LLM (uses Groq quota).

The dataset is synthetic, so report any number with that qualifier.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Class counts printed by the notebook for its 571-row test set
NOTEBOOK_SUPPORT = {
    "Critical Error": 48,
    "Error": 47,
    "HTTP Status": 304,
    "Resource Usage": 49,
    "Security Alert": 123,
}


def compute_metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    out = {"accuracy": accuracy_score(y_true, y_pred), "labels": labels}
    for avg in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[avg] = {"precision": p, "recall": r, "f1": f1}
    return out


def print_block(title, y_true, y_pred, show_matrix=True):
    y_true, y_pred = list(y_true), list(y_pred)
    m = compute_metrics(y_true, y_pred)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Rows: {len(y_true)}   Accuracy: {m['accuracy']:.3f}")
    for avg in ("macro", "weighted"):
        v = m[avg]
        print(f"{avg:>8}: precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}")
    print(f"Predicted 'Unclassified': {np.mean([p == 'Unclassified' for p in y_pred]):.1%}")
    print()
    print(classification_report(y_true, y_pred, labels=m["labels"], zero_division=0))
    if show_matrix:
        print("Confusion matrix (rows = true, columns = predicted):")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=m["labels"]),
                           index=m["labels"], columns=m["labels"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/dataset/synthetic_logs.csv")
    ap.add_argument("--model", default="models/log_classifier.joblib")
    ap.add_argument("--with-llm", action="store_true",
                    help="also run LegacyCRM rows through the LLM (uses Groq quota)")
    args = ap.parse_args()

    pd.set_option("display.max_columns", None)  # show the whole confusion matrix
    pd.set_option("display.width", 250)

    import joblib
    from sentence_transformers import SentenceTransformer
    from classify import classify_log
    from processor_regex import classify_with_regex

    df = pd.read_csv(args.data)
    bert_df = df[df["complexity"] == "bert"].reset_index(drop=True)
    regex_df = df[df["complexity"] == "regex"].reset_index(drop=True)
    llm_df = df[df["complexity"] == "llm"].reset_index(drop=True)
    print(f"Dataset rows: {len(df)}  (bert {len(bert_df)}, regex {len(regex_df)}, llm {len(llm_df)})")
    print(f"Notebook trained on 1903 rows; bert rows here: {len(bert_df)}")

    clf = joblib.load(args.model)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding the bert rows (this takes a little while)...")
    X = embedder.encode(bert_df["log_message"].tolist(), show_progress_bar=False)
    y = bert_df["target_label"].values
    idx = np.arange(len(bert_df))
    _, X_test, _, y_test, _, idx_test = train_test_split(
        X, y, idx, test_size=0.3, random_state=42
    )

    # 1. classifier only (should reproduce the notebook report)
    print_block("1. Saved classifier on the held-out 30% (notebook split)",
                y_test, clf.predict(X_test))
    got = pd.Series(y_test).value_counts().to_dict()
    if got == NOTEBOOK_SUPPORT:
        print("\nSplit check: OK, the class counts match the notebook's report.")
    else:
        print("\nWARNING: the class counts differ from the notebook's report.")
        print("  notebook:", NOTEBOOK_SUPPORT)
        print("  here:    ", got)
        print("  The notebook probably filtered the rows differently. Send me the cell that")
        print("  defines embeddings_filtered before trusting these numbers.")

    # 2. full pipeline on the same held-out rows
    test_rows = bert_df.iloc[idx_test].reset_index(drop=True)
    preds = [classify_log(s, m) for s, m in zip(test_rows["source"], test_rows["log_message"])]
    intercepted = sum(classify_with_regex(m) is not None for m in test_rows["log_message"])
    print_block("2. Full pipeline (regex, then classifier, 50% rule) on the same held-out rows",
                test_rows["target_label"], preds)
    print(f"\nHeld-out rows that regex intercepted before the classifier: {intercepted}")

    # 3. regex-routed rows
    rpreds = [classify_log(s, m) for s, m in zip(regex_df["source"], regex_df["log_message"])]
    rmatch = [classify_with_regex(m) for m in regex_df["log_message"]]
    matched = np.array([x is not None for x in rmatch])
    print_block("3. Pipeline on the regex-routed rows", regex_df["target_label"], rpreds)
    print(f"\nRegex coverage on these rows: {matched.mean():.1%}")
    if matched.any():
        acc = np.mean([t == p for t, p, k in zip(regex_df["target_label"], rmatch, matched) if k])
        print(f"Accuracy among the rows regex did match: {acc:.1%}")
    print("Rows regex missed fall through to the classifier, which has no "
          "'User Action' or 'System Notification' class.")

    # 4. LLM rows
    llm_preds = []
    if args.with_llm and len(llm_df):
        llm_preds = [classify_log(s, m) for s, m in zip(llm_df["source"], llm_df["log_message"])]
        print_block(f"4. LLM path on the {len(llm_df)} LegacyCRM rows (too few to be a real metric)",
                    llm_df["target_label"], llm_preds, show_matrix=False)
    else:
        print("\n4. LLM path skipped (add --with-llm to run the LegacyCRM rows).")

    parts = [
        test_rows.assign(split="heldout_bert", predicted=preds),
        regex_df.assign(split="regex", predicted=rpreds),
    ]
    if llm_preds:
        parts.append(llm_df.assign(split="llm", predicted=llm_preds))
    out = pd.concat(parts, ignore_index=True)
    out["correct"] = out["target_label"] == out["predicted"]
    out.to_csv("evaluation_output.csv", index=False)
    print("\nSaved evaluation_output.csv. Read the rows where correct is False.")


if __name__ == "__main__":
    main()
