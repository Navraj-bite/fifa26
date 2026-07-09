"""
Backtest the trained models specifically on 2026 FIFA World Cup matches played
so far. This is the headline evidence: real accuracy/log-loss on this exact,
live tournament, using only pre-match information (Elo/form computed causally,
model trained on data strictly before the tournament started).
"""
import json
import pickle

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, LABEL_ORDER
from model import HOLDOUT_END, evaluate, logreg_proba, xgb_proba


def main():
    feat_df = pd.read_parquet("results/feature_frame.parquet")
    with open("results/models.pkl", "rb") as f:
        models = pickle.load(f)
    scaler, logreg, xgb = models["scaler"], models["logreg"], models["xgb"]

    wc26 = feat_df[
        (feat_df["tournament"] == "FIFA World Cup")
        & (feat_df["date"] >= HOLDOUT_END)
        & feat_df["result"].notna()
    ].dropna(subset=FEATURE_COLUMNS).copy()

    print(f"2026 World Cup matches played so far (with confirmed scores): {len(wc26)}\n")

    lp, lc = logreg_proba(scaler, logreg, wc26)
    xp, xc = xgb_proba(xgb, wc26)

    results = {}
    results["logreg_wc2026"] = evaluate("LogReg [WC 2026]", wc26["result"].values, lp, lc)
    results["xgb_wc2026"] = evaluate("XGBoost [WC 2026]", wc26["result"].values, xp, xc)

    # Per-match detail using the XGBoost model (marginally better on holdout)
    wc26 = wc26.reset_index(drop=True)
    xp_df = pd.DataFrame(xp, columns=[f"p_{c}" for c in xc])
    detail = pd.concat([wc26[["date", "home_team", "away_team", "home_score", "away_score", "result"]], xp_df], axis=1)
    detail["predicted"] = xc[np.argmax(xp, axis=1)]
    detail["correct"] = detail["predicted"] == detail["result"]

    pd.set_option("display.width", 140)
    print("\nPer-match detail (XGBoost):")
    print(detail.to_string(index=False))

    detail.to_csv("results/wc2026_backtest_detail.csv", index=False)
    with open("results/wc2026_backtest_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/wc2026_backtest_detail.csv, results/wc2026_backtest_metrics.json")


if __name__ == "__main__":
    main()
