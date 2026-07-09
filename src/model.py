"""
Train a multinomial logistic regression baseline and an XGBoost classifier to
predict match outcome (home win / draw / away win) from Elo + form features.

Time-based split: never shuffle time-ordered sports data. Train on everything
before TRAIN_END, evaluate generically on the holdout window right before the
2026 World Cup kicked off, then separately backtest on the World Cup itself
(see backtest.py) -- that backtest is the headline result.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from elo import compute_elo_history
from features import FEATURE_COLUMNS, LABEL_ORDER, build_feature_frame
from load_data import load_results, played_mask as pm

TRAIN_END = "2024-01-01"       # train: everything strictly before this
HOLDOUT_END = "2026-06-11"     # generic time-based test window: [TRAIN_END, HOLDOUT_END)
                                 # 2026 World Cup start date; WC itself is backtested separately


def prepare_datasets():
    df = load_results()
    mask = pm(df)
    elo_df = compute_elo_history(df, mask)
    feat_df = build_feature_frame(elo_df)
    played = feat_df[mask].dropna(subset=FEATURE_COLUMNS).copy()

    train = played[played["date"] < TRAIN_END]
    holdout = played[(played["date"] >= TRAIN_END) & (played["date"] < HOLDOUT_END)]
    return feat_df, mask, train, holdout


def fit_models(train: pd.DataFrame):
    X_train = train[FEATURE_COLUMNS].values
    y_train = train["result"].values

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    logreg = LogisticRegression(max_iter=1000, multi_class="multinomial")
    logreg.fit(X_train_scaled, y_train)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    label_to_idx = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    y_train_idx = np.array([label_to_idx[y] for y in y_train])
    xgb.fit(X_train, y_train_idx)

    return scaler, logreg, xgb


def evaluate(name, y_true, proba, classes):
    pred = classes[np.argmax(proba, axis=1)]
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, proba, labels=classes)
    print(f"{name:22s} accuracy={acc:.3f}  log_loss={ll:.3f}  n={len(y_true)}")
    return {"accuracy": acc, "log_loss": ll, "n": len(y_true)}


def logreg_proba(scaler, logreg, df):
    X = scaler.transform(df[FEATURE_COLUMNS].values)
    proba = logreg.predict_proba(X)
    return proba, logreg.classes_


def xgb_proba(xgb, df):
    proba = xgb.predict_proba(df[FEATURE_COLUMNS].values)
    return proba, np.array(LABEL_ORDER)


def main():
    feat_df, mask, train, holdout = prepare_datasets()
    print(f"Train matches:   {len(train)} (before {TRAIN_END})")
    print(f"Holdout matches: {len(holdout)} ({TRAIN_END} to {HOLDOUT_END})\n")

    # Validation fit: train-only, so the holdout window gives an honest
    # time-based generic accuracy/log-loss reading (never seen during fit).
    val_scaler, val_logreg, val_xgb = fit_models(train)

    results = {}
    for name, df in [("train", train), ("holdout", holdout)]:
        lp, lc = logreg_proba(val_scaler, val_logreg, df)
        xp, xc = xgb_proba(val_xgb, df)
        results[f"logreg_{name}"] = evaluate(f"LogReg [{name}]", df["result"].values, lp, lc)
        results[f"xgb_{name}"] = evaluate(f"XGBoost [{name}]", df["result"].values, xp, xc)

    # Production fit: all data strictly before the World Cup started, so the
    # WC-2026 backtest and forward simulation get the strongest possible model
    # while still never touching a single 2026 World Cup match during fit.
    pre_wc = pd.concat([train, holdout]).sort_values("date")
    prod_scaler, prod_logreg, prod_xgb = fit_models(pre_wc)
    print(f"\nProduction fit on {len(pre_wc)} matches (all data before {HOLDOUT_END})")

    with open("results/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    import pickle
    with open("results/models.pkl", "wb") as f:
        pickle.dump({"scaler": prod_scaler, "logreg": prod_logreg, "xgb": prod_xgb}, f)

    feat_df.to_parquet("results/feature_frame.parquet")
    print("\nSaved: results/model_metrics.json, results/models.pkl, results/feature_frame.parquet")


if __name__ == "__main__":
    main()
