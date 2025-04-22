# train_and_predict.py

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def load_data(base_dir):
    """Load and concatenate all four training trials."""
    dfs = []
    for t in range(1, 5):
        X = pd.read_csv(f"{base_dir}/trial{t:02d}.x.t.csv", header=None)
        y = pd.read_csv(f"{base_dir}/trial{t:02d}.y.t.csv", header=None, squeeze=True)
        df = pd.concat([X, y.rename("label")], axis=1)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def feature_engineer(df, pca=None, scaler=None, fit=True):
    """Create PCA features + scale; fit on train, apply on val/test."""
    X = df.drop(columns="label", errors="ignore").values
    if fit:
        pca = PCA(n_components=20, random_state=42).fit(X)
    X_pca = pca.transform(X)
    X_aug = np.hstack([X, X_pca])
    if fit:
        scaler = StandardScaler().fit(X_aug)
    X_scaled = scaler.transform(X_aug)
    return X_scaled, pca, scaler

def build_mlp(input_dim):
    """Construct a deep MLP with batch norm and dropout."""
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_save(base_dir):
    # 1) Load & preprocess training data
    full = load_data(base_dir)
    X_full, pca, scaler = feature_engineer(full, fit=True)
    y_full = full["label"].values

    # 2) Hold‑out split for tuning (20%)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full,
        test_size=0.2,
        random_state=42,
        stratify=y_full
    )

    # 3) Train MLP
    mlp = build_mlp(X_tr.shape[1])
    es  = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    mlp.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=512,
        callbacks=[es, rlr],
        verbose=2
    )

    # 4) Train LightGBM
    lgb_tr  = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42
    }
    gbm = lgb.train(
        params,
        lgb_tr,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # 5) OOF predictions for stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_mlp = np.zeros(len(y_full))
    oof_lgb = np.zeros(len(y_full))
    for train_idx, val_idx in kf.split(X_full):
        # MLP fold
        fold_mlp = build_mlp(X_full.shape[1])
        fold_mlp.fit(
            X_full[train_idx], y_full[train_idx],
            epochs=50, batch_size=512, verbose=0
        )
        oof_mlp[val_idx] = fold_mlp.predict(X_full[val_idx]).ravel()
        # GBM fold
        fold_gbm = lgb.train(
            params,
            lgb.Dataset(X_full[train_idx], label=y_full[train_idx]),
            num_boost_round=gbm.best_iteration
        )
        oof_lgb[val_idx] = fold_gbm.predict(X_full[val_idx])

    # 6) Train meta‑model
    meta_X = np.vstack([oof_mlp, oof_lgb]).T
    meta = LogisticRegression()
    meta.fit(meta_X, y_full)

    # 7) Local hold‑out evaluation
    p_mlp_val   = mlp.predict_proba(X_val)[:,1]
    p_gbm_val   = gbm.predict(X_val)
    blend_val   = meta.predict_proba(np.vstack([p_mlp_val, p_gbm_val]).T)[:,1]
    y_hat_val   = (blend_val >= 0.5).astype(int)
    bal_acc_val = balanced_accuracy_score(y_val, y_hat_val)
    print(f"\nHold‑out balanced accuracy: {bal_acc_val:.4f}\n")

    # 8) Write out Test1 & Test2 predictions
    for split in ["Test1-Pred", "Test2-Pred"]:
        out_dir = os.path.join(base_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        for t in range(1, 5):
            Xv = pd.read_csv(f"{base_dir}/trial{t:02d}.x.v.csv", header=None).values
            Xv_pca    = pca.transform(Xv)
            Xv_aug    = np.hstack([Xv, Xv_pca])
            Xv_scaled = scaler.transform(Xv_aug)
            p_m = mlp.predict_proba(Xv_scaled)[:,1]
            p_g = gbm.predict(Xv_scaled)
            blend = meta.predict_proba(np.vstack([p_m, p_g]).T)[:,1]
            np.savetxt(
                os.path.join(out_dir, f"trial{t:02d}.y.v.csv"),
                blend,
                delimiter=","
            )
        print(f"Wrote blended probabilities to {split}/")

if __name__ == "__main__":
    BASE_DIR = "spr25-gait2-Mudhathir-main"
    train_and_save(BASE_DIR)
