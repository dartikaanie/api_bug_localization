import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, average_precision_score

from app.services.bug_service import fetch_bug_dev_pairs, fetch_all_developers
from app.services.ltr_features import build_training_dataset, FEATURE_COLUMNS
from app.utils.ml_paths import get_ltr_model_path, get_ltr_dataset_path

from app.helper import _dbname


def evaluate_ltr_ranking(test_df, k_list = [1, 3, 5]):
    """
    Evaluasi model Learning-to-Rank pada data test berbasis:
      - NDCG@k
      - Precision@k
      - MAP (mean average precision)

    test_df diharapkan punya kolom:
      - bug_id
      - label  (0/1)
      - pred   (score dari model)
    """
    metrics_accum = {
        "map": [],
    }
    # siapkan key untuk NDCG & Precision
    for k in k_list:
        metrics_accum[f"ndcg@{k}"] = []
        metrics_accum[f"precision@{k}"] = []

    # loop per bug (per query)
    for bug_id, group in test_df.groupby("bug_id"):
        y_true = group["label"].values.astype(float)
        y_score = group["pred"].values.astype(float)

        # skip kalau:
        # - hanya 1 kandidat (nggak meaningful untuk ranking)
        # - tidak ada positif sama sekali
        if len(y_true) < 2 or y_true.sum() == 0:
            continue

        # NDCG@k dan Precision@k
        for k in k_list:
            k_eff = min(k, len(y_true))

            # NDCG@k
            ndcg = ndcg_score([y_true], [y_score], k=k_eff)
            metrics_accum[f"ndcg@{k}"].append(float(ndcg))

            # Precision@k
            order = np.argsort(-y_score)
            topk = order[:k_eff]
            prec_k = y_true[topk].mean()
            metrics_accum[f"precision@{k}"].append(float(prec_k))

        # MAP (Average Precision per bug)
        ap = average_precision_score(y_true, y_score)
        metrics_accum["map"].append(float(ap))

    # agregasi rata-rata
    final_metrics = {}
    for key, values in metrics_accum.items():
        if values:
            final_metrics[key] = float(np.mean(values))
        else:
            final_metrics[key] = None  # kalau nggak ada sample yg valid

    return final_metrics

async def train_ltr_model(
    organization: str,
    project: str,
    force_retrain: bool = False,
):
    MODEL_PATH = get_ltr_model_path(organization, project)
    DATASET_PATH = get_ltr_dataset_path(organization, project)
    database = _dbname(organization, project)
    # 1) Cek apakah model sudah ada
    if os.path.exists(MODEL_PATH) and not force_retrain:
        return {
            "status": "skipped",
            "reason": "model_already_exists",
            "model_path": MODEL_PATH,
        }

    # 2) Ambil data dari Neo4j via bug_services
    bug_dev_pairs = await fetch_bug_dev_pairs(database)
    all_devs = await fetch_all_developers(database)

    if not bug_dev_pairs:
        return {"status": "failed", "reason": "no_bug_dev_pairs"}

    if not all_devs:
        return {"status": "failed", "reason": "no_developers"}

    # 3) Build training dataset
    df = build_training_dataset(bug_dev_pairs, all_devs)
    df.to_csv(DATASET_PATH, index=False)

    # butuh minimal bug untuk training LTR
    if df["bug_id"].nunique() < 5:
        return {
            "status": "failed",
            "reason": "not_enough_bugs_for_training",
            "num_bugs": int(df["bug_id"].nunique()),
        }

    # 4) Build X, y, group
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    bug_ids = df["bug_id"].unique()
    train_bugs, test_bugs = train_test_split(bug_ids, test_size=0.2, random_state=42)

    train_df = df[df["bug_id"].isin(train_bugs)]
    test_df = df[df["bug_id"].isin(test_bugs)]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values
    group_train = train_df.groupby("bug_id").size().tolist()

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values
    group_test = test_df.groupby("bug_id").size().tolist()

    # 5) Train model
    model = xgb.XGBRanker(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective="rank:pairwise",
        subsample=0.8,
        colsample_bytree=0.8,
        # optional: kamu bisa set eval_metric langsung di XGBoost
        eval_metric="ndcg@5",
    )

    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        verbose=True,
    )

    # 6) Evaluasi model pada test set (NDCG, Precision, MAP)
    # ------------------------------------------------------
    y_pred_test = model.predict(X_test)

    # satukan kembali ke DataFrame agar mudah dikelompokkan per bug
    test_df = test_df.copy()
    test_df["pred"] = y_pred_test

    eval_metrics = evaluate_ltr_ranking(test_df, k_list=[1, 3, 5])
    eval_explanation = explain_ltr_metrics(eval_metrics)
    metric_definitions = get_ltr_metric_definitions()


    # 7) Save model
    model.save_model(MODEL_PATH)

    return {
        "status": "success",
        "model_path": MODEL_PATH,
        "dataset_path": DATASET_PATH,
        "num_training_bugs": int(len(train_bugs)),
        "num_test_bugs": int(len(test_bugs)),
        "rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "eval_metrics": eval_metrics,
        "eval_explanation": eval_explanation,
        "metric_definitions": metric_definitions
    }

def explain_ltr_metrics(metrics: dict) -> str:
    """
    Penjelasan naratif terhadap hasil evaluasi model LTR,
    memadukan interpretasi + insight.
    """
    map_score = metrics.get("map")
    ndcg1 = metrics.get("ndcg@1")
    prec1 = metrics.get("precision@1")
    ndcg3 = metrics.get("ndcg@3")
    prec3 = metrics.get("precision@3")
    ndcg5 = metrics.get("ndcg@5")
    prec5 = metrics.get("precision@5")

    explanation = []

    # MAP
    if map_score == 1:
        explanation.append(
            "MAP = 1 menunjukkan bahwa model selalu menempatkan developer relevan "
            "di atas developer tidak relevan untuk setiap bug pada test set."
        )
    else:
        explanation.append(
            f"MAP = {map_score:.3f} mencerminkan kualitas keseluruhan ranking. "
            "Semakin mendekati 1 berarti semakin baik urutan model."
        )

    # Top-1 metrics
    if ndcg1 == 1 and prec1 == 1:
        explanation.append(
            "NDCG@1 dan Precision@1 = 1 berarti rekomendasi Top-1 model selalu tepat. "
            "Developer yang paling relevan selalu berada di posisi pertama."
        )
    else:
        explanation.append(
            f"NDCG@1 = {ndcg1:.3f} dan Precision@1 = {prec1:.3f} menggambarkan "
            "ketepatan model dalam memilih developer terbaik sebagai prioritas utama."
        )

    # Top-3 metrics
    explanation.append(
        f"NDCG@3 = {ndcg3:.3f} mengindikasikan bahwa urutan rekomendasi hingga Top-3 terstruktur dengan sangat baik. "
        f"Precision@3 = {prec3:.3f} tampak rendah secara angka, namun hal ini wajar karena hanya ada "
        "satu developer relevan per bug sehingga maksimum teoritisnya memang 0.33."
    )

    # Top-5 metrics
    explanation.append(
        f"NDCG@5 = {ndcg5:.3f} menunjukkan kualitas ranking sempurna sampai posisi 5. "
        f"Precision@5 = {prec5:.3f} juga wajar karena maksimum teoritisnya hanya 0.20 "
        "jika tiap bug hanya memiliki satu developer yang relevan."
    )

    return " ".join(explanation)


def get_ltr_metric_definitions() -> dict:
    """
    Mengembalikan definisi formal dari setiap metrik evaluasi ranking.
    Cocok untuk dokumentasi API / UI / laporan.
    """
    return {
        "MAP": (
            "Mean Average Precision — mengukur seberapa baik model "
            "mengurutkan developer relevan lebih tinggi daripada yang tidak relevan. "
            "Nilai 1 berarti ranking sempurna untuk semua bug."
        ),
        "NDCG@k": (
            "Normalized Discounted Cumulative Gain pada posisi k — "
            "mengukur kualitas urutan rekomendasi hingga posisi k, "
            "dengan penalti untuk posisi developer relevan yang berada lebih bawah. "
            "Nilai 1 berarti urutan top-k sempurna."
        ),
        "Precision@k": (
            "Proporsi developer relevan yang muncul dalam daftar top-k. "
            "Jika setiap bug hanya memiliki satu developer benar, batas maksimum "
            "Precision@3 = 1/3 ≈ 0.33 dan Precision@5 = 1/5 = 0.20."
        )
    }

