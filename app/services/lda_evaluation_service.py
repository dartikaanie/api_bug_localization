# ml_engine/lda_evaluator_gensim.py

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
from app.helper import _dbname

# === Konfigurasi path: sesuaikan dengan .env kamu ===
# Di script training kamu:
#   PATH_NLP_OUT -> folder bugs_clean.csv
#   PATH_LDA_OUT -> folder artefak LDA (lda_model.gensim, lda_dictionary.dict, topic_mat.npy)
BASE_DIR = Path(__file__).resolve().parent.parent  # misal: project root di atas ml_engine/

ML_ENGINE_DIR: Path = Path(os.getenv("ML_ENGINE_DIR", BASE_DIR / "ml_engine")).resolve()
NLP_OUT_DIR = ML_ENGINE_DIR / os.getenv("PATH_NLP_OUT", "out_nlp")
LDA_OUT_DIR = ML_ENGINE_DIR / os.getenv("PATH_LDA_OUT", "out_lda")


def _tokenize_clean_text(texts: List[str]) -> List[List[str]]:
    """
    Harus konsisten dengan fungsi _tokenize_clean_text di 02_lda_topics.py (gensim):
    clean_text sudah dipreproses → cukup split spasi.
    """
    return [str(t).split() for t in texts]


def _build_corpus(dictionary: corpora.Dictionary, tokenized_docs: List[List[str]]):
    """
    Build corpus BoW dari tokenized docs dan dictionary yang sudah tersimpan.
    Tidak memanggil filter_extremes lagi, supaya konsisten dengan training.
    """
    return [dictionary.doc2bow(doc) for doc in tokenized_docs]


def _doc_topic_matrix_from_model(
    lda_model: models.LdaModel,
    corpus,
    num_topics: int,
) -> np.ndarray:
    """
    Fallback kalau topic_mat.npy hilang:
    generate doc-topic matrix dari model & corpus.
    """
    n_docs = len(corpus)
    mat = np.zeros((n_docs, num_topics), dtype=np.float32)
    for i, bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        for tid, prob in doc_topics:
            mat[i, tid] = prob
    return mat


def evaluate_lda_model(org_slug: str, project_slug: str, user_id: str):
    """
    Evaluasi model LDA (gensim) EasyFix:

    - Load bugs_clean.csv → ambil clean_text
    - Tokenisasi sama persis dengan saat training
    - Load dictionary + model
    - Build corpus (BoW) dari dictionary & tokenized docs
    - Hitung:
        * coherence (c_v berbasis teks)
        * log_perplexity (berbasis corpus)
        * num_topics, num_documents
        * top keywords per topic
    - Baca topic_mat.npy kalau ada (untuk dicek/ditampilkan)
    """
    nameOfProject= _dbname(org_slug, project_slug)
    # nameOfProject = os.getenv("PROJECT_NAME", "idneasyfix")
    BUGS_CLEAN_PATH = NLP_OUT_DIR / nameOfProject/"bugs_clean.csv"
    LDA_MODEL_PATH = LDA_OUT_DIR / nameOfProject/"lda_model.gensim"
    DICTIONARY_PATH = LDA_OUT_DIR / nameOfProject/"lda_dictionary.dict"
    TOPIC_MAT_PATH = LDA_OUT_DIR / nameOfProject/"topic_mat.npy"
    BUG_IDS_PATH = LDA_OUT_DIR / nameOfProject/"bug_ids.txt"

    # --- Cek artefak & input ---
    if not BUGS_CLEAN_PATH.exists():
        raise FileNotFoundError(f"bugs_clean.csv not found at: {BUGS_CLEAN_PATH}")

    if not LDA_MODEL_PATH.exists():
        raise FileNotFoundError(f"LDA model not found at: {LDA_MODEL_PATH}")

    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(f"LDA dictionary not found at: {DICTIONARY_PATH}")

    # --- Load data cleaning hasil NLP ---
    df = pd.read_csv(BUGS_CLEAN_PATH)
    if "clean_text" not in df.columns:
        raise ValueError("Missing 'clean_text' column in bugs_clean.csv")

    texts_raw = df["clean_text"].fillna("").astype(str).tolist()
    tokenized_docs = _tokenize_clean_text(texts_raw)

    if not tokenized_docs:
        raise ValueError("No documents available for evaluation (tokenized_docs empty).")

    # --- Load artefak LDA ---
    dictionary: corpora.Dictionary = corpora.Dictionary.load(str(DICTIONARY_PATH))
    lda_model: models.LdaModel = models.LdaModel.load(str(LDA_MODEL_PATH))

    num_topics = lda_model.num_topics

    # Build corpus BoW dari teks + dictionary tersimpan
    corpus = _build_corpus(dictionary, tokenized_docs)
    num_documents = len(corpus)

    # --- Coherence (c_v berbasis teks) ---
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_score = coherence_model.get_coherence()

    # --- log_perplexity ---
    # gensim LdaModel.log_perplexity menerima corpus → log-likelihood per token ~ semakin tinggi (kurang negatif) lebih baik
    log_perplexity = lda_model.log_perplexity(corpus)

    # --- topic_mat: pakai file yang sudah disimpan, atau regenerate kalau tidak ada ---
    if TOPIC_MAT_PATH.exists():
        topic_mat = np.load(TOPIC_MAT_PATH)
    else:
        topic_mat = _doc_topic_matrix_from_model(lda_model, corpus, num_topics)

    # --- Top keywords per topic ---
    top_keywords: Dict[str, Any] = {}
    topn = 10
    for topic_id in range(num_topics):
        words_probs = lda_model.show_topic(topic_id, topn=topn)
        top_keywords[f"topic_{topic_id}"] = [w for (w, _) in words_probs]

    # --- Bug IDs (untuk memastikan alignment topic_mat dengan bug) ---
    bug_ids: Optional[List[int]] = None
    if BUG_IDS_PATH.exists():
        with open(BUG_IDS_PATH, "r", encoding="utf-8") as f:
            bug_ids = [int(line.strip()) for line in f if line.strip()]

    explanations = {
        "coherence_score": (
            "Coherence c_v (0–1) mengukur seberapa koheren kata-kata dalam satu topik "
            "berdasarkan co-occurrence di dokumen. Semakin tinggi, topik biasanya makin bermakna."
        ),
        "log_perplexity": (
            "Log-perplexity dihitung terhadap corpus BoW. Nilai yang lebih tinggi (kurang negatif) "
            "biasanya menunjukkan model yang lebih baik dalam merepresentasikan distribusi kata."
        ),
        "num_topics": (
            "Jumlah topik yang dipelajari model LDA. Nilai ini ditentukan oleh proses tuning "
            "atau resolve_lda_params pada tahap training."
        ),
        "topic_mat": (
            "Matrix distribusi topik per dokumen dengan shape (num_documents x num_topics). "
            "Setiap baris merepresentasikan probabilitas topik untuk satu bug report."
        ),
        "top_keywords": (
            "Kata-kata dengan probabilitas tertinggi pada masing-masing topik. "
            "Digunakan untuk menginterpretasikan makna topik secara kualitatif."
        ),
    }

    result: Dict[str, Any] = {
        "status": "success",
        "artifacts": {
            "nlp_out_dir": str(NLP_OUT_DIR),
            "lda_out_dir": str(LDA_OUT_DIR),
            "bugs_clean_path": str(BUGS_CLEAN_PATH),
            "model_path": str(LDA_MODEL_PATH),
            "dictionary_path": str(DICTIONARY_PATH),
            "topic_mat_path": str(TOPIC_MAT_PATH) if TOPIC_MAT_PATH.exists() else None,
            "bug_ids_path": str(BUG_IDS_PATH) if BUG_IDS_PATH.exists() else None,
        },
        "num_topics": int(num_topics),
        "num_documents": int(num_documents),
        "explanations": explanations,
        "metrics": {
            "coherence_score": float(coherence_score),
            "coherence_type": "c_v",
            "log_perplexity": float(log_perplexity),
        },
        "top_keywords": top_keywords,
        "bug_ids": bug_ids,
    }

    return result
