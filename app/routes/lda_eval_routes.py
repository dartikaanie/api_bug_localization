# routers/ml_lda.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.services.lda_evaluation_service import evaluate_lda_model
from app.deps import get_current_user  

router = APIRouter(
    prefix="/ml/lda",
    tags=["Machine Learning - LDA"],
)


def _uid(u):
    return u["uid"] if isinstance(u, dict) else getattr(u, "uid", None)


class LDAEvalResponse(BaseModel):
    status: str
    artifacts: Dict[str, Optional[str]]
    num_topics: int
    num_documents: int
    metrics: Dict[str, Any]
    top_keywords: Dict[str, List[str]]
    bug_ids: Optional[List[int]]
    explanations: Dict[str, str]


@router.get("/evaluate", response_model=LDAEvalResponse)
def evaluate_lda(
    organization: str,
    project: str,
    user = Depends(get_current_user),
):
    """
    Evaluasi model LDA (gensim) yang digunakan di EasyFix untuk
    suatu organization & project tertentu.

    Authorization:
    - Hanya user yang merupakan member project (org_slug, project_slug)
      yang boleh mengakses endpoint ini.

    Mengembalikan:
    - Lokasi artefak (bugs_clean.csv, lda_model.gensim, lda_dictionary.dict, topic_mat.npy)
    - num_topics, num_documents
    - coherence_score (c_v) dan log_perplexity
    - top_keywords per topic
    - bug_ids (urutan bug yang align dengan topic_mat)
    - penjelasan metrik untuk kebutuhan laporan / UI
    """
    # --- Authorization: pastikan user adalah member project ---
    if not _uid(user):
        raise HTTPException(status_code=401, detail="Not authenticated")

    # try:
    #     # fungsi ini diasumsikan akan raise PermissionError kalau user
    #     # bukan member dari project tsb
    #     ensure_project_member(
    #         org_slug=org_slug,
    #         project_slug=project_slug,
    #         user_id=user_id,
    #     )
    # except PermissionError as e:
    #     raise HTTPException(status_code=403, detail=str(e))

    # --- Evaluasi model LDA untuk org & project ini ---
    try:
        # kalau artefak LDA dipisah per project, service bisa gunakan
        # org_slug & project_slug untuk resolve path (mis. out_lda/{org}/{project})
        result = evaluate_lda_model(
            org_slug=organization,
            project_slug=project,
            user_id=user,
        )
        return result

    except FileNotFoundError as e:
        # artefak / input training tidak lengkap
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        # masalah di data (mis. clean_text kosong setelah preprocess)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate LDA model: {e}",
        )
