# app/services/project_service.py
import re
import os
from typing import Dict, Any, Tuple
from firebase_admin import firestore
from app.core.firebase import db

from pathlib import Path
from typing import Dict, Any

from neo4j import GraphDatabase
from app.config import settings
from app.helper import _dbname

from app.core.neo4j_conn import get_driver

# sudah ada sebelumnya:
ML_ENGINE_DIR = Path(settings.ML_ENGINE_DIR)
ML_PYTHON_BIN = settings.ML_PYTHON_BIN
ML_MAIN_SCRIPT = settings.ML_MAIN_SCRIPT
ML_DATASOURCE_BASE = ML_ENGINE_DIR / "datasource"

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 -]+", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"

def _dbname(org: str, proj: str) -> str:
    clean = lambda x: re.sub(r"[^a-z0-9]+", "", x.strip().lower())
    base = f"{clean(org)}{clean(proj)}"
    # Firestore & Neo4j naming-safe + panjang aman
    return base[:63] if len(base) > 63 else base

async def create_project_simple(
    organization_name: str,
    project_name: str,
    owner_uid: str,
) -> Dict[str, str]:
    org_slug  = _slugify(organization_name)
    proj_slug = _slugify(project_name)
    dbname    = _dbname(organization_name, project_name)

    org_ref  = db.collection("organizations").document(org_slug)
    proj_ref = org_ref.collection("projects").document(proj_slug)
    user_ref = db.collection("users").document(owner_uid)

     # ---- add PROJECT ke users node (field projects) ----
    project_entry = {
        "org_id": org_slug,      # di screenshot: "idn"
        "project_id": proj_slug, # di screenshot: "easyfix"
    }

    user_ref.set(
        {
            # ArrayUnion akan menambahkan item jika belum ada;
            # kalau map-nya identik, Firestore tidak menduplikasi
            "projects": firestore.ArrayUnion([project_entry]),
            "updated_at": firestore.SERVER_TIMESTAMP,
            # optional: kalau user baru belum punya uid/created_at, bisa ditambah di sini
            "uid": owner_uid,
        },
        merge=True,
    )

    
    # ---- create PROJECT ----
    proj_ref.set({
        "name": project_name,
        "slug": proj_slug,
        "organization_name": organization_name,
        "organization_slug": org_slug,
        "database_name": dbname,
        "data_collection_name": dbname,
        "owner_uid": owner_uid,
        "members": firestore.ArrayUnion([owner_uid]),
        "status": "active",
        "updated_at": firestore.SERVER_TIMESTAMP,
        "created_at": firestore.SERVER_TIMESTAMP,
    }, merge=False)

    return {
        "organization_name": organization_name,
        "project_name": project_name,
        "database_name": dbname,
        "data_collection_name": dbname,
        "org_slug": org_slug,
        "project_slug": proj_slug,
        "project_path": f"organizations/{org_slug}/projects/{proj_slug}",
    }


async def get_organization(organization_name: str) -> Dict[str, Any]:
    """
    Ambil data organisasi berdasarkan nama (bukan slug).
    Raise ValueError jika tidak ditemukan.
    """
    org_slug = _slugify(organization_name)
    org_ref = db.collection("organizations").document(org_slug)
    org_doc = org_ref.get()

    if not org_doc.exists:
        raise ValueError(f"Organization '{organization_name}' not found")

    data = org_doc.to_dict() or {}
    return {
        "organization_name": organization_name,
        "org_slug": org_slug,
        "organization_path": f"organizations/{org_slug}",
        "data": data,
    }


async def get_project(organization_name: str, project_name: str) -> Dict[str, Any]:
    """
    Ambil data project di bawah suatu organisasi.
    Raise ValueError jika organization atau project tidak ditemukan.
    """
    org_info = await get_organization(organization_name)
    org_slug = org_info["org_slug"]

    proj_slug = _slugify(project_name)
    proj_ref = (
        db.collection("organizations")
        .document(org_slug)
        .collection("projects")
        .document(proj_slug)
    )
    proj_doc = proj_ref.get()

    if not proj_doc.exists:
        raise ValueError(
            f"Project '{project_name}' not found under organization '{organization_name}'"
        )

    proj_data = proj_doc.to_dict() or {}
    return {
        "organization_name": organization_name,
        "project_name": project_name,
        "org_slug": org_slug,
        "project_slug": proj_slug,
        "project_path": f"organizations/{org_slug}/projects/{proj_slug}",
        "data": proj_data,
    }

async def get_ml_status(organization_name: str, project_name: str):
    """
    Ambil status process ML engine dari field `ml_status` di Firestore.
    Raise ValueError jika org atau project tidak ditemukan.
    """
    proj_info = await get_project(organization_name, project_name)
    project_data = proj_info["data"]

    ml_status = project_data.get("ml_status")
    return {
        "org_slug": proj_info["org_slug"],
        "project_slug": proj_info["project_slug"],
        "ml_status": ml_status or {},
    }




def check_ml_environment(database_name: str) -> Dict[str, Any]:
    """
    Cek environment sebelum menjalankan ML engine:
    - Folder ML engine ada?
    - File datasource <database_name>.jsonl ada?
    - Koneksi ke Neo4j OK?
    """
    result: Dict[str, Any] = {
        "ok": True,
        "ml_engine_dir_ok": True,
        "datasource_ok": True,
        "neo4j_ok": True,
        "datasource_path": None,
        "neo4j_error": None,
    }

    # 1) cek folder ML engine
    if not ML_ENGINE_DIR.is_dir():
        result["ok"] = False
        result["ml_engine_dir_ok"] = False
        return result

    # 2) cek file datasource
    datasource_path = ML_DATASOURCE_BASE / f"{database_name}.jsonl"
    result["datasource_path"] = str(datasource_path)

    if not datasource_path.exists():
        result["ok"] = False
        result["datasource_ok"] = False
        # lanjut cek Neo4j juga supaya user dapat info lengkap

    # 3) cek koneksi Neo4j + database
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        with driver:
            with driver.session(database=database_name) as session:
                session.run("RETURN 1 AS ok").single()
    except Exception as e:
        result["ok"] = False
        result["neo4j_ok"] = False
        result["neo4j_error"] = f"{e.__class__.__name__}: {e}"

    return result

def _model_path(org: str, proj: str):
    base = f"{org}{proj}".lower()
    return f"{ML_ENGINE_DIR}/out_lda/{base}/lda_sklearn_model_meta.npz"

def _datasource_path(org: str, proj: str):
    base = f"{org}{proj}".lower()
    return f"{ML_DATASOURCE_BASE}/{base}.jsonl"


async def check_project_status(org: str, proj: str) -> Tuple[int, Dict[str, Any]]:
    status: Dict[str, Any] = {
        "project_created": False,
        "datasource_created": False,
        "model_created": False,
        "db_stored": False,
    }

    # --- Step 1: Check project exists in Firestore ---
    try:
        db = firestore.Client()
        org_ref = db.collection("organizations").document(org)
        proj_ref = org_ref.collection("projects").document(proj)
        proj_doc = proj_ref.get()

        if proj_doc.exists:
            status["project_created"] = True
        else:
            # project belum ada -> berhenti di step 1
            return 1, status
    except Exception as e:
        print(f"[STATUS] Error checking project in Firestore: {e}")
        # gagal cek project -> anggap belum lewat step 1
        return 1, status

    # --- Step 2: Check datasource files exist ---
    datasource_path = _datasource_path(org, proj)
    if os.path.exists(datasource_path):
        status["datasource_created"] = True
    else:
        # datasource belum ada -> berhenti di step 2
        return 2, status

    # --- Step 3: Check model files exist ---
    model_path = _model_path(org, proj)
    print(model_path)  # ini yang muncul di log: ml_engine/out_lda/idneasyfix/...
    if os.path.exists(model_path):
        status["model_created"] = True
    else:
        # model belum ada -> berhenti di step 3
        return 3, status

    # --- Step 4: Check data stored in Neo4j / DB ---
    try:
        driver = await get_driver()
        dbname = _dbname(org, proj)

        async with driver.session(database=dbname) as session:
            result = await session.run("MATCH (b:Bug) RETURN count(b) AS cnt")
            record = await result.single()
            bug_count = record["cnt"] if record else 0

        if bug_count > 0:
            status["db_stored"] = True
    except Exception as e:
        print(f"[STATUS] Error checking db_stored in Neo4j: {e}")
        # kalau cek DB error, jangan bikin API crash, anggap step 4 belum lewat

    # PENTING: SELALU ADA RETURN DI PALING AKHIR
    # step 4 = sudah sampai cek DB (apapun hasil db_stored)
    return 4, status
