"""
DigiCare API server.

Endpoints:
    POST /reports/process        - upload and process a lab report image or PDF
    GET  /patients/{id}/findings - all findings for a patient
    GET  /patients/{id}/brief    - clinical brief for a patient
    POST /patients/{id}/chat     - ask a question about a patient
    GET  /health                 - health check

Environment variables:
    GEMINI_API_KEY  - required
    DATABASE_URL    - optional, PostgreSQL connection string

Run locally:
    set GEMINI_API_KEY=your_key
    python digicare_api.py
    open http://localhost:8090/docs
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

from digicare_pipeline import DigiCarePipeline

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATABASE_URL   = os.environ.get("DATABASE_URL", "")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set.")

_pipeline: Optional[DigiCarePipeline] = None

def get_pipeline() -> DigiCarePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DigiCarePipeline(
            gemini_api_key=GEMINI_API_KEY,
            db_url=DATABASE_URL,
        )
    return _pipeline


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="DigiCare API",
        description="AI clinical intelligence layer for vdocs EMR.",
        version="1.0.0",
    )

    @app.on_event("startup")
    def startup_event():
        get_pipeline()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "gemini_configured": bool(GEMINI_API_KEY),
            "db_configured": bool(DATABASE_URL),
        }

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".pdf"}

    @app.post("/reports/process")
    async def process_report(
        file: UploadFile = File(..., description="Lab report image or PDF"),
        patient_id: str  = Form(...),
        report_id:  str  = Form(...),
        report_date: str = Form(..., description="YYYY-MM-DD"),
    ):
        """
        Upload a lab report. Returns structured findings and a clinical brief.
        PDFs are split by page and each page is processed separately.
        """
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

        ext = Path(file.filename or "upload").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Accepted: PNG, JPG, TIFF, BMP, WEBP, PDF",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            result = get_pipeline().process_report(
                image_path=tmp_path,
                patient_id=patient_id,
                report_id=report_id,
                report_date=report_date,
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/patients/{patient_id}/findings")
    def get_findings(patient_id: str, loinc_concept: Optional[str] = None):
        """All stored findings for a patient, optionally filtered by LOINC concept."""
        findings = get_pipeline().storage.get_patient_findings(patient_id, loinc_concept)
        if not findings and not DATABASE_URL:
            return {
                "message": "No database configured — findings are not persisted between sessions.",
                "findings": [],
            }
        return {"patient_id": patient_id, "findings_count": len(findings), "findings": findings}

    @app.get("/patients/{patient_id}/brief")
    def get_brief(patient_id: str):
        """Generate a clinical brief from all stored findings for a patient."""
        return get_pipeline().get_patient_summary(patient_id)

    class ChatRequest(BaseModel):
        question: str

    @app.post("/patients/{patient_id}/chat")
    def chat(patient_id: str, body: ChatRequest):
        """Answer a clinical question about a patient, with source citations."""
        return get_pipeline().ask(patient_id=patient_id, question=body.question)

    @app.post("/demo/process-local")
    def process_local(
        image_path: str  = Form(...),
        patient_id: str  = Form(...),
        report_id:  str  = Form(...),
        report_date: str = Form(...),
    ):
        """Process a file already on the server. For local testing only."""
        path = Path(image_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        result = get_pipeline().process_report(
            image_path=path,
            patient_id=patient_id,
            report_id=report_id,
            report_date=report_date,
        )
        return JSONResponse(content=result)


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first: pip install fastapi uvicorn python-multipart")
    else:
        import uvicorn
        print("\nDigiCare API starting on http://localhost:8090")
        print("Swagger UI: http://localhost:8090/docs\n")
        uvicorn.run("digicare_api:app", host="0.0.0.0", port=8090, reload=True)
