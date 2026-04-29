"""Example FastAPI server for Modern ASR.

Run:
    uv add fastapi uvicorn
    uvicorn examples.api_server:app --reload

Endpoints:
    POST /transcribe
    GET /models
"""

from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

from modern_asr import ASRPipeline, list_models

app = FastAPI(title="Modern ASR API")

# Global pipeline instance (can be swapped per request or kept hot)
_pipeline: ASRPipeline | None = None


class TranscribeResponse(BaseModel):
    text: str
    model_id: str
    language: str | None
    duration: float | None = None


@app.on_event("startup")
def startup() -> None:
    global _pipeline
    _pipeline = ASRPipeline("sensevoice-small")


@app.get("/models")
def get_models() -> dict:
    return {"models": list_models()}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("sensevoice-small"),
    language: str = Form("auto"),
    task: str = Form("transcribe"),
) -> TranscribeResponse:
    global _pipeline
    if _pipeline is None:
        _pipeline = ASRPipeline(model)
    elif _pipeline.model is None or _pipeline.model.model_id != model:
        _pipeline.switch_model(model)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = _pipeline(tmp_path, task=task, language=language)

    return TranscribeResponse(
        text=result.text,
        model_id=result.model_id or model,
        language=result.language,
        duration=result.duration,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
