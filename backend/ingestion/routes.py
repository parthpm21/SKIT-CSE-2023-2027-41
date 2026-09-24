from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "truthlens-preprocessing-gateway"}


import tempfile
from pathlib import Path
from fastapi import UploadFile, File
from .preprocessing import preprocess_image_file

@router.post("/preprocess/image")
async def preprocess_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    tensor = preprocess_image_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return {"filename": file.filename, "shape": list(tensor.shape),
            "mean": float(tensor.mean()), "std": float(tensor.std())}

from .preprocessing import preprocess_video_file

@router.post("/preprocess/video")
async def preprocess_video(file: UploadFile = File(...), num_frames: int = 16):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    tensor = preprocess_video_file(tmp_path, num_frames=num_frames)
    Path(tmp_path).unlink(missing_ok=True)
    return {"filename": file.filename, "num_frames_extracted": int(tensor.shape[0]),
            "shape": list(tensor.shape)}





from backend.inference.router import route_inference

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = route_inference(tmp_path, file.filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {
        "filename": file.filename,
        "media_type": result.media_type,
        "label": result.label,
        "confidence": result.confidence,
        "model_version": result.model_version,
        "details": result.details,
    }