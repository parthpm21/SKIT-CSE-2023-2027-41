class AnomalyMap(BaseModel):
    """Output of the Anomaly Decoder — spatial or temporal localization."""
    granularity: str  # "pixel" | "frame"
    values: List[float] = Field(..., description="Flattened heatmap or per-frame score sequence")
    shape: List[int]   # original (H, W) for pixel maps, or [T] for frame sequences


class ConfidenceResult(BaseModel):
    """Output of the Confidence Decoder — scalar authenticity verdict."""
    label: str          # "real" | "fake"
    confidence: float


class FusionResult(BaseModel):
    sample_id: str
    branches_used: List[str]
    anomaly_map: AnomalyMap
    confidence: ConfidenceResult
    fusion_model_version: str