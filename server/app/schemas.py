from typing import Optional
from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    gradcam_url: str
    processing_time_ms: Optional[float] = None

class PredictBatchItem(BaseModel):
    filename: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    gradcam_url: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None

class ModelsResponseItem(BaseModel):
    id: str
    name: str
    params: str
    input_size: str
    trained_on: str
    roc_auc: float

class Health(BaseModel):
    status: str
    version: str = "1.0.0"
