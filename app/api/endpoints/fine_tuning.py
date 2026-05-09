from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from astro_chatbot_service.models.schemas import FineTuneDatasetRequest, FineTuneJobRequest
from astro_chatbot_service.services.fine_tuning import FineTuningService

router = APIRouter(prefix="/fine-tuning")


@router.post("/dataset")
def build_dataset(
    request: FineTuneDatasetRequest,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    service = FineTuningService(db)
    return service.build_dataset_payload(request)


@router.post("/job-payload")
def build_job_payload(
    request: FineTuneJobRequest,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    service = FineTuningService(db)
    return service.build_job_payload(request)

