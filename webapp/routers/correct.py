"""Correction inference endpoint."""

from fastapi import APIRouter, HTTPException, status
from webapp.schemas import CorrectRequest, CorrectResponse
from webapp.services.model_manager import model_manager

router = APIRouter(prefix="/api", tags=["correct"])


@router.post("/correct", response_model=CorrectResponse)
async def post_correct(request: CorrectRequest) -> CorrectResponse:
    """Run batched greedy OCR correction and return character-level diffs."""
    if model_manager.is_training:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"reason": "training_in_progress"},
        )

    try:
        return await model_manager.correct(
            text=request.text,
            adapter=request.adapter,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as r_err:
        if str(r_err) == "training_in_progress":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"reason": "training_in_progress"},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(r_err),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
