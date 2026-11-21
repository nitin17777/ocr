"""FastAPI router for dataset evaluation analytics and failure exploration."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException, status

from webapp.schemas import (
    AnalyticsSummaryResponse,
    ConfusionMatrixData,
    FailuresResponse,
)
from webapp.services.analytics import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def get_summary():
    """Retrieve high-level accuracy metrics and domain-stratified breakdown."""
    try:
        data = analytics_service.get_summary()
        return AnalyticsSummaryResponse(**data)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate analytics summary: {str(e)}",
        )


@router.get("/confusion", response_model=ConfusionMatrixData)
async def get_confusion(
    top: int = Query(20, ge=4, le=50, description="Top characters to include in confusion matrix"),
    domain: Optional[str] = Query(None, description="Optional domain filter ('math' or 'text')"),
):
    """Retrieve character-level confusion matrix heatmap and sorted confusion pairs."""
    try:
        data = analytics_service.get_confusion(top=top, domain=domain)
        return ConfusionMatrixData(**data)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate confusion matrix: {str(e)}",
        )


@router.get("/failures", response_model=FailuresResponse)
async def get_failures(
    q: Optional[str] = Query(None, description="Case-insensitive search query in input, target, or prediction"),
    domain: Optional[str] = Query(None, description="Domain filter ('math' or 'text')"),
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(20, ge=1, le=100, description="Rows per page"),
):
    """Search and paginate failed predictions with character-level diff opcodes."""
    try:
        data = analytics_service.get_failures(
            q=q,
            domain=domain,
            page=page,
            page_size=page_size,
        )
        return FailuresResponse(**data)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch failures: {str(e)}",
        )
