"""Human review API: list, read and decide the cases the assistant sent to a reviewer.

The reviewer endpoints are protected by the backend itself: every call needs the `X-Reviewer-Token` header (the value of
`REVIEWER_TOKEN`) and an `X-Reviewer-Id` (a name or stable id, recorded with the decision). With no `REVIEWER_TOKEN`
configured the endpoints answer 503: there is no open mode. The frontend only forwards the headers; it does not decide.

`GET /api/review-status/{id}` is for the engineer who asked: it needs no token (the id is an unguessable random value)
and shows only the state and the outcome in plain words.
"""

from __future__ import annotations

import hmac
import os
import socket
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query
from pydantic import BaseModel, Field

from app.services import review_queue

router = APIRouter(prefix="/api", tags=["review"])

ReviewId = Path(pattern=r"^[0-9a-f]{32}$", description="The review id returned with the answer")


class ReviewDecision(BaseModel):
    decision: Literal["confirm", "correct", "reject"]
    reason: str | None = Field(default=None, max_length=review_queue.MAX_REASON_CHARS)
    corrected_text: str | None = Field(default=None, max_length=review_queue.MAX_CORRECTION_CHARS)
    evidence: list[str] | None = Field(default=None, max_length=review_queue.MAX_EVIDENCE)


def reviewer(
    x_reviewer_token: str | None = Header(default=None),
    x_reviewer_id: str | None = Header(default=None),
) -> str:
    """The authenticated reviewer's id. Raises 503 (not configured), 401 (token) or 400 (id)."""
    token = os.getenv("REVIEWER_TOKEN", "")
    if not token:
        raise HTTPException(status_code=503, detail="human review is not configured on this server")
    if not x_reviewer_token or not hmac.compare_digest(x_reviewer_token.encode("utf-8"), token.encode("utf-8")):
        raise HTTPException(status_code=401, detail="a valid X-Reviewer-Token is required")
    reviewer_id = (x_reviewer_id or "").strip()
    if not review_queue.REVIEWER_ID.fullmatch(reviewer_id):
        raise HTTPException(status_code=400, detail="an X-Reviewer-Id (a name or id) is required")
    return reviewer_id


@router.get("/reviews")
def list_reviews(
    status: Literal["pending", "decided", "all"] = Query("pending"),
    limit: int = Query(100, ge=1, le=500),
    _: str = Depends(reviewer),
):
    """The review list. `queue_instance` is the backend pod whose queue this is (one replica is needed for one queue)."""
    return {
        "queue_instance": os.getenv("HOSTNAME") or socket.gethostname(),
        "counts": review_queue.counts(),
        "items": review_queue.list_items(status, limit),
    }


@router.get("/reviews/{review_id}")
def read_review(review_id: str = ReviewId, _: str = Depends(reviewer)):
    try:
        return review_queue.get_item(review_id)
    except review_queue.NotFound:
        raise HTTPException(status_code=404, detail="no such review") from None


@router.post("/reviews/{review_id}/decision")
def decide_review(body: ReviewDecision, review_id: str = ReviewId, who: str = Depends(reviewer)):
    try:
        return review_queue.decide(review_id, body.decision, who, body.reason, body.corrected_text, body.evidence)
    except review_queue.NotFound:
        raise HTTPException(status_code=404, detail="no such review") from None
    except review_queue.AlreadyDecided:
        raise HTTPException(status_code=409, detail="this case has already been decided") from None
    except review_queue.InvalidDecision as error:
        raise HTTPException(status_code=422, detail={"problems": error.problems}) from None


@router.get("/review-status/{review_id}")
def review_status(review_id: str = ReviewId):
    """For the engineer: where their answer is in review, and the outcome in plain words."""
    try:
        return review_queue.engineer_view(review_id)
    except review_queue.NotFound:
        raise HTTPException(status_code=404, detail="no such review") from None
