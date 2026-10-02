"""API-13, API-14: chỉ số đánh giá và thư viện giao dịch mẫu."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from ..config import API_PREFIX
from ..deps import get_artifacts
from ..errors import ApiError
from ..loader import Artifacts
from ..schemas import SampleList

router = APIRouter(prefix=API_PREFIX, tags=["chỉ số và mẫu"])


@router.get("/metrics", summary="API-13 — nội dung metrics.json, toàn bộ hoặc một phần",
            response_description="Cấu trúc ở docs/06 §6.2")
def metrics(section: str | None = Query(None, description="Tên một khóa của metrics.json, ví dụ test_scores"),
            loaded: Artifacts = Depends(get_artifacts)):
    # JSON dựng sẵn lúc khởi động: mã hoá lại 57.000 điểm mỗi lần gọi thì chậm vô ích
    if section is None:
        return Response(loaded.metrics_json, media_type="application/json")
    if section not in loaded.metrics:
        raise ApiError(400, "INVALID_REQUEST", f"metrics.json không có mục {section!r}",
                       {"available": list(loaded.metrics)})
    return Response(loaded.metrics_section(section), media_type="application/json")


@router.get("/samples", response_model=SampleList, summary="API-14 — thư viện giao dịch mẫu")
def samples(category: Literal["fraud_easy", "fraud_hard", "legit_easy", "legit_hard"] | None = None,
            loaded: Artifacts = Depends(get_artifacts)):
    pool = loaded.sample_pool
    items = [
        {**{k: item[k] for k in ("id", "category", "label", "risk_score", "description", "features")},
         "amount": item["features"]["Amount"]}
        for item in pool["items"]
        if category is None or item["category"] == category
    ]
    return {"items": items, "categories": pool["categories"], "model_version": pool["model_version"],
            "reference_threshold": pool["reference_threshold"]}
