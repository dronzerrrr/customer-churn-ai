"""FastAPI service. Run from the project root:
    uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager
from typing import Callable

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from api.schemas import (
    BatchRequest, BatchResponse, CustomerInput, PredictionResponse,
)
from src.models.batch import run_batch
from src.models.service import ChurnService

log = logging.getLogger("churn-api")
MAX_ERRORS_RETURNED = 100


def create_app(loader: Callable[[], ChurnService] = ChurnService.load) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.service = loader()          # model is loaded ONCE, at startup
            app.state.load_error = None
            log.info("Model loaded: %s", app.state.service.metadata.get("selected_model"))
        except Exception as exc:                  # keep the API up so /health can explain
            app.state.service = None
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            log.error("Model failed to load: %s", app.state.load_error)
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        version="1.0.0",
        description="Scores customers for churn risk using a model trained on the IBM Telco "
                    "sample dataset. Probabilities are estimates, not certainties.",
        lifespan=lifespan,
    )

    def get_service(request: Request) -> ChurnService:
        service = request.app.state.service
        if service is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded ({request.app.state.load_error}). "
                       "Run: python -m src.models.train")
        return service

    @app.get("/health", tags=["system"])
    def health(request: Request):
        loaded = request.app.state.service is not None
        return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
                "detail": None if loaded else request.app.state.load_error}

    @app.get("/model-info", tags=["system"])
    def model_info(service: ChurnService = Depends(get_service)):
        return service.info()

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(customer: CustomerInput,
                explain: bool = Query(False, description="Include the top reasons (SHAP)."),
                service: ChurnService = Depends(get_service)):
        X, estimated = customer.to_frame()
        scored = service.score(X).iloc[0]
        reasons = service.explain(X)["reasons"] if explain else None
        return PredictionResponse(
            customerID=customer.customerID,
            churn_probability=float(scored["churn_probability"]),
            predicted_churn=bool(scored["predicted_churn"]),
            risk_band=str(scored["risk_band"]),
            threshold=service.threshold,
            total_charges_estimated=estimated,
            reasons=reasons,
        )

    @app.post("/predict-batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(payload: BatchRequest, service: ChurnService = Depends(get_service)):
        df = pd.DataFrame(payload.customers)
        result = run_batch(df, service)
        if result.missing_columns:
            raise HTTPException(status_code=422, detail={
                "message": "Required fields are missing from the request.",
                "missing_fields": result.missing_columns})
        preds = []
        for rec in result.scored.to_dict(orient="records"):
            preds.append({"row": rec["row"], "customerID": rec.get("customerID"),
                          "churn_probability": rec["churn_probability"],
                          "predicted_churn": bool(rec["predicted_churn"]),
                          "risk_band": rec["risk_band"]})
        return BatchResponse(
            n_received=result.n_received, n_scored=result.n_scored,
            n_rejected=result.n_rejected,
            n_total_charges_estimated=result.n_estimated_total_charges,
            threshold=service.threshold, predictions=preds,
            errors=result.errors[:MAX_ERRORS_RETURNED],
            errors_truncated=len(result.errors) > MAX_ERRORS_RETURNED)

    return app


app = create_app()
