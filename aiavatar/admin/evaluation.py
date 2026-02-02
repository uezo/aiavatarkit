from datetime import datetime
import logging
import os
from typing import List
import uuid
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from ..eval.dialog import DialogEvaluator, Scenario
from .auth import create_api_key_dependency

logger = logging.getLogger(__name__)


class EvaluationRequest(BaseModel):
    scenarios: List[Scenario] = Field(description="List of evaluation scenarios")


class EvaluationResponse(BaseModel):
    scenarios: List[Scenario] = Field(description="List of evaluation scenarios including results")


class EvaluationStartResponse(BaseModel):
    evaluation_id: str = Field(description="Id to retrieve evaluation results")


class EvaluationAPI:
    def __init__(
        self,
        evaluator: DialogEvaluator = None
    ):
        self.evaluator = evaluator

    def get_router(self):
        router = APIRouter()

        async def run_evaluation_task(evaluation_id: str, scenarios: List[Scenario]):
            try:
                results = await self.evaluator.run(dataset=scenarios, detailed=True)
                os.makedirs("evaluation_results", exist_ok=True)
                self.evaluator.save_results(results, f"evaluation_results/{evaluation_id}.json")
            except Exception as ex:
                logger.error(f"Evaluation failed {evaluation_id}: {ex}")

        @router.post(
            "/evaluate",
            response_model=EvaluationStartResponse,
            tags=["Evaluation"],
            summary="Start dialog evaluation",
            description="Start dialog evaluation in the background task. You can retrieve the results with evaluation_id after evaluation finished.",
            response_description="Id to retrieve evaluation results.",
            responses={
                200: {"description": "Successfully start evaluation"},
                500: {"description": "Internal server error"}
            }
        )
        async def evaluate(
            request: EvaluationRequest,
            background_tasks: BackgroundTasks
        ) -> EvaluationStartResponse:
            if not self.evaluator:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Evaluator is not set"
                )

            evaluation_id = None
            try:
                evaluation_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
                background_tasks.add_task(run_evaluation_task, evaluation_id, request.scenarios)
                return EvaluationStartResponse(evaluation_id=evaluation_id)
            except Exception as ex:
                logger.error(f"Starting evaluation failed {evaluation_id}: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while starting evaluation"
                )

        @router.get(
            "/evaluate/{evaluation_id}",
            tags=["Evaluation"],
            summary="Get dialog evaluation results",
            description="Get dialog evaluation results with evaluation_id.",
            response_description="List of evaluation scenario including results.",
            responses={
                200: {"description": "Successfully returns evaluation results"},
                404: {"description": "Results not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_evaluation_status(
            evaluation_id: str
        ) -> EvaluationResponse:
            if not self.evaluator:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Evaluator is not set"
                )

            filepath = f"evaluation_results/{evaluation_id}.json"
            if not os.path.exists(filepath):
                raise HTTPException(status_code=404, detail="Evaluation not found")
            try:
                results = self.evaluator.load_results(filepath)
                return EvaluationResponse(scenarios=results)
            except Exception as ex:
                logger.error(f"Retrieving evaluation results failed {evaluation_id}: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving evaluation results"
                )

        return router


def setup_evaluation_api(
    app: FastAPI,
    *,
    evaluator: DialogEvaluator = None,
    api_key: str = None
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []
    app.include_router(EvaluationAPI(evaluator=evaluator).get_router(), dependencies=deps)
