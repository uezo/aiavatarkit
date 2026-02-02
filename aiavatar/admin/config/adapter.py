import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from ...adapter.base import Adapter

logger = logging.getLogger(__name__)


class AdapterConfig(BaseModel):
    """Dynamic config model that only contains fields returned by each adapter's get_config()."""

    class Config:
        extra = "allow"


class AdapterConfigResponse(BaseModel):
    name: str = Field(
        description="Name of the adapter"
    )
    type: str = Field(
        description="Type of adapter component"
    )
    config: AdapterConfig = Field(
        description="Configuration of adapter component"
    )


class UpdateAdapterConfigRequest(BaseModel):
    config: AdapterConfig = Field(
        description="Configuration of adapter component"
    )


class AdapterConfigAPI:
    def __init__(
        self,
        adapters: Dict[str, Adapter]
    ):
        self.adapters = adapters

    def get_router(self):
        router = APIRouter()

        @router.get(
            "/config/adapters",
            tags=["Config"],
            summary="List all adapter configurations",
            description="Retrieve the current configuration settings for all registered adapters",
            response_description="List of adapter configurations",
            responses={
                200: {"description": "Successfully retrieved adapter configurations"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_adapters() -> List[AdapterConfigResponse]:
            try:
                return [
                    AdapterConfigResponse(
                        name=name,
                        type=adapter.__class__.__name__,
                        config=AdapterConfig(**adapter.get_config()),
                    )
                    for name, adapter in self.adapters.items()
                ]

            except Exception as ex:
                logger.error(f"Error retrieving adapter configurations: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving adapter configurations"
                )

        @router.get(
            "/config/adapter/{name}",
            tags=["Config"],
            summary="Get adapter configuration",
            description="Retrieve the current configuration settings for a specific adapter",
            response_description="Current adapter configuration",
            responses={
                200: {"description": "Successfully retrieved adapter configuration"},
                404: {"description": "Adapter not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def get_config_adapter(name: str) -> AdapterConfigResponse:
            try:
                adapter = self.adapters.get(name)
                if adapter is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Adapter '{name}' not found"
                    )

                return AdapterConfigResponse(
                    name=name,
                    type=adapter.__class__.__name__,
                    config=AdapterConfig(**adapter.get_config()),
                )

            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error retrieving adapter configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving adapter configuration"
                )

        @router.post(
            "/config/adapter/{name}",
            tags=["Config"],
            summary="Update adapter configuration",
            description="Update configuration settings for a specific adapter. Only non-null values will be updated.",
            response_description="Dictionary of successfully updated configuration parameters",
            responses={
                200: {"description": "Successfully updated adapter configuration"},
                404: {"description": "Adapter not found"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_config_adapter(name: str, request: UpdateAdapterConfigRequest) -> Dict[str, Any]:
            try:
                adapter = self.adapters.get(name)
                if adapter is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Adapter '{name}' not found"
                    )

                return adapter.set_config(request.config.model_dump())

            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error updating adapter configuration: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while updating adapter configuration"
                )

        return router
