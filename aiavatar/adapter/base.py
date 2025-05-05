from abc import ABC, abstractmethod
from ..sts.models import STSResponse
from ..sts.pipeline import STSPipeline


class Adapter(ABC):
    def __init__(self, sts: STSPipeline):
        self.sts = sts
        self.sts.handle_response = self.handle_response
        self.sts.stop_response = self.stop_response

    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, session_id: str, context_id: str):
        pass
