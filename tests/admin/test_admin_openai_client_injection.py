from types import SimpleNamespace

import aiavatar.admin as admin_module


def test_default_evaluator_reuses_preconfigured_openai_client(monkeypatch):
    created = []

    class FakeChatGPTService:
        def __init__(
            self,
            *,
            openai_client=None,
            model="gpt-5.4",
            temperature=None,
            reasoning_effort=None,
        ):
            self.openai_client = openai_client
            self.model = model
            self.temperature = temperature
            self.reasoning_effort = reasoning_effort
            self.system_prompt = None
            created.append(self)

    monkeypatch.setattr(admin_module, "ChatGPTService", FakeChatGPTService)
    client = object()
    source = FakeChatGPTService(
        openai_client=client,
        model="azure-deployment-name",
        temperature=0.2,
        reasoning_effort="low",
    )
    adapter = SimpleNamespace(sts=SimpleNamespace(llm=source))

    evaluator = admin_module._default_evaluator(adapter)

    assert evaluator.llm is source
    assert evaluator.evaluation_llm is created[1]
    assert evaluator.evaluation_llm.openai_client is client
    assert evaluator.evaluation_llm.model == "azure-deployment-name"
