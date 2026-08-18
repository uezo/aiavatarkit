from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Dict, Optional


class ControlTagConfigResolver:
    """Resolve a control-tag identifier from an application-owned config map.

    Attributes listed in ``protected_overrides`` retain their configured values
    when an identifier is resolved. Tags without an identifier pass through.
    """

    def __init__(
        self,
        configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        key_attribute: str = "id",
        protected_overrides: Optional[Iterable[str]] = None,
    ):
        self.key_attribute = key_attribute.lower()
        if protected_overrides is None:
            protected_overrides = {"type", "src"}
        elif isinstance(protected_overrides, str):
            raise TypeError("Protected overrides must be an iterable of attribute names")
        self.protected_overrides = frozenset(
            self._normalize_attribute_name(name)
            for name in protected_overrides
        )
        self._configs = {}
        self.set_configs(configs or {})

    def set_configs(self, configs: Mapping[str, Mapping[str, Any]]):
        self._configs = self._normalize_configs(configs)

    def update_configs(self, configs: Mapping[str, Mapping[str, Any]]):
        self._configs = {
            **self._configs,
            **self._normalize_configs(configs),
        }

    def set_config(self, config_id: str, attributes: Mapping[str, Any]):
        self.update_configs({config_id: attributes})

    @staticmethod
    def _normalize_attribute_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("Protected override names must be non-empty strings")
        return name.lower()

    @staticmethod
    def _normalize_configs(configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(configs, Mapping):
            raise TypeError("Control tag configs must be a mapping")

        normalized = {}
        for config_id, attributes in configs.items():
            if not isinstance(config_id, str) or not config_id:
                raise ValueError("Control tag config IDs must be non-empty strings")
            if not isinstance(attributes, Mapping):
                raise TypeError(f"Control tag config must be a mapping: {config_id}")
            normalized[config_id] = deepcopy(dict(attributes))
        return normalized

    def __call__(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        config_id = attributes.get(self.key_attribute)
        if config_id is None:
            return dict(attributes)
        if not isinstance(config_id, str) or not config_id:
            raise ValueError(f"{self.key_attribute} must be a non-empty string")

        configured = self._configs.get(config_id)
        if configured is None:
            raise ValueError(f"Unknown control tag config ID: {config_id}")

        resolved = deepcopy(configured)
        resolved.pop(self.key_attribute, None)
        resolved.update({
            name: value
            for name, value in attributes.items()
            if name != self.key_attribute and name not in self.protected_overrides
        })
        return resolved
