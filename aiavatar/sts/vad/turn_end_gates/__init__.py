from .base import TurnEndDecision, TurnEndGate


def __getattr__(name):
    if name == "SmartTurnEndGate":
        from .smart_turn import SmartTurnEndGate
        return SmartTurnEndGate
    if name == "NamoTurnEndGate":
        from .namo_turn import NamoTurnEndGate
        return NamoTurnEndGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TurnEndDecision", "TurnEndGate", "SmartTurnEndGate", "NamoTurnEndGate"]
