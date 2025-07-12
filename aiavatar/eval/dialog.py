from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from uuid import uuid4
from ..sts.llm import LLMService


# Constants
EVALUATION_SUCCESS_MARKER = "[result:true]"
DEFAULT_EVALUATION_SYSTEM_PROMPT = (
    "Compare the output content and evaluate whether the output is appropriate "
    "based on the evaluation criteria. If appropriate, output [result:true]. "
    "Regardless of whether it's appropriate or not, provide a reason for your evaluation."
)
DEFAULT_USER_ID = "eval_user"
DEFAULT_TURN_USER_ID = "turn_user"
DEFAULT_SCENARIO_USER_ID = "scenario_eval_user"


# Data Classes
@dataclass
class EvaluationResult:
    result: bool
    reason: str


@dataclass
class Turn:
    input_text: Optional[str] = None
    expected_output_text: Optional[str] = None
    evaluation_criteria: Optional[str] = None
    actual_output_text: Optional[str] = None
    evaluation_result: Optional[EvaluationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "input_text": self.input_text,
            "expected_output": self.expected_output_text,
            "evaluation_criteria": self.evaluation_criteria,
            "actual_output": self.actual_output_text
        }
        if self.evaluation_result:
            data["evaluation_result"] = {
                "result": self.evaluation_result.result,
                "reason": self.evaluation_result.reason
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        turn = cls(
            input_text=data.get("input_text"),
            expected_output_text=data.get("expected_output"),
            evaluation_criteria=data.get("evaluation_criteria"),
            actual_output_text=data.get("actual_output")
        )
        if "evaluation_result" in data:
            eval_data = data["evaluation_result"]
            turn.evaluation_result = EvaluationResult(
                result=eval_data["result"],
                reason=eval_data["reason"]
            )
        return turn


@dataclass
class Scenario:
    turns: List[Turn] = field(default_factory=list)
    goal: Optional[str] = None
    scenario_evaluation_result: Optional[EvaluationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "goal": self.goal,
            "turns": [turn.to_dict() for turn in self.turns]
        }
        if self.scenario_evaluation_result:
            data["scenario_evaluation_result"] = {
                "result": self.scenario_evaluation_result.result,
                "reason": self.scenario_evaluation_result.reason
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        turns = [Turn.from_dict(turn_data) for turn_data in data.get("turns", [])]
        scenario = cls(turns=turns, goal=data.get("goal"))
        if "scenario_evaluation_result" in data:
            eval_data = data["scenario_evaluation_result"]
            scenario.scenario_evaluation_result = EvaluationResult(
                result=eval_data["result"],
                reason=eval_data["reason"]
            )
        return scenario
    
    def has_execution_results(self) -> bool:
        """Check if this scenario already has execution results"""
        return all(turn.actual_output_text is not None for turn in self.turns)


# Validation
class ValidationError(Exception):
    pass


class DataValidator:
    @staticmethod
    def validate_turn(turn: Turn) -> None:
        if not turn.input_text:
            raise ValidationError("Turn must have input_text")
    
    @staticmethod
    def validate_scenario(scenario: Scenario) -> None:
        if not scenario.turns:
            raise ValidationError("Scenario must have at least one turn")
        
        for i, turn in enumerate(scenario.turns):
            try:
                DataValidator.validate_turn(turn)
            except ValidationError as ve:
                raise ValidationError(f"Turn {i+1}: {ve}")
    
    @staticmethod
    def validate_scenarios(scenarios: List[Scenario]) -> None:
        if not scenarios:
            raise ValidationError("Dataset must contain at least one scenario")
        
        for i, scenario in enumerate(scenarios):
            try:
                DataValidator.validate_scenario(scenario)
            except ValidationError as ve:
                raise ValidationError(f"Scenario {i+1}: {ve}")


# Evaluator
class DialogEvaluator:
    def __init__(self, llm: LLMService, evaluation_llm: LLMService = None):
        self.llm = llm
        self.evaluation_llm = evaluation_llm
        if self.evaluation_llm and not self.evaluation_llm.system_prompt:
            self.evaluation_llm.system_prompt = DEFAULT_EVALUATION_SYSTEM_PROMPT

    async def get_llm_response(self, llm: LLMService, context_id: str, user_id: str, text: str) -> str:
        result_text = ""
        try:
            async for resp in llm.chat_stream(
                context_id=context_id, 
                user_id=user_id, 
                text=text
            ):
                result_text += resp.text
            return result_text
        except Exception as ex:
            print(f"Error during turn processing: {ex}")
            raise

    async def process_turn(self, context_id: str, turn: Turn) -> str:
        if not self.llm:
            raise ValidationError("LLM service is required for processing turns")
        
        DataValidator.validate_turn(turn)
        actual_output_text = await self.get_llm_response(
            llm=self.llm,
            context_id=context_id, 
            user_id=DEFAULT_TURN_USER_ID, 
            text=turn.input_text
        )

        return actual_output_text

    async def process_scenario(self, scenario: Scenario):
        context_id = str(uuid4())

        for i, turn in enumerate(scenario.turns, 1):
            print(f"\rProcessing turn {i}/{len(scenario.turns)}: {turn.input_text[:50]}...", end="", flush=True)
            try:
                turn.actual_output_text = await self.process_turn(context_id, turn)
            except Exception as ex:
                print(f"\nError in turn {i}: {ex}. Stopping scenario processing.")
                raise
        print()

    async def evaluate_turn_output(self, output_text: str, evaluation_criteria: str) -> EvaluationResult:
        if not output_text or not evaluation_criteria:
            raise ValidationError("Both output_text and evaluation_criteria are required")
        
        eval_input_text = f"## Output\n{output_text}\n\n## Evaluation Criteria\n{evaluation_criteria}"
        eval_result_text = await self.get_llm_response(
            llm=self.evaluation_llm,
            context_id=str(uuid4()), 
            user_id=DEFAULT_USER_ID, 
            text=eval_input_text
        )

        result = EVALUATION_SUCCESS_MARKER in eval_result_text.lower()
        return EvaluationResult(result, eval_result_text)

    async def evaluate_scenario_goal(self, scenario: Scenario) -> EvaluationResult:
        DataValidator.validate_scenario(scenario)
        
        if not scenario.goal:
            raise ValidationError("Scenario must have a goal for evaluation")
        
        conversation_text = ""
        for turn in scenario.turns:
            if not turn.actual_output_text:
                raise ValidationError("All turns must have actual_output_text for scenario evaluation")
            conversation_text += f"User: {turn.input_text}\nAssistant: {turn.actual_output_text}\n\n"
        
        eval_input_text = (
            f"## Full Conversation\n{conversation_text}\n## Goal\n{scenario.goal}\n\n"
            "Evaluate whether the goal was achieved based on the full conversation above."
        )
        eval_result_text = await self.get_llm_response(
            self.evaluation_llm,
            context_id=str(uuid4()), 
            user_id=DEFAULT_SCENARIO_USER_ID, 
            text=eval_input_text
        )

        result = EVALUATION_SUCCESS_MARKER in eval_result_text.lower()
        return EvaluationResult(result, eval_result_text)

    async def run(self, *, dataset: Union[List[Scenario], str], detailed: bool = True, overwrite_execution: bool = False, overwrite_evaluation: bool = False) -> List[Scenario]:
        # Load dataset from file if string path is provided
        if isinstance(dataset, str):
            filepath = dataset
            try:
                dataset = self.load_results(filepath)
                print(f"Loaded {len(dataset)} scenarios from {filepath}")
            except Exception as ex:
                print(f"Failed to load dataset from {filepath}: {ex}")
                raise
        
        # Validate dataset
        try:
            DataValidator.validate_scenarios(dataset)
        except ValidationError as ve:
            print(f"Dataset validation failed: {ve}")
            raise
        
        scenarios = []
        total_scenarios = len(dataset)
        
        for scenario_idx, scenario in enumerate(dataset, 1):
            try:
                if scenario.has_execution_results() and not overwrite_execution:
                    print(f"[{scenario_idx}/{total_scenarios}] Use pre-executed scenario: {scenario.goal}")
                else:
                    print(f"[{scenario_idx}/{total_scenarios}] Processing scenario: {scenario.goal}")
                    await self.process_scenario(scenario)
            except Exception as ex:
                print(f"[{scenario_idx}/{total_scenarios}] Scenario failed: {ex}. Continuing with next scenario.")
                continue

            if self.evaluation_llm:
                # Evaluate each turn (only in detailed mode)
                if detailed:
                    for i, turn in enumerate(scenario.turns):
                        if turn.evaluation_criteria:
                            # Check if this turn already has evaluation result
                            if turn.evaluation_result is not None and not overwrite_evaluation:
                                continue  # Skip this turn evaluation

                            print(f"\rEvaluating turn {i+1}/{len(scenario.turns)}: {turn.input_text[:50]}...", end="", flush=True)
                            try:
                                turn.evaluation_result = await self.evaluate_turn_output(
                                    output_text=turn.actual_output_text,
                                    evaluation_criteria=turn.evaluation_criteria
                                )
                            except Exception as ex:
                                print(f"\nTurn {i+1} evaluation failed: {ex}. Continuing with next turn.")
                                continue
                    if detailed and any(turn.evaluation_criteria for turn in scenario.turns):
                        print()

                # Evaluate overall scenario
                if scenario.scenario_evaluation_result is None or overwrite_evaluation:
                    print("Evaluating overall scenario...")
                    try:
                        scenario.scenario_evaluation_result = await self.evaluate_scenario_goal(scenario)
                    except Exception as ex:
                        print(f"Scenario evaluation failed: {ex}. Continuing with next scenario.")

            scenarios.append(scenario)
            print(f"✓ Scenario {scenario_idx} completed")

        print(f"✓ All {total_scenarios} scenario(s) completed!")
        return scenarios

    def print_results(self, scenarios: List[Scenario], detailed: bool = True):
        for i, scenario in enumerate(scenarios):
            print(f"\n=== Scenario {i+1} ===")
            print(f"Goal: {scenario.goal}")
            
            if detailed:
                evaluated_turns = [turn for turn in scenario.turns if turn.evaluation_result]
                
                if evaluated_turns:
                    for j, turn in enumerate(scenario.turns):
                        if turn.evaluation_result:
                            print(f"\nTurn {j+1}:")
                            print(f"  Input: {turn.input_text}")
                            print(f"  Expected Output: {turn.expected_output_text}")
                            print(f"  Actual Output: {turn.actual_output_text}")
                            print(f"  Evaluation Criteria: {turn.evaluation_criteria}")
                            print(f"  Result: {'✓ PASS' if turn.evaluation_result.result else '✗ FAIL'}")
                            print(f"  Reason: {turn.evaluation_result.reason}")
                    
                    # Turn evaluation summary
                    passed = sum(1 for turn in evaluated_turns if turn.evaluation_result.result)
                    total = len(evaluated_turns)
                    print(f"\nSummary: {passed}/{total} turns passed ({passed/total*100:.1f}%)")
            
            # Overall scenario evaluation
            if scenario.scenario_evaluation_result:
                print(f"\n=== Overall Scenario Evaluation ===")
                print(f"Goal Achievement: {'✓ SUCCESS' if scenario.scenario_evaluation_result.result else '✗ FAILED'}")
                print(f"Reason: {scenario.scenario_evaluation_result.reason}")

    def save_results(self, scenarios: List[Scenario], filepath: str):
        data = {
            "timestamp": datetime.now().isoformat(),
            "scenarios": [scenario.to_dict() for scenario in scenarios]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_results(filepath: str) -> List[Scenario]:
        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            raise IOError(f"Failed to read file {filepath}: {ex}")

        if "scenarios" not in data:
            raise ValueError("Dataset file must contain 'scenarios' key")
        
        scenarios = []
        try:
            for scenario_data in data["scenarios"]:
                scenario = Scenario.from_dict(scenario_data)
                scenarios.append(scenario)
        except Exception as ex:
            raise ValueError(f"Failed to parse scenario data: {ex}")
        
        return scenarios
