# Evaluation

Scenario-based evaluation runs scripted conversations against a pipeline and scores the
results, so a prompt change can be checked against something better than a hunch. Scoring
can be done by an LLM, by your own Python function, or by both.

AIAvatarKit includes a comprehensive evaluation framework for testing and assessing AI avatar conversations. The `DialogEvaluator` enables scenario-based conversation execution with automatic evaluation capabilities.

## Features

- **Scenario Execution**: Run predefined dialog scenarios against your AI system
- **Turn-by-Turn Evaluation**: Evaluate each conversation turn against specific criteria
- **Goal Assessment**: Evaluate overall scenario objective achievement
- **Result Management**: Save, load, and display evaluation results

## Basic Usage

```python
import asyncio
from aiavatar.eval.dialog import DialogEvaluator, Scenario, Turn
from aiavatar.sts.llm.chatgpt import ChatGPTService

async def main():
    # Initialize LLM services
    llm = ChatGPTService(openai_api_key="your_api_key")
    evaluation_llm = ChatGPTService(openai_api_key="your_api_key")
    
    # Create evaluator
    evaluator = DialogEvaluator(
        llm=llm,                    # LLM for conversation
        evaluation_llm=evaluation_llm  # LLM for evaluation
    )
    
    # Define scenario
    scenario = Scenario(
        name="Order tracking support",
        goal="Provide efficient and helpful customer service for order tracking inquiries",
        turns=[
            Turn(
                input_text="Hello, I need help with my order",
                evaluation_criteria="Responds politely and shows willingness to help"
            ),
            Turn(
                input_text="My order number is 12345",
                evaluation_criteria="Acknowledges the order number and proceeds appropriately"
            )
        ]
    )
    
    # Run evaluation
    results = await evaluator.run(
        dataset=[scenario],
        detailed=True,                # Enable turn-by-turn evaluation
        overwrite_execution=False,    # Skip if already executed
        overwrite_evaluation=False    # Skip if already evaluated
    )
    
    # Display results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(main())
```

Example Output:

```
=== Scenario 1 ===
Goal: Provide helpful customer support

Turn 1:
  Input: Hello, I need help with my order
  Actual Output: Hello! I'd be happy to help you with your order. Could you please provide your order number?
  Result: ✓ PASS
  Reason: The response is polite, helpful, and appropriately asks for the order number.

Turn 2:
  Input: My order number is 12345
  Actual Output: Thank you for providing order number 12345. Let me look that up for you.
  Result: ✓ PASS
  Reason: Acknowledges the order number and shows willingness to help.

Summary: 2/2 turns passed (100.0%)

=== Overall Scenario Evaluation ===
Goal Achievement: ✓ SUCCESS
Reason: The AI successfully provided helpful customer support by responding politely and efficiently handling the order inquiry.
```

## File-Based Evaluation

Load scenarios from JSON files:

```json
{
  "scenarios": [
    {
      "goal": "Basic greeting and assistance",
      "turns": [
        {
          "input_text": "Hello",
          "expected_output_text": "Friendly greeting",
          "evaluation_criteria": "Responds warmly and appropriately"
        }
      ]
    }
  ]
}
```

```python
# Load and evaluate from file
results = await evaluator.run(dataset="test_scenarios.json")

# Save results back to file
evaluator.save_results(results, "results.json")
```

## Configuration Options

```python
# Execution modes
results = await evaluator.run(
    dataset=scenarios,
    detailed=True,                # Turn-by-turn evaluation
    overwrite_execution=True,     # Re-run conversations
    overwrite_evaluation=True     # Re-evaluate results
)

# Simple mode (scenario-level evaluation only)
results = await evaluator.run(
    dataset=scenarios,
    detailed=False
)
```

## Running evaluations from the Admin Panel

`setup_admin_panel()` mounts the evaluation API alongside the rest of the panel, so scenarios
can be run against the live application over HTTP.

```python
from aiavatar.admin import setup_admin_panel
from aiavatar.eval.dialog import DialogEvaluator
from aiavatar.sts.llm.chatgpt import ChatGPTService

eval_llm = ChatGPTService(openai_api_key=OPENAI_API_KEY)
evaluator = DialogEvaluator(llm=aiavatar_app.sts.llm, evaluation_llm=eval_llm)

setup_admin_panel(app, adapter=aiavatar_app, evaluator=evaluator)
```

Passing `evaluator` is optional. When the pipeline's LLM is a `ChatGPTService`, the panel
builds a default evaluator from it — same credentials, model, and reasoning settings — so
evaluation works without any extra wiring. Supply your own when you want a different judge
model or custom evaluation functions.

Two endpoints appear under the panel's API prefix:

| Route | Purpose |
| --- | --- |
| `POST /admin/api/evaluate` | Starts an evaluation in the background and returns an `evaluation_id` |
| `GET /admin/api/evaluate/{evaluation_id}` | Returns the results once the run has finished |

Evaluation runs as a background task, so the POST returns immediately. Poll the GET endpoint
with the returned id. See [Administration](admin.md).

## Logic-based evaluation

In addition to LLM-based evaluation using `evaluation_criteria`, you can evaluate more explicitly using custom logic functions.

```python
# Make evaluation function(s)
def evaluate_weather_tool_call(output_text, tool_call, evaluation_criteria, result, eval_result_text):
    if tool_call is not None and tool_call.name != "get_weather":
        # Overwrite result and reason
        return False, f"Incorrect tool call: {tool_call.name}"
    else:
        # Pass through
        return result, eval_result_text

# Register evaluation function(s)
evaluator = DialogEvaluator(
    llm=aiavatar_app.sts.llm,
    evaluation_llm=eval_llm,
    evaluation_functions={"evaluate_weather_tool_call_func": evaluate_weather_tool_call}
)

# Use evaluation function in scenario
scenario = Scenario(
    turns=[
        Turn(input_text="Hello", expected_output_text="Hi", evaluation_criteria="Greeting"),
        Turn(input_text="What is the weather in Tokyo?", expected_output_text="It's sunny.", evaluation_criteria="Answer the weather based on the result of calling get_weather tool.", evaluation_function_name="evaluate_weather_tool_call_func"),
    ],
    goal="Answer the weather in Tokyo based on the result of get_weather."
)
```

## See also

- [Administration](admin.md) — running evaluations from the Admin Panel
- [LLM](llm.md) — the service under test
- [Guardrail](guardrail.md) — enforcing what evaluation measures

---

[← Documentation index](../README.md#-documentation)
