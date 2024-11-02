# "llamaso": LlamaCPP for Structured Output

This repository, "llamaso", demonstrates a simple implementation for getting a structured output from a large language model framework based on ["LlamaCPP"](https://github.com/ggerganov/llama.cpp).

Moreover, this repository uses "LlamaGrammar" and pydantic "BaseModel" to get structured output.

## Installing this Repository

Run the following commands for installation in the shell,

```bash
pip install poetry  # Installs poetry
poetry install      # Installs this package
```

## Example

The following demo is from the ["demo.py"](/demo.py),

```python
import json

from pydantic import BaseModel, Field

from llamaso.inferencing_model import InferencingModel


# Pydantic model as a Structured Output (SO) Template
class SomeStructuredOutput(BaseModel):
    answer: int = Field(..., description="The answer to the question.")
    reasoning: str = Field(..., description="Provide reasoning for the answer.")


# Load the LlamaCPP model
LLAMA_MODEL = InferencingModel(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",  # Swappable LlamaCPP compatible model from
    filename="Llama-3.2-3B-Instruct-Q4_0.gguf",      # Huggingface
    n_ctx=2048,
)


if __name__ == "__main__":
    prompt = "What is a 3+3?"

    response = LLAMA_MODEL.infer(prompt=prompt, schema=SomeStructuredOutput) # Parse SO class here!

    if isinstance(response, SomeStructuredOutput):
        print(json.dumps(response.model_dump(), indent=2))

```

Output:

```bash
Loading inferencing model...
{
  "answer": 6,
  "reasoning": "Basic addition operation"
}
```
