import json

from pydantic import BaseModel, Field

from llamaso.inferencing_model import InferencingModel


# Pydantic model as a Structured Output Template
class SomeStructuredOutput(BaseModel):
    answer: int = Field(..., description="The answer to the question.")
    reasoning: str = Field(..., description="Provide reasoning for the answer.")


# Load the LlamaCPP model
LLAMA_MODEL = InferencingModel(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_0.gguf",
    n_ctx=2048,
)


if __name__ == "__main__":
    prompt = "What is a 3+3?"

    response = LLAMA_MODEL.infer(prompt=prompt, schema=SomeStructuredOutput)

    if isinstance(response, SomeStructuredOutput):
        print(json.dumps(response.model_dump(), indent=2))
