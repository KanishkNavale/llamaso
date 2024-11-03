import unittest

from pydantic import BaseModel, Field

from llamaso.inferencing_model import InferencingModel


class SampleSO(BaseModel):
    answer: int = Field(..., description="The answer to the question.")
    reasoning: str = Field(..., description="Provide reasoning for the answer.")


class TestDemo(unittest.TestCase):
    def setUp(self):
        self.llm = InferencingModel(
            repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="Llama-3.2-3B-Instruct-Q4_0.gguf",
            n_ctx=2048,
        )

    def test_infer(self):
        prompt = "What is a 3+3?"
        response = self.llm.infer(prompt=prompt, schema=SampleSO)

        # Assertions
        self.assertIsInstance(response, SampleSO)
        self.assertEqual(response.answer, 6)


if __name__ == "__main__":
    unittest.main()
