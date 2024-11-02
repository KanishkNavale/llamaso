import json
from typing import Dict, List, Optional, Type, Union

import json_repair as jr
from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar
from pydantic import BaseModel

from llamaso.utils import trim_contexts


class InferencingModel:
    def __init__(
        self,
        repo_id: str,
        filename: str,
        n_ctx: int,
        n_gpu_layers: int = -1,
        model_dir: str = ".models",
        verbose: bool = False,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model_dir = model_dir
        self.model: Optional[Llama] = None
        self.verbose = verbose

        self._messages: List[Dict[str, str]] = []

    def _load_inferencing_model(self) -> Llama:
        if not isinstance(self.model, Llama):
            print("Loading inferencing model...")
            self.model = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                model_dir=self.model,
                verbose=self.verbose,
            )
            return self.model

        return self.model

    def _add_message(self, message: Dict[str, str]) -> None:
        self._messages = trim_contexts(
            contexts=self._messages,
            llama_instance=self._load_inferencing_model(),
            future_msg=message,
        )
        self._messages.append(message)

    def _compile_schema_grammar(
        self,
        schema: Type[BaseModel],
    ) -> Optional[LlamaGrammar]:
        valid_schema: Optional[str] = None

        if issubclass(schema, BaseModel):
            valid_schema = json.dumps(schema.model_json_schema(), indent=2)

        return LlamaGrammar.from_json_schema(valid_schema) if valid_schema else None

    def infer(
        self, prompt: str, schema: Optional[Type[BaseModel]]
    ) -> Union[str, BaseModel]:
        self._add_message({"role": "user", "content": prompt})

        if schema:
            schema_grammar = self._compile_schema_grammar(schema=schema)

        llm_response = self.model.create_chat_completion(  # type: ignore
            messages=self._messages,  # type: ignore
            grammar=schema_grammar if schema_grammar else None,
        )["choices"][0]["message"]

        self._add_message(llm_response)  # type: ignore

        content = llm_response["content"]

        if not isinstance(content, str):
            return "Error: Couldn't get a valid response."

        if schema and issubclass(schema, BaseModel):
            try:
                content = jr.loads(content)

                if isinstance(content, list):
                    content = content[-1]

                if isinstance(content, Dict):
                    response = schema(**content)
                    return response

            except Exception:
                raise ValueError("Couldn't parse the response as the schema provided.")

        return content
