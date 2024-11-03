import json
from typing import Dict, List, Optional, Type

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar
from pydantic import BaseModel


def compile_schema_grammar(
    schema: Type[BaseModel],
) -> Optional[LlamaGrammar]:
    valid_schema: Optional[str] = None

    if issubclass(schema, BaseModel):
        valid_schema = json.dumps(schema.model_json_schema(), indent=2)

    return LlamaGrammar.from_json_schema(valid_schema) if valid_schema else None


def trim_contexts(
    contexts: List[Dict[str, str]],
    llama_instance: Llama,
    future_msg: Dict[str, str],
) -> List[Dict[str, str]]:
    model_context_len = llama_instance.n_ctx()
    tokenizer = llama_instance.tokenizer()

    comp_ctx_len = model_context_len - len(tokenizer.encode(json.dumps(future_msg)))
    total_ctx_len = len(tokenizer.encode(json.dumps(contexts)))

    if total_ctx_len < comp_ctx_len:
        return contexts

    while total_ctx_len > comp_ctx_len:
        if contexts and contexts[0]["role"] == "user":
            contexts.pop(0)

            if contexts and contexts[0]["role"] == "assistant":
                contexts.pop(0)

        total_ctx_len = len(tokenizer.encode(json.dumps(contexts)))

    return contexts
