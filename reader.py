"""
reader.py — optimized reader for CMU HW2 RAG

Fixes:
• prevents prompt from being returned as answer
• removes hidden max_length truncation
• produces concise factual answers
• compatible with pipeline.py (FewShotReader alias)
"""

from __future__ import annotations

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


class Reader:

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 64,
    ):

        print(f"[Reader] Loading {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # CRITICAL FIX — remove hidden truncation limit
        self.model.generation_config.max_length = None
        self.model.generation_config.max_new_tokens = max_new_tokens

        self.max_new_tokens = max_new_tokens

        print("[Reader] Ready")


    def generate_answer(
        self,
        query: str,
        contexts: list[str],
    ) -> str:

        if not contexts:
            return "unknown"

        context = "\n\n".join(contexts[:10])

        prompt = f"""
Answer the question using ONLY the context.

Return ONLY the short answer phrase.
Do NOT explain.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # decode ONLY generated tokens
        new_tokens = outputs[0][input_length:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        # truncate safely
        answer = re.split(r"[.\n]", answer)[0]

        answer = " ".join(answer.split()[:12])

        answer = answer.strip()

        if len(answer) == 0:
            return "unknown"

        return answer


    # pipeline.py calls this function
    def answer(
        self,
        query: str,
        contexts: list[str],
    ) -> str:

        return self.generate_answer(query, contexts)


# REQUIRED alias for pipeline.py
FewShotReader = Reader