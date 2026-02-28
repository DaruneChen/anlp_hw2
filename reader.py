"""
reader.py — Improved LLM reader for CMU RAG assignment.

Goals:
- maximize factual extraction accuracy
- minimize hallucinations
- produce concise answers for F1 / ROUGE metrics
- assignment-compliant (HuggingFace open models only)
"""

from __future__ import annotations

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


SYSTEM_PROMPT = """You are a factual QA system for Pittsburgh and Carnegie Mellon University.

CRITICAL RULES:
- Use ONLY the provided context
- Answer in as few words as possible
- Prefer names, dates, numbers, or short phrases
- Do NOT explain
- Do NOT include extra words
- If answer not present, output: unknown
"""


FEW_SHOT_EXAMPLES = [
    {
        "context":
        "Pittsburgh was named in 1758 after William Pitt, the British Secretary of State.",
        "question":
        "Who is Pittsburgh named after?",
        "answer":
        "William Pitt",
    },
    {
        "context":
        "Carnegie Mellon University was founded in 1900 by Andrew Carnegie.",
        "question":
        "When was Carnegie Mellon University founded?",
        "answer":
        "1900",
    },
]


def clean_answer(ans: str) -> str:
    """
    Aggressive cleaning to improve leaderboard metrics.
    """
    ans = ans.strip()

    ans = ans.split("\n")[0]

    ans = ans.strip('"').strip("'")

    ans = re.sub(r"\.$", "", ans)

    if "," in ans and len(ans.split()) > 4:
        ans = ans.split(",")[0]

    words = ans.split()
    if len(words) > 10:
        ans = " ".join(words[:10])

    ans = ans.strip()

    if not ans:
        return "unknown"

    return ans


def build_prompt(question: str, context_chunks: list[str]) -> str:

    examples = ""

    for ex in FEW_SHOT_EXAMPLES:
        examples += (
            f"Context: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n\n"
        )

    context = "\n\n".join(context_chunks)

    prompt = f"""{SYSTEM_PROMPT}

{examples}
Context:
{context}

Question: {question}

Answer:"""

    return prompt


class Reader:

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 32,
    ):

        print(f"[Reader] Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        use_cuda = torch.cuda.is_available()

        dtype = torch.float16 if use_cuda else torch.float32

        device_map = "auto" if use_cuda else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

        device = 0 if use_cuda else -1

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            device=device,
        )

        print("[Reader] Ready.")

    def answer(
        self,
        question: str,
        context_chunks: list[str],
    ) -> str:

        if not context_chunks:
            return "unknown"

        prompt = build_prompt(question, context_chunks)

        output = self.pipe(prompt)[0]["generated_text"]

        answer = output[len(prompt):]

        answer = clean_answer(answer)

        return answer