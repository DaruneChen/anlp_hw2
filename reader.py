"""
reader.py — LLM reader for RAG.

Goals:
- concise factual answers for overlap metrics
- avoid hallucinations: if not in context => "unknown"
- CPU-safe loading (no fp16 on CPU)
"""

from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


SYSTEM_PROMPT = """You are a precise factual QA system for Pittsburgh and Carnegie Mellon University.

Rules:
- Use ONLY the provided context.
- Answer as briefly as possible (name/date/number/short phrase).
- Do NOT explain.
- If the answer is not explicitly stated in the context, output: unknown
"""


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer:"""


class Reader:
    def __init__(self, model_name: str, max_new_tokens: int = 64):
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
            device=device,
        )
        print("[Reader] Model loaded.")

    def answer(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_prompt(question, context_chunks)
        out = self.pipe(prompt)[0]["generated_text"]
        ans = out[len(prompt):].strip()

        ans = ans.split("\n", 1)[0].strip()
        ans = ans.strip('"').strip("'").strip()

        if ans.lower().startswith("answer:"):
            ans = ans.split(":", 1)[1].strip()

        if not ans:
            return "unknown"

        # prevent multi-sentence rambling
        if "." in ans and len(ans.split()) > 6:
            ans = ans.split(".", 1)[0].strip()

        return ans


FEW_SHOT_EXAMPLES = [
    {
        "question": "Who is Pittsburgh named after?",
        "context": "Pittsburgh was named in 1758 after William Pitt, the British Secretary of State.",
        "answer": "William Pitt",
    },
    {
        "question": "When was Carnegie Mellon University founded?",
        "context": "Carnegie Mellon University was founded in 1900 by Andrew Carnegie as the Carnegie Technical Schools.",
        "answer": "1900",
    },
]


def build_few_shot_prompt(question: str, context_chunks: list[str]) -> str:
    examples = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples += (
            f"Context: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n\n"
        )

    context = "\n\n".join(context_chunks)
    return f"""{SYSTEM_PROMPT}

{examples}Context:
{context}

Question: {question}
Answer:"""


class FewShotReader(Reader):
    def answer(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_few_shot_prompt(question, context_chunks)
        out = self.pipe(prompt)[0]["generated_text"]
        ans = out[len(prompt):].strip().split("\n", 1)[0].strip()
        ans = ans.strip('"').strip("'").strip()
        if ans.lower().startswith("answer:"):
            ans = ans.split(":", 1)[1].strip()
        return ans if ans else "unknown"