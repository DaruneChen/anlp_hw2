"""
reader.py — Generate answers using a HuggingFace LLM given retrieved context.

Default model: mistralai/Mistral-7B-Instruct-v0.2  (< 32B, released before Jan 2025)
Swap model_name for any other compliant HuggingFace model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


SYSTEM_PROMPT = """You are a helpful assistant that answers factual questions about Pittsburgh and CMU.
Use ONLY the provided context to answer. If the answer is not in the context, say "I don't know."
Keep your answer concise — one sentence or a short phrase when possible."""


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer:"""


class Reader:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 device: str = "auto", max_new_tokens: int = 128):
        print(f"[Reader] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        print("[Reader] Model loaded.")

    def answer(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_prompt(question, context_chunks)
        output = self.pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix, keep only new tokens
        answer = output[len(prompt):].strip()
        # Clean up — stop at newline
        answer = answer.split("\n")[0].strip()
        return answer


# ── Few-shot variant (no fine-tuning needed) ─────────────────────────────────

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
        examples += f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    context = "\n\n".join(context_chunks[:3])
    return f"""{SYSTEM_PROMPT}

{examples}Context:
{context}

Question: {question}
Answer:"""


class FewShotReader(Reader):
    def answer(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_few_shot_prompt(question, context_chunks)
        output = self.pipe(prompt)[0]["generated_text"]
        answer = output[len(prompt):].strip().split("\n")[0].strip()
        return answer


if __name__ == "__main__":
    # Quick test with a dummy context (no GPU required for testing prompt logic)
    reader = FewShotReader()
    q = "What is the name of the annual pickle festival held in Pittsburgh?"
    ctx = [
        "Picklesburgh is an annual food festival held in Pittsburgh, Pennsylvania, celebrating all things pickle.",
        "The festival typically takes place in July on the Roberto Clemente Bridge downtown."
    ]
    print(f"Q: {q}")
    print(f"A: {reader.answer(q, ctx)}")
