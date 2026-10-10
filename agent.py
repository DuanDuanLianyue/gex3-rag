"""Grounded university IT support agent backed by the supplied policy file."""

import argparse
import re

from langchain_ollama import ChatOllama

from rag_tool import search_university_it_policies

DEFAULT_QUESTION = "What is the exact phone number for the university IT Service Desk?"
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{6,}\d)(?!\d)")


def answer_question(question, model_name="qwen3:0.6b"):
    # An agent must consult the retrieval tool before it can formulate an answer.
    context = search_university_it_policies.invoke({"query": question})

    prompt = f"""You are a university IT support assistant.
Use only the POLICY CONTEXT below. Do not use prior knowledge and never invent a
phone number, URL, email address, policy, or procedure. If the requested fact is
not present, state clearly that it is not available in the supplied policies.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Give one concise answer."""

    response = ChatOllama(
        model=model_name,
        temperature=0,
        reasoning=False,
    ).invoke(prompt)
    answer = response.content.strip()

    # Deterministic grounding guard: a phone number may only be repeated when the
    # retrieved policy context itself contains a phone number.
    if "phone" in question.lower() and not PHONE_PATTERN.search(context):
        return (
            "The exact University IT Service Desk phone number is not available "
            "in the supplied university IT policies."
        )
    return answer


def main():
    parser = argparse.ArgumentParser(description="Run the university IT RAG agent.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default="qwen3:0.6b")
    args = parser.parse_args()
    print(answer_question(args.question, args.model))


if __name__ == "__main__":
    main()
