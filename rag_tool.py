"""LangChain tool that exposes only the supplied university policy data."""

from langchain_core.tools import tool

from policy_loader import load_policy_documents
from rag import PolicyRetriever, format_context


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

retriever = PolicyRetriever(documents)


@tool
def search_university_it_policies(query: str) -> str:
    """Search the supplied university IT policies for facts relevant to a query."""
    return format_context(retriever.retrieve(query, k=5))
