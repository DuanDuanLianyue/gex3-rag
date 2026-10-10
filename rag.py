"""Small, local BM25 retriever for the supplied policy knowledge base."""

import re

from rank_bm25 import BM25Okapi

from policy_loader import load_policy_documents


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


def tokenize(text):
    """Return normalized word tokens suitable for BM25 matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


class PolicyRetriever:
    def __init__(self, policy_documents=documents):
        self.documents = list(policy_documents)
        self.index = BM25Okapi([tokenize(doc.page_content) for doc in self.documents])

    def retrieve(self, query, k=5):
        """Return the highest-scoring policies, excluding zero-score matches."""
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for index in ranked:
            if scores[index] <= 0 or len(results) >= k:
                break
            results.append((self.documents[index], float(scores[index])))
        return results


def format_context(results):
    if not results:
        return "No relevant policy was found in the supplied knowledge base."
    sections = []
    for document, score in results:
        sections.append(
            "[{} | topic={} | score={:.3f}]\n{}".format(
                document.metadata["policy_id"],
                document.metadata["topic"],
                score,
                document.page_content,
            )
        )
    return "\n\n".join(sections)
