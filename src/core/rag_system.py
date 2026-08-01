# Orchestrates retrieval + generation.
#
# Phase 1: a plain function chain (embed -> search -> prompt -> LLM), no
# LangGraph yet. Phase 3 will swap the body of answer() to graph.invoke(...)
# once the LangGraph agent exists — the public answer() signature below is
# designed to stay stable across that change so chat_interface.py never
# needs to change.
from src import config  # for get_llm() and RETRIEVAL_TOP_K
from src.ingestion import embeddings, vector_store  # to embed the query and search Qdrant

# Instructs the LLM to only use the retrieved context and not hallucinate; {context} is filled in per-query.
SYSTEM_PROMPT = (
    "You are a personal assistant answering questions about Vincent based only on "
    "the context provided below. If the answer isn't in the context, say you don't "
    "have that information — do not make anything up.\n\n"
    "Context:\n{context}"
)


class RAGSystem:
    def __init__(self):
        # One Qdrant client and one LLM instance for the app's lifetime.
        self.client = vector_store.get_qdrant_client()  # the shared local Qdrant connection
        self.llm = config.get_llm()  # the chat model, per LLM_PROVIDER in .env

    def answer(self, query: str) -> dict:
        """Single-turn Q&A: embed the query, retrieve the top-k chunks,
        stuff them into the system prompt, and ask the LLM. Returns both
        the answer and the source chunks used, so the UI can show what
        the answer was grounded in."""
        query_vector = embeddings.embed_query(query)  # turn the user's question into a vector
        hits = vector_store.search(self.client, query_vector, k=config.RETRIEVAL_TOP_K)  # find the closest chunks

        context = "\n\n---\n\n".join(hit.payload["text"] for hit in hits)  # join retrieved chunk texts for the prompt
        sources = [  # a parallel list of source metadata, for display in the UI
            {"source": hit.payload.get("source"), "text": hit.payload.get("text"), "score": hit.score}
            for hit in hits
        ]

        messages = [  # LangChain's (role, content) tuple format for chat models
            ("system", SYSTEM_PROMPT.format(context=context)),  # instructions + retrieved context
            ("human", query),  # the user's actual question
        ]
        response = self.llm.invoke(messages)  # call the LLM and get its reply

        return {"answer": response.content, "sources": sources}  # the text answer plus what it was grounded in
