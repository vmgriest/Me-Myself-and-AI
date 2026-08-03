# State schema for the agent
#
# This TypedDict is the contract every graph node reads from and writes
# partial updates to. LangGraph merges each node's returned dict into the
# overall state using the reducer declared per-field (only "messages" has
# one here — add_messages appends/replaces messages instead of overwriting
# the whole list).
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage  # base type for chat messages LangGraph understands
from langgraph.graph.message import add_messages  # reducer: appends new messages, applies RemoveMessage entries


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full chat history for this thread, persisted by the checkpointer
    rewritten_query: str  # standalone version of the latest question, pronouns/context resolved
    retrieved_chunks: list[dict]  # parent-level context from retrieval.retrieve_parent_context()
    summary: str  # running summary of older turns, once history gets long
    answer: str  # the latest generated answer text, read by rag_system.answer()
    grounded: bool  # whether "generate" considers its own answer supported by the context — see generation.py
    web_results: list[dict]  # live Tavily search results, only populated when the docs didn't have the answer
    used_web_search: bool  # guards against looping: web search is attempted at most once per turn
