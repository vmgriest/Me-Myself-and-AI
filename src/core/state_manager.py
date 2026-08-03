# Manages thread state and memory
#
# Thin wrapper only — LangGraph's checkpointer already persists full state
# (including "messages", via the add_messages reducer) keyed by thread_id.
# This module just owns the single checkpointer instance for the app's
# lifetime and generates/tracks a thread_id per chat session; it does not
# store messages itself.
import uuid  # to generate unique thread ids

from langgraph.checkpoint.memory import MemorySaver  # in-memory checkpointer — resets on restart, fine for local dev

_checkpointer = MemorySaver()  # one shared instance for the whole process


def get_checkpointer() -> MemorySaver:
    return _checkpointer  # handed to graph.compile(checkpointer=...) in graph.py


def new_thread_id() -> str:
    return str(uuid.uuid4())  # a fresh id for a new chat session


def get_thread_config(thread_id: str) -> dict:
    # the shape LangGraph expects for its "config" argument on invoke()/stream()
    return {"configurable": {"thread_id": thread_id}}
