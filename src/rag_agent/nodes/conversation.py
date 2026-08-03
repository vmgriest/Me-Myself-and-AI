# Summarize history, rewrite queries
#
# Both nodes are built as factories (make_x_node(llm)) so graph.py can hand
# them the single shared LLM instance via closure, rather than each node
# re-creating its own client.
from langchain_core.messages import RemoveMessage  # marker that tells add_messages to drop a message by id

from src.rag_agent.prompts import REWRITE_QUERY_PROMPT, SUMMARIZE_PROMPT
from src.rag_agent.state import AgentState

SUMMARIZE_AFTER_N_MESSAGES = 8  # once history exceeds this, fold older turns into "summary"
KEEP_LAST_N_MESSAGES = 2  # how many recent messages stay verbatim after summarizing


def make_rewrite_query_node(llm):
    def rewrite_query(state: AgentState) -> dict:
        messages = state["messages"]  # full history for this thread (persisted by the checkpointer)
        latest_message = messages[-1].content  # the newest human message, the one we're resolving

        if len(messages) == 1:  # first turn in the conversation — nothing to resolve against
            return {"rewritten_query": latest_message}  # use it as-is, skip the extra LLM call

        recent = messages[:-1][-6:]  # a handful of prior turns for context (not the whole history)
        recent_text = "\n".join(f"{m.type}: {m.content}" for m in recent)  # flatten to plain text for the prompt

        prompt = REWRITE_QUERY_PROMPT.format(
            summary=state.get("summary") or "(none yet)",  # fall back to a placeholder if no summary exists yet
            recent_messages=recent_text,
            latest_message=latest_message,
        )
        response = llm.invoke(prompt)  # ask the LLM to produce a standalone version of the question
        return {"rewritten_query": response.content.strip()}  # store it for the retrieve node to use

    return rewrite_query


def make_summarize_node(llm):
    def summarize_history(state: AgentState) -> dict:
        messages = state["messages"]  # full history for this thread
        if len(messages) <= SUMMARIZE_AFTER_N_MESSAGES:  # history still short — nothing to do
            return {}  # empty dict = no state changes

        to_summarize = messages[:-KEEP_LAST_N_MESSAGES]  # older messages being folded into the summary
        to_keep = messages[-KEEP_LAST_N_MESSAGES:]  # most recent messages, kept verbatim
        new_messages_text = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)  # flatten for the prompt

        prompt = SUMMARIZE_PROMPT.format(
            existing_summary=state.get("summary") or "(none yet)",  # extend the prior summary, don't restart it
            new_messages=new_messages_text,
        )
        response = llm.invoke(prompt)  # ask the LLM to produce the updated summary

        # RemoveMessage tells the add_messages reducer to drop these from history now that they're summarized
        removals = [RemoveMessage(id=m.id) for m in to_summarize]
        return {"summary": response.content.strip(), "messages": removals}

    return summarize_history
