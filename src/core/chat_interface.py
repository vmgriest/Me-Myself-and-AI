# Handles Gradio chat streaming & message display
#
# Phase 1 doesn't actually stream tokens (respond() returns the full string
# once the LLM call finishes) — it just renders the answer plus a
# collapsible list of the source chunks it was grounded in.
import gradio as gr  # the chat UI framework

from src.core.rag_system import RAGSystem  # the backend that actually answers questions


def _format_response(result: dict) -> str:
    """Appends a <details> block listing each retrieved source (file,
    similarity score, short excerpt) below the answer, so grounding is
    visible in the chat UI."""
    answer = result["answer"]  # the LLM's text reply
    sources = result["sources"]  # the chunks that were retrieved for this query
    if not sources:  # nothing retrieved (shouldn't normally happen) — just show the answer
        return answer

    lines = [answer, "", "<details><summary>Sources</summary>", ""]  # start building a markdown+HTML block
    for s in sources:  # one bullet per retrieved chunk
        excerpt = s["text"][:200].replace("\n", " ")  # first 200 chars, flattened to one line
        lines.append(f"- **{s['source']}** (score {s['score']:.2f}): {excerpt}...")  # e.g. "- **bio.md** (score 0.58): ..."
    lines.append("</details>")  # close the collapsible block
    return "\n".join(lines)  # join everything into one markdown string


def build_interface(rag_system: RAGSystem) -> gr.Blocks:
    # gr.ChatInterface handles all the chat history/UI plumbing — respond()
    # just needs to take the latest message and return a reply.
    def respond(message: str, history: list[dict]) -> str:
        result = rag_system.answer(message)  # run retrieval + generation for this message
        return _format_response(result)  # format it (answer + sources) for display

    return gr.ChatInterface(
        fn=respond,  # the callback Gradio calls on every user message
        title="About Me",  # page title shown in the browser tab / header
        description="Ask me anything about Vincent.",  # subtitle shown under the title
    )
