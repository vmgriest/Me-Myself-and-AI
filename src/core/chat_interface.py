# Handles Gradio chat streaming & message display
#
# Phase 3: each browser session gets its own thread_id (via gr.State,
# Gradio's per-session state mechanism), so LangGraph's checkpointer keeps
# separate conversation histories per user rather than one shared thread.
# Also lets the user pick which LLM answers, via a model dropdown.
import gradio as gr  # the chat UI framework

from src import config  # for list_available_models()/default_model_choice()
from src.core.rag_system import RAGSystem  # the backend that actually answers questions
from src.core.state_manager import new_thread_id  # generates a fresh thread id for a new session


def _format_response(result: dict) -> str:
    """Appends a <details> block listing each retrieved source (file,
    similarity score, short excerpt) below the answer, so grounding is
    visible in the chat UI."""
    answer = result["answer"]  # the LLM's text reply
    sources = result["sources"]  # the chunks (and/or live web results) used for this query
    if not sources:  # nothing retrieved (shouldn't normally happen) — just show the answer
        return answer

    lines = [answer, "", "<details><summary>Sources</summary>", ""]  # start building a markdown+HTML block
    for s in sources:  # one bullet per retrieved chunk / web result
        excerpt = s["text"][:200].replace("\n", " ")  # first 200 chars, flattened to one line
        lines.append(f"- **{s['source']}** (score {s['score']:.2f}): {excerpt}...")  # e.g. "- **bio.md** (score 0.58): ..."
    lines.append("</details>")  # close the collapsible block
    return "\n".join(lines)  # join everything into one markdown string


def build_interface(rag_system: RAGSystem) -> gr.Blocks:
    model_choices = config.list_available_models()  # every model actually usable right now (Ollama pulled + cloud keys set)
    default_choice = config.default_model_choice()
    if default_choice not in model_choices and model_choices:  # .env's default isn't actually available — fall back
        default_choice = model_choices[0]

    # gr.ChatInterface handles all the chat history/UI plumbing — respond()
    # just needs to take the latest message (plus our extra model/thread_id
    # state) and return a reply (plus the possibly-just-created thread_id).
    def respond(message: str, history: list[dict], model_choice: str, thread_id: str | None):
        if thread_id is None:  # first message of a new browser session
            thread_id = new_thread_id()  # mint a fresh id so this session gets its own LangGraph thread
        result = rag_system.answer(message, thread_id, model_choice)  # run one turn of the agent for this thread
        return _format_response(result), thread_id  # reply text, plus the (possibly new) thread_id to persist

    # Wrapped in an explicit Blocks so model_dropdown and thread_state can be pre-rendered (see below) —
    # ChatInterface itself is returned by build_interface() and .launch()'d directly, so this Blocks *is*
    # the whole app.
    with gr.Blocks(title="About Me") as demo:
        gr.Markdown("# About Me\nAsk me anything about Vincent.")
        model_dropdown = gr.Dropdown(
            choices=model_choices, value=default_choice, label="Model", interactive=bool(model_choices)
        )
        thread_state = gr.State(None)  # ONE shared component instance — must be reused as both input and output
        # below, or Gradio treats them as two unrelated components and the thread_id never round-trips back in.
        # Rendering both of these here, before ChatInterface, means they're already `is_rendered` by the time
        # ChatInterface sees them as additional_inputs — which is what stops ChatInterface from wrapping them
        # in its own separate (and, for thread_state, pointlessly empty) "Additional Inputs" accordion. The
        # dropdown instead shows up exactly where it's placed above: always visible, not tucked in a collapsed
        # accordion the user would have to know to open.

        gr.ChatInterface(
            fn=respond,  # the callback Gradio calls on every user message
            additional_inputs=[model_dropdown, thread_state],  # model choice + per-session thread_id
            additional_outputs=[thread_state],  # writes the (possibly new) thread_id back into that same session state
        )

    return demo
