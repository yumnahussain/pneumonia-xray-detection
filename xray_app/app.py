"""Gradio app: upload a chest X-ray, get the fine-tuned model's reading, ask follow-ups.

Run:  python app.py           (add SHARE=1 for a public link, e.g. from Colab)
Test the UI without a GPU:  MOCK_MODEL=1 python app.py
"""

from __future__ import annotations

import os
import traceback

import gradio as gr

import safety
from model import load_model

MODEL = None  # loaded lazily so the page can start even while weights download


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


# --------------------------------------------------------------------------- handlers


def analyze(image, acknowledged, _history):
    """Check the upload, run the classifier, and start a fresh conversation."""
    empty_state = {"summary": None}
    if not acknowledged:
        gr.Warning("Please confirm you understand this is an educational tool first.")
        return [], empty_state, gr.update()
    if image is None:
        gr.Warning("Upload a chest X-ray first.")
        return [], empty_state, gr.update()

    model = get_model()

    # Image gate, part 1: cheap color check. X-rays are grayscale.
    if safety.color_saturation(image) > safety.MAX_SATURATION:
        return [{"role": "assistant", "content": safety.NOT_XRAY_MESSAGE}], empty_state, gr.update(value=None)

    # Image gate, part 2: ask the base model whether it's a chest X-ray.
    try:
        p_xray = model.is_chest_xray(image)
    except Exception:
        traceback.print_exc()
        raise gr.Error("The model failed while checking the image. See the server log.")
    if p_xray < safety.XRAY_PROB_THRESHOLD:
        return [{"role": "assistant", "content": safety.NOT_XRAY_MESSAGE}], empty_state, gr.update(value=None)

    try:
        label, confidence, _raw = model.classify(image)
    except Exception:
        traceback.print_exc()
        raise gr.Error("The model failed while reading the image. See the server log.")

    message = safety.format_prediction(label, confidence, model.config.adapter_revision)
    state = {"summary": safety.classifier_summary(label, confidence)}
    return [{"role": "assistant", "content": message}], state, gr.update()


def ask(question, history, image, state):
    """Screen the question, ask the model, guard the answer."""
    question = (question or "").strip()
    if not question:
        return history, ""
    history = list(history or [])
    history.append({"role": "user", "content": question})

    screening = safety.screen_message(question)

    # Emergencies and self-harm get a fixed response; the model is never asked.
    if screening.block_model:
        history.append({"role": "assistant", "content": screening.prefix})
        return history, ""

    if image is None or not state or not state.get("summary"):
        history.append({
            "role": "assistant",
            "content": "Please upload a chest X-ray and press **Analyze X-ray** before asking questions.",
        })
        return history, ""

    system_prompt = safety.SYSTEM_PROMPT + "\n\n" + state["summary"]
    if screening.extra_instruction:
        system_prompt += "\n\nFor this question: " + screening.extra_instruction

    # Earlier turns for the model: skip the long result card and fixed safety messages,
    # and strip warnings we added so the model sees only real conversation.
    model_history = []
    for turn in history[:-1]:
        content = turn["content"] if isinstance(turn["content"], str) else ""
        if turn["role"] == "assistant":
            if content.startswith(("**Model result", "**This sounds like", "I'm really sorry", "This doesn't look")):
                continue
            content = content.split("\n\n---\n*Safety note")[0]
            if content.startswith("> ⚠️"):
                content = content.split("\n\n", 1)[-1]
        model_history.append({"role": turn["role"], "content": content})
    # Gemma needs strictly alternating user/assistant turns starting with user.
    model_history = _alternate(model_history)

    try:
        answer = get_model().chat(image, system_prompt, model_history, question)
    except Exception:
        traceback.print_exc()
        raise gr.Error("The model failed while answering. See the server log.")

    guarded = safety.guard_output(answer, screening.intent)
    reply = guarded.text
    if screening.prefix:
        reply = screening.prefix + "\n\n" + reply
    history.append({"role": "assistant", "content": reply})
    return history, ""


def _alternate(turns):
    """Drop turns so roles alternate user/assistant and the list starts with user."""
    out = []
    for t in turns:
        if not out and t["role"] != "user":
            continue
        if out and out[-1]["role"] == t["role"]:
            out[-1] = t
        else:
            out.append(t)
    if out and out[-1]["role"] == "user":
        out.pop()
    return out


def clear_all():
    return None, [], {"summary": None}, ""


# --------------------------------------------------------------------------- layout

CSS = """
.disclaimer {border-left: 4px solid #d97706; padding: 10px 14px; border-radius: 6px;
             background: rgba(217,119,6,0.08);}
footer {display: none !important;}
"""

EXAMPLE_QUESTIONS = [
    "What features would support pneumonia on this image?",
    "What else could look like this? Give me a short differential.",
    "Walk me through a systematic approach to reading this X-ray.",
    "Why might the model be wrong on this image?",
]

with gr.Blocks(
    title="Chest X-Ray Study Tool",
    css=CSS,
    theme=gr.themes.Soft(),
    delete_cache=(1800, 3600),  # remove uploaded files from Gradio's temp cache
) as demo:
    gr.Markdown("# Chest X-Ray Study Tool\nFine-tuned MedGemma 4B · pneumonia vs. normal")
    gr.Markdown(safety.DISCLAIMER, elem_classes="disclaimer")

    state = gr.State({"summary": None})

    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(
                type="pil",
                sources=["upload"],
                image_mode="RGB",
                label="Chest X-ray (frontal view)",
                height=380,
            )
            acknowledged = gr.Checkbox(
                label="I understand this is an educational tool and cannot diagnose anyone.",
                value=False,
            )
            analyze_btn = gr.Button("Analyze X-ray", variant="primary")
            gr.Markdown(f"<sub>{safety.PRIVACY_NOTE}</sub>")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                height=520,
                label="Results and follow-up questions",
                show_copy_button=True,
            )
            with gr.Row():
                question = gr.Textbox(
                    placeholder="Ask a follow-up question about this image…",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send", scale=1)
            gr.Examples(EXAMPLE_QUESTIONS, inputs=question, label="Example questions")
            clear_btn = gr.Button("Start over", variant="secondary")

    analyze_btn.click(analyze, [image, acknowledged, chatbot], [chatbot, state, image])
    send_btn.click(ask, [question, chatbot, image, state], [chatbot, question])
    question.submit(ask, [question, chatbot, image, state], [chatbot, question])
    clear_btn.click(clear_all, None, [image, chatbot, state, question])
    # A new upload invalidates the previous reading.
    image.upload(lambda: ([], {"summary": None}), None, [chatbot, state])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)  # one GPU job at a time
    demo.launch(share=os.environ.get("SHARE", "0") == "1")
