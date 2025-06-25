import os
import gradio as gr
from huggingface_hub import InferenceClient
from pipeline import get_retrieval_answer

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(message, history, system_message, max_tokens, temperature, top_p):
    # First try PDF-based answer
    retrieval = get_retrieval_answer(message)
    if retrieval:
        return retrieval

    # Otherwise fallback to Zephyr
    messages = [{"role": "system", "content": system_message}]
    for user, bot in history:
        if user:
            messages.append({"role": "user", "content": user})
        if bot:
            messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": message})

    response = ""
    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = chunk.choices[0].delta.content or ""
        response += token
        yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful AI assistant for analyzing PDF filings.", label="System message"),
        gr.Slider(1, 2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(0.1, 4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p"),
    ],
)

if __name__ == "__main__":
    demo.launch()