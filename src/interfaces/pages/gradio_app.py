from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient
from src.application.search_cloud import search_movies_cloud as sear
from src.application.database_status import (
    get_database_state,
    POPULATE_HINT,
    UNREACHABLE_HINT,
)
from src.schemas.movie import DatabaseStatus

def welcome(name):
    return f"Welcome to Gradio, {name}!"

_css_path = Path(__file__).parent / "static" / "chat.css"
css = _css_path.read_text(encoding="utf-8")

def prompt_template():
    return """You are a helpful assistant that helps users find information about movies from a knowledge base. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."""



def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """
    response = ""

    status, _count = get_database_state()
    if status == DatabaseStatus.EMPTY:
        yield POPULATE_HINT
        return
    if status == DatabaseStatus.UNREACHABLE:
        yield UNREACHABLE_HINT
        return

    results = sear(message, n_results=5)

    if results['documents'] and len(results['documents'][0]) > 0:
        documents = results['documents'][0]
        distances = results['distances'][0]
        for i, (doc, dist) in enumerate(zip(documents, distances), 1):
            response += f"{i}. {doc}\n   Distance: {dist:.4f}\n\n"
    else:
        response = "No results found."

    yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

history = [
    {"role": "assistant", "content": "Hello, I'm a semantic search about movies, you can put some informations about movies, and I will try guess what movie is😉"},
]

chatbot = gr.Chatbot( history)


chat = gr.ChatInterface(
    fn=respond,
    title="Movie Knowledge Chat",
    chatbot=chatbot
)

with gr.Blocks( css=css) as demo:


    gr.HTML("""
<center>
    <img src='https://i.pinimg.com/originals/36/5e/68/365e6851d51814a090210f47911147ce.gif' 
         alt='Chatbot Animation' 
         width='300'
         style='border-radius: 20px;'>
</center>
""")
    gr.HTML("</center>")
    chat.render()


if __name__ == "__main__":
    demo.launch()
