from generate import generate_text
import gradio as gr

def gradio_generate(prompt):
    return generate_text(prompt, max_length=100)

iface = gr.Interface(fn=gradio_generate, inputs="text", outputs="text", title="GPT-2 Text Generator")
iface.launch()
