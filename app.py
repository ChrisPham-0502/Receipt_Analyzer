import gradio as gr
import pandas as pd
from models import Reciept_Analyzer
from utils import find_product, get_info
import os

model = Reciept_Analyzer()

sample_images = []
for img_file in os.listdir("samples/"):
    sample_images.append(os.path.join("samples", img_file))

def predict(image):
    results = model.forward(image)
    return results

def create_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Receipt Analyzer")

        with gr.Row():
            # Cột bên trái
            with gr.Column():
                gr.Markdown("### Upload your invoice or example image")
                image_input = gr.Image(label="Invoice", type="filepath")

                

                res = None
                def on_image_selected(image_path):
                    global res
                    res = predict(image_path)
                    final = get_info(res)
                    print(res)
                    return final

                def handle_input(item_name):
                    global res
                    result = find_product(item_name, res)
                    return result
                

                gr.Markdown("### Examples")
                example = gr.Examples(
                    inputs=image_input,
                    examples=sample_images
                )

            # Cột bên phải
            with gr.Column():
                result_output = gr.Textbox(label="Analysis result")
                image_input.change(fn=on_image_selected, inputs=image_input, outputs=result_output)
                gr.Markdown("### Search item information")
                item_input = gr.Textbox(label="Item name")
                output = gr.Textbox(label="Results")

                search_button = gr.Button("Submit")
                search_button.click(fn=handle_input, inputs=item_input, outputs=output)

    return app


# Chạy ứng dụng
app = create_interface()
app.launch()
