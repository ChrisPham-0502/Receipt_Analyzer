import gradio as gr
import pandas as pd
from models import Reciept_Analyzer
from utils import find_product
import os
model = Reciept_Analyzer()

sample_images = []
for img_file in os.listdir("samples/"):
    sample_images.append(os.path.join("samples", img_file))

def predict(image):
    results = model.forward(image)
    return results

def handle_input(image, item_name):
    df = predict(image)  # Phân tích hóa đơn và trả về dataframe
    print(df)
    result = find_product(item_name, df)
    return result

# Thiết kế giao diện với Gradio
def create_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Ứng dụng phân tích hóa đơn siêu thị")

        with gr.Row():
            # Cột bên trái
            with gr.Column():
                gr.Markdown("### Tải lên hóa đơn hoặc chọn ảnh mẫu")
                image_input = gr.Image(label="Ảnh hóa đơn", type="filepath")

                gr.Markdown("### Ảnh mẫu")
                example = gr.Examples(
                    inputs=image_input,
                    examples=sample_images
                )

            # Cột bên phải
            with gr.Column():
                gr.Markdown("### Tìm kiếm thông tin item")
                item_input = gr.Textbox(label="Tên item cần tìm")
                output = gr.Textbox(label="Kết quả")

                search_button = gr.Button("Tìm kiếm")
                search_button.click(fn=handle_input, inputs=[image_input, item_input], outputs=output)

    return app


# Chạy ứng dụng
app = create_interface()
app.launch()