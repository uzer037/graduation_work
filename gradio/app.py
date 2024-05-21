import os
import pathlib

from transformers import pipeline
import gradio as gr
import torch
from uuid import uuid4
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = pipeline("image-classification", model="test_trainer", device=device)


def check_image(img: Image) -> float:
    resp = model(img)
    return resp[0]['score']


def format_score(score: float) -> str:
    if score >= 0.5:
        return f"{round(score * 100, 2)}% Good"
    else:
        return f"{round((1 - score) * 100, 2)}% Bad"


def check_gallery(gallery: list[Image]) -> (gr.Gallery, gr.File):
    session_id = uuid4()
    scores = [check_image(img[0]) for img in gallery]
    res = [(gallery[i][0], format_score(scores[i])) for i in range(len(gallery))]

    if not pathlib.Path("./checks").is_dir():
        os.mkdir("./checks")
    path = "./checks/" + str(session_id)
    threshold = 0.5
    fnames = [img[0].filename.replace("\\", "/").split("/")[-1] for img in gallery]
    reals = [fnames[i] for i in range(len(scores)) if scores[i] >= threshold]
    fakes = [fnames[i] for i in range(len(scores)) if scores[i] < threshold]

    with open(path + "_good.txt", "w") as f:
        for fname in reals:
            f.write(fname + "\n")
    with open(path + "_bad.txt", "w") as f:
        for fname in fakes:
            f.write(fname + "\n")

    download_results = gr.File(label="Скачать списки изображений с артефактами (good) / без артефактов (bad)",
                               value=[path + "_good.txt", path + "_bad.txt"],
                               interactive=False, file_count="multiple", visible=True)
    clear_btn = gr.ClearButton(value="Очистить", components=[inp, download_results_btn], visible=True)
    return res, download_results, clear_btn


def on_clear():
    return (gr.File(label="Скачать список реальных фото", interactive=False, visible=False),
            gr.ClearButton(value="Очистить", components=[inp,download_results_btn], visible=False))

custom_css = """
#toggle_dark {
background-color: black;
}
"""

with gr.Blocks(theme=gr.themes.Base()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=50):
                gr.Markdown("""
                # Фильтр сгенерированных изображений\n\n
                Фильтр позволяет отсеять сгенерированные изображения низкого качества.<br>
                Загрузите изображения, чтобы начать проверку. Результаты проверки доступны в течении часа.
                """)
            with gr.Column(scale=0):
                toggle_dark = gr.Button(value="Темная/Светлая тема", icon="img/dark-mode.svg", elem_id="toggle_dark")

        inp = gr.Gallery(show_download_button=False, show_share_button=False, show_label=False, type="pil", columns=3)
        download_results_btn = gr.File(label="Скачать список реальных фото (доступно в течении часа)",
                                       interactive=False, file_count="single", visible=False)
        clear_btn = gr.ClearButton(value="Очистить", components=[inp, download_results_btn], visible=False)


    # Logic
    inp.upload(fn=check_gallery, inputs=inp, outputs=[inp, download_results_btn, clear_btn])
    clear_btn.click(on_clear,[],[download_results_btn, clear_btn])

    toggle_dark.click(None, js="""
        () => {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
            } else {
                url.searchParams.set('__theme', 'light');
            }
            window.location.href = url.href;
        }
        """)

demo.launch(allowed_paths=["./checks/*"], share=True)
