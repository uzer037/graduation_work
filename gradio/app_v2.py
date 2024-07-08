from transformers import pipeline
import gradio as gr
import torch
from uuid import uuid4
from pathlib import Path
import shutil
from inmemzip import get_zip_buffer, format_fname

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = pipeline("image-classification", model="test_trainer", device=device)

default_threshold = 0.75
description = """
# Фильтр сгенерированных изображений\n\n
Фильтр позволяет отсеять сгенерированные изображения низкого качества.<br>
Загрузите изображения, чтобы начать проверку. Результаты проверки доступны в течении часа.
"""


def format_score(score: float, thr: float) -> str:
    return f"{'✅' if score >= thr else '❌'} Качество: {round(score * 100, 2)} ({'хорошее' if score >= thr else 'плохое'})"


def get_gallery_value(files: list, scores, thr: float) -> list[tuple]:
    return [(files[i][0], format_score(scores[i], thr)) for i in range(len(files))]


def score_files(files: list, threshold: gr.State) -> (gr.Gallery, gr.File, gr.Row):
    if len(files) == 0:
        return

    # getting image from [image, label], passing through model and getting score for good from [good, bad]
    scores = [model(img[0])[0]['score'] for img in files]
    return str(uuid4()), scores, get_gallery_value(files, scores, threshold), gr.update(visible=True)


with gr.Blocks(theme=gr.themes.Base()) as demo:
    # Variables
    check_id_state = gr.State("")
    scores_state = gr.State([])
    threshold = gr.State(default_threshold)
    # UI
    with gr.Column():
        description_label = gr.Markdown(description)
        toggle_dark = gr.Button(value="Темная/Светлая тема", icon="img/dark-mode.svg", elem_id="toggle_dark")
        gallery = gr.Gallery(columns=3, show_download_button=False, show_share_button=False, show_label=False)
        thr_slider = gr.Slider(value=round(default_threshold*100), minimum=50, step=1, label="Порог точности определения")
        with gr.Row(visible=False) as post_upload_block:
            download_btn = gr.DownloadButton(label="Скачать результаты", variant="primary")
            download_block = gr.File(label="Скачать файлы", visible=False)
            clear_btn = gr.ClearButton(gallery, value="Очистить")

    # Logic

    def prepare_download(check_id, files, scores, thr):
        base_dir = f"checks/{check_id}"
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        # File names
        goods = [files[i][0] for i in range(len(files)) if scores[i] >= thr]
        bads = [files[i][0] for i in range(len(files)) if scores[i] < thr]
        with open(f"{base_dir}/good_list.txt", mode="w") as f:
            f.write("\n".join([format_fname(file) for file in goods]))
        with open(f"{base_dir}/bad_list.txt", mode="w") as f:
            f.write("\n".join([format_fname(file) for file in bads]))
        # Files
        with open(f"{base_dir}/good_images.zip", mode="wb") as f:
            f.write(get_zip_buffer(goods).getvalue())
        with open(f"{base_dir}/bad_images.zip", mode="wb") as f:
            f.write(get_zip_buffer(bads).getvalue())

        fnames = ["good_list.txt", "bad_list.txt", "good_images.zip", "bad_images.zip"]
        files = [f"{base_dir}/{fname}" for fname in fnames]
        return gr.update(visible=False), gr.update(visible=True, value=files)

    def on_thr_update(value, files, scores):
        thr = 0.01 * value
        if files is None or scores is None or len(files) == 0 or len(scores) == 0:
            return thr, gr.update(), gr.update(visible=True), gr.update(visible=False)
        else:
            return thr, gr.Gallery(get_gallery_value(files, scores, thr)), gr.update(visible=True), gr.update(visible=False)

    def on_clear():
        path = Path(f"checks/{check_id_state.value}")
        if path.exists():
            shutil.rmtree(path)
        return gr.update(visible=False)

    gallery.upload(fn=score_files, inputs=[gallery, threshold],
                   outputs=[check_id_state, scores_state, gallery, post_upload_block])
    thr_slider.change(on_thr_update, inputs=[thr_slider, gallery, scores_state],
                      outputs=[threshold, gallery, download_btn, download_block])
    clear_btn.click(on_clear, outputs=[post_upload_block])
    download_btn.click(prepare_download,
                       inputs=[check_id_state, gallery, scores_state, threshold],
                       outputs=[download_btn, download_block])

    toggle_dark.click(None, js="""
        () => {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'light') {
                url.searchParams.set('__theme', 'light');
            } else {
                url.searchParams.set('__theme', 'dark');
            }
            window.location.href = url.href;
        }
        """)

demo.launch(allowed_paths=["./checks/*"], share=True)
