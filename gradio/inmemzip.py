import io
import zipfile
from PIL import Image


def format_fname(fname: str) -> str:
    return fname.replace("\\", "/").split("/")[-1]

def get_image_buffer(image: Image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, 'PNG')
    img_buffer.seek(0)
    return img_buffer


def get_zip_buffer(list_of_files):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for path in list_of_files:
            img = Image.open(path)
            zip_file.writestr(format_fname(path), get_image_buffer(img).read())
    zip_buffer.seek(0)
    return zip_buffer
