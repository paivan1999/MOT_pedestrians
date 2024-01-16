from io import BytesIO
import win32clipboard
from PIL import Image

def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

def copy(path):
    image = Image.open(path)

    output = BytesIO()
    image.convert("RGB").save(output, "DIB")
    data = output.getvalue()
    output.close()

    send_to_clipboard(win32clipboard.CF_DIB, data)


def get_new_file_name(path)->str:
    n = 0
    with open(path, "r") as f:
        n = int(f.read())
    with open(path, "w") as f:
        f.write(str(n + 1))
    return f"{n}.jpg"
