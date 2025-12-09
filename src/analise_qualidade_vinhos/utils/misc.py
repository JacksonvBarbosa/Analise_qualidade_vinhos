import os
from PIL import Image, ImageDraw, ImageFont

LOGO_PATH = "logo.png"
HOSPITAL_NAME = "Hospital TechSaúde"
PRIMARY_COLOR = "#ED1455"
ACCENT_COLOR = "#0081A7"


def create_logo(path=LOGO_PATH, hospital=HOSPITAL_NAME):
    if os.path.exists(path):
        return path

    img = Image.new("RGBA", (400, 100), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    draw.rectangle((20, 20, 70, 80), fill=PRIMARY_COLOR)
    draw.rectangle((40, 0, 50, 100), fill=ACCENT_COLOR)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((90, 30), hospital, fill=PRIMARY_COLOR, font=font)

    img.save(path)
    return path
