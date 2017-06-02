import os
from PIL import Image
from resizeimage import resizeimage

PATH = 'train'
SAVE = 'thumb'

def resize(filename):
    with open(filename, 'r+b') as f:
        with Image.open(f) as image:
            thumb = resizeimage.resize_thumbnail(image, [320, 320])
            return thumb

for typename in os.listdir(PATH):
    if not typename.startswith('.'):
        if not os.path.exists(os.path.join(SAVE, typename)):
            os.makedirs(os.path.join(SAVE, typename))
        for filename in os.listdir(os.path.join(PATH, typename)):
            if not filename.startswith('.'):
                thumb = resize(os.path.join(PATH, typename, filename))
                thumb.save(os.path.join(SAVE, typename, filename))
