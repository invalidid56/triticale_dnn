# Classify raw images by label
import random
import shutil

import pandas as pd
import os
from PIL import Image, ImageFilter
from random import randint


def blur(img: Image):
    return img.filter(ImageFilter.BLUR)


def rb_shift(img):  # Green is and Crucial Factor
    width, height = img.size
    newimg = Image.new("RGB", (width, height), (255, 255, 255))
    newpxl = newimg.load()
    pxl = img.load()

    for h in range(height):
        for w in range(width):
            r, g, b = pxl[w, h]
            newpxl[w, h] = (255-r, g, 255-b)
    return newimg


def rotation(img):
    return img.rotate(randint(0, 360))


def reverse(img):
    return img.rotate(180)


def main(raw_dir, label):
    df = pd.read_excel(label)

    temp = []
    for file in df.Sorted:
        if os.path.exists(os.path.join(raw_dir, file+'.tif')):
            temp.append(os.path.join(raw_dir, file+'.tif'))
        else:
            temp.append(pd.NA)
    temp = pd.Series(temp, name='File')
    df = pd.concat([df, temp], axis=1)

    def dead_or_alive(robust):
        return robust >= 60

    df.robust = df.robust.map(dead_or_alive)
    df = df.dropna()

    '''
    if os.path.exists('raw_data/data'):
        shutil.rmtree('raw_data/data')
    os.mkdir('raw_data/data')
    os.mkdir('raw_data/data/alive')
    os.mkdir('raw_data/data/dead')
    '''

    if os.path.exists('raw_data/hq_data'):
        shutil.rmtree('raw_data/hq_data')
    os.mkdir('raw_data/hq_data')
    os.mkdir('raw_data/hq_data/alive')
    os.mkdir('raw_data/hq_data/dead')

    tasks = [blur, rotation, rotation, reverse]

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    for sample in df.values:
        robust = sample[1]
        file = sample[2]

        img_type = 'alive' if robust else 'dead'
        img = Image.open(file)
        img = expand2square(img, (0, 0, 0))
        img = img.resize((1024, 1024), Image.ANTIALIAS)  # 4:3
        rgb_img = img.convert('RGB')
        imgs = [task(rgb_img) for task in tasks] + [rgb_img]    # Data Augmentation

        for k, img in enumerate(imgs):
            img.save(os.path.join('raw_data/hq_data', img_type, sample[0] + str(k) + '.png'))


if __name__ == '__main__':
    main(label='raw_data/label/triticale.xlsx',
         raw_dir='raw_data/raw_img')
