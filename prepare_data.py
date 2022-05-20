# Classify raw images by label
import shutil

import pandas as pd
import os
from PIL import Image


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

    if os.path.exists('raw_data/data'):
        shutil.rmtree('raw_data/data')
    os.mkdir('raw_data/data')
    os.mkdir('raw_data/data/alive')
    os.mkdir('raw_data/data/dead')

    for sample in df.values:
        robust = sample[1]
        file = sample[2]
        if robust:
            img = Image.open(file)
            img = img.resize((256, 192), Image.ANTIALIAS)  # 4:3
            rgb_img = img.convert('RGB')
            rgb_img.save(os.path.join('raw_data/data/alive', sample[0]+'.png'))

        else:
            img = Image.open(file)
            img = img.resize((256, 192), Image.ANTIALIAS)
            rgb_img = img.convert('RGB')
            rgb_img.save(os.path.join('raw_data/data/dead', sample[0] + '.png'))


if __name__ == '__main__':
    main(label='raw_data/label/triticale.xlsx',
         raw_dir='raw_data/raw_img')
