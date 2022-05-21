# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# Generate Triticale Images Using Gan
# generate.py model_name img_num

import os.path
import sys

import tensorflow as tf
import keras
import tqdm
from keras.models import load_model


def main(model_name, img_num, params='params.txt'):
    types = ['alive', 'dead']
    #
    # Read Hyperparams
    #
    para = []
    if not os.path.exists(params):
        params_f = open(params, 'w')
        params_f.write("EPOCH=1\n")
        params_f.write("LEARNING_RATE=0.00001\n")
        params_f.write("BATCH=32\n")
        params_f.write("LATENT=128\n")
        params_f.close()

    for i, line in enumerate(open(params, 'r')):
        if not i == 1:
            para.append(int(line.strip().split('=')[1]))
        else:
            para.append(float(line.strip().split('=')[1]))

    LATENT = para[3]

    #
    # For Each Types (Alive, Dead)
    #

    for t in types:
        Model = load_model(os.path.join(model_name, 'model', t))

        random_latent_vectors = tf.random.normal(shape=(img_num, LATENT))
        generated_images = Model(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        pbar = tqdm.trange(range(img_num))
        for i in pbar:
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(os.path.join(model_name, "generated", t, "generated_img_final_{0}.png".format(i)))
            pbar.set_description("Generating {0} Triticale Image..".format(t))

    print('===== Generation Finished! =====')


if __name__ == '__main__':
    main(
        model_name=sys.argv[1],
        img_num=int(sys.argv[2])
    )
