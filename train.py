# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# Train Gan models using Triticale Images
# train.py data_dir model_name

import os.path
import shutil
import sys
import model
from keras.models import save_model
from keras.callbacks import TensorBoard
from tensorflow import keras


def main(data_dir, model_name, params='param.txt'):
    #
    # Check and Make Directories
    #
    types = ['alive', 'dead']
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    os.mkdir(model_name)
    os.mkdir(os.path.join(model_name, 'callbacks'))
    os.mkdir(os.path.join(model_name, 'alive'))
    os.mkdir(os.path.join(model_name, 'callbacks', 'alive'))
    os.mkdir(os.path.join(model_name, 'dead'))
    os.mkdir(os.path.join(model_name, 'callbacks', 'dead'))

    #
    # Read Hyperparams
    #
    para = []
    if not os.path.exists(params):
        params_f = open(params, 'w')
        params_f.write("EPOCH=500\n")
        params_f.write("LEARNING_RATE=0.00001\n")
        params_f.write("BATCH=32\n")
        params_f.write("LATENT=128\n")
        params_f.close()

    for i, line in enumerate(open(params, 'r')):
        if not i == 1:
            para.append(int(line.strip().split('=')[1]))
        else:
            para.append(float(line.strip().split('=')[1]))

    EPOCH, LEARNING_RATE, BATCH, LATENT = para

    #
    # For Each Data Types (Alive and Dead)
    #

    for t in types:
        #
        # Load Dataset
        #
        dataset = keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_dir, t), label_mode=None, batch_size=BATCH, image_size=(192, 256)
        )
        dataset = dataset.map(lambda x: x/255.0)

        #
        # Build and Fit Model
        #
        gan, gan_callback = model.model_gan(train_type=t,
                                            latent=LATENT,
                                            lr=LEARNING_RATE,
                                            model_name=model_name)
        CB = TensorBoard(log_dir=os.path.join(model_name, t, 'logs'))
        ES = keras.callbacks.EarlyStopping(monitor='g_loss', patience=10)

        gan.fit(
            dataset, epochs=EPOCH, callbacks=[gan_callback, ES, CB]
        )

        #
        # Save Model
        #

        os.mkdir(os.path.join(model_name, 'model'))
        os.mkdir(os.path.join(model_name, 'model', t))
        save_model(
            gan.generator,
            os.path.join('model', t),
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )


if __name__ == '__main__':
    main(data_dir=sys.argv[1],
        model_name=sys.argv[2])
