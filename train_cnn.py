# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# Train CNN Model
# train.py model_name data_dir

import os
import sys
import shutil
from tensorflow import keras
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import save_model

import model


def main(model_name, data_dir, params='param.txt'):
    #
    # Check and Make Directories
    #
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    os.mkdir(model_name)
    os.mkdir(os.path.join(model_name, 'model'))
    os.mkdir(os.path.join(model_name, 'logs'))


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
        params_f.write("EPOCH_CNN=500\n")
        params_f.write("LEARNING_RATE_CNN=0.0001\n")
        params_f.write("BATCH_CNN=4\n")
        params_f.close()

    for i, line in enumerate(open(params, 'r')):
        if not (i == 1 or i == 5):
            para.append(int(line.strip().split('=')[1]))
        else:
            para.append(float(line.strip().split('=')[1]))

    EPOCH, LEARNING_RATE, BATCH = para[-4:-1]

    #
    # Load Dataset
    #
    dataset = keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir), batch_size=BATCH, image_size=(1024, 1024)
    )

    #
    # Build and Fit Model
    #
    CB = TensorBoard(log_dir=os.path.join(model_name, 'logs'))
    ES = EarlyStopping(monitor='accuracy', patience=10)

    Model = model.model_cnn(model_name, lr=LEARNING_RATE)

    _ = Model.fit(dataset, epochs=EPOCH, callbacks=[ES, CB])

    #
    # Save Model
    #
    save_model(
        Model,
        os.path.join(model_name, 'model'),
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


if __name__ == '__main__':
    main(model_name=sys.argv[1],
         data_dir=sys.argv[2])
