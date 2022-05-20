# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# Define and Compile DCGAN model for generating triticale image
# Can't Use by Direct Call, use like: model.model_gan('alive', 0.00005, 128, 'test')


import tensorflow as tf
import keras
import os
from keras.models import Sequential, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dropout, Dense, Reshape, BatchNormalization
from keras.metrics import Mean
from keras.callbacks import Callback
from keras.optimizers import adam_v2
from keras.losses import BinaryCrossentropy


def model_gan(train_type, lr, latent, model_name):
    discriminator = Sequential([
        Input(shape=(192, 256, 3)),
        Conv2D(64, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ],
        name="discriminator",
    )
    discriminator.summary()

    latent_dim = latent
    generator = Sequential(
        [
            Input(shape=(latent_dim,)),
            Dense(32 * 24 * 128),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((24, 32, 128)),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),  # 64, 48
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),  # 128, 96
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),  # 256, 192
            BatchNormalization(),
            LeakyReLU(alpha=0.2),

            Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
        ],
        name="generator",
    )
    generator.summary()

    class GAN(keras.Model):
        def __init__(self, discriminator, generator, latent_dim):
            super(GAN, self).__init__()
            self.discriminator = discriminator
            self.generator = generator
            self.latent_dim = latent_dim

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super(GAN, self).compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn = loss_fn
            self.d_loss_metric = Mean(name="d_loss")
            self.g_loss_metric = Mean(name="g_loss")

        @property
        def metrics(self):
            return [self.d_loss_metric, self.g_loss_metric]

        def train_step(self, real_images):
            # Sample random points in the latent space
            batch_size = tf.shape(real_images)[0]
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Decode them to fake images
            generated_images = self.generator(random_latent_vectors)

            # Combine them with real images
            combined_images = tf.concat([generated_images, real_images], axis=0)

            # Assemble labels discriminating real from fake images
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )
            # Add random noise to the labels - important trick!
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Assemble labels that say "all real images"
            misleading_labels = tf.zeros((batch_size, 1))

            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                predictions = self.discriminator(self.generator(random_latent_vectors))
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Update metrics
            self.d_loss_metric.update_state(d_loss)
            self.g_loss_metric.update_state(g_loss)
            return {
                "d_loss": self.d_loss_metric.result(),
                "g_loss": self.g_loss_metric.result(),
            }

    class GANMonitor(Callback):
        def __init__(self, num_img=3, latent_dim=128):
            self.num_img = num_img
            self.latent_dim = latent_dim

        def on_epoch_end(self, epoch, logs=None):
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images *= 255
            generated_images.numpy()
            for i in range(self.num_img):
                img = keras.preprocessing.image.array_to_img(generated_images[i])
                img.save(os.path.join(model_name, "callbacks", train_type, "generated_img_%03d_%d.png" % (epoch, i)))

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=adam_v2.Adam(learning_rate=lr),
        g_optimizer=adam_v2.Adam(learning_rate=lr),
        loss_fn=BinaryCrossentropy(),
    )

    return gan, GANMonitor(num_img=4, latent_dim=latent_dim)
