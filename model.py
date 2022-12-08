import os
import datetime
import time

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    ConvLSTM2D,
    BatchNormalization,
    Activation,
    MaxPool3D,
    MaxPool2D,
    Concatenate,
    Input,
    Conv3D,
    Conv2DTranspose,
    Conv3DTranspose,
    Conv2D,
    LeakyReLU,
    Flatten,
    Dense,
    Dropout,
    UpSampling2D,
    UpSampling3D,
)
import tensorflow as tf
from tensorflow import keras


class UNet(keras.Model):
    def __init__(self, window_size=5, activation=keras.layers.LeakyReLU):
        super().__init__()
        self.window_size = window_size
        self.activation = activation

    def conv_block_lstm(self, input_layer, num_filters):
        x = Bidirectional(
            ConvLSTM2D(
                filters=num_filters,
                kernel_size=(5, 5),
                padding="same",
                return_sequences=True,
            )
        )(input_layer)
        x = BatchNormalization()(x)
        x = self.activation()(x)

        x = Bidirectional(
            ConvLSTM2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=True,
            )
        )(input_layer)
        x = BatchNormalization()(x)
        x = self.activation()(x)

        return x

    def conv_block(self, input_layer, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = self.activation()(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = self.activation()(x)

        return x

    def encoder_block(self, input_layer, num_filters):
        x = self.conv_block_lstm(input_layer, num_filters)
        p = MaxPool3D((1, 2, 2))(x)
        x = Conv3D(filters=num_filters, kernel_size=(self.window_size, 1, 1))(x)
        x = tf.keras.backend.squeeze(x, axis=1)
        x = Conv2DTranspose(filters=num_filters, strides=(4, 4), kernel_size=(1, 1))(x)
        return x, p

    def decoder_block(self, input_layer, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def get_model(self, input_shape):
        inputs = Input(shape=input_shape)

        s1, p1 = self.encoder_block(inputs, 32)
        s2, p2 = self.encoder_block(p1, 64)
        # s3, p3 = self.encoder_block(p2, 64)
        # s4, p4 = self.encoder_block(p3, 64)

        b1 = self.conv_block_lstm(p2, 128)
        b1 = Conv3D(filters=128, kernel_size=(self.window_size, 1, 1))(b1)
        b1 = tf.keras.backend.squeeze(b1, axis=1)
        b1 = Conv2DTranspose(filters=128, strides=(4, 4), kernel_size=(1, 1))(b1)

        # d1 = self.decoder_block(b1, s3, 64)
        # d1 = self.decoder_block(b1, s3, 64)
        d1 = self.decoder_block(b1, s2, 64)
        d2 = self.decoder_block(d1, s1, 32)

        outputs = Conv2D(3, 1, padding="same", activation='sigmoid')(d2)

        model = Model(inputs, outputs, name="UNet")

        return model


def get_discriminator_model(
    input_shape, initializer=tf.random_normal_initializer(0.0, 0.02)
):

    input_image = Input(shape=input_shape, name="input_image")

    b1 = Conv2D(
        64, (5, 5), strides=(2, 2), kernel_initializer=initializer, padding="same"
    )(input_image)
    b1 = BatchNormalization()(b1)
    b1 = LeakyReLU()(b1)
    b1 = Dropout(0.3)(b1)

    b2 = Conv2D(
        128, (5, 5), strides=(2, 2), kernel_initializer=initializer, padding="same"
    )(b1)
    b2 = BatchNormalization()(b2)
    b2 = LeakyReLU()(b2)
    b2 = Dropout(0.3)(b2)

    b3 = Conv2D(
        256, (3, 3), strides=(2, 2), kernel_initializer=initializer, padding="same"
    )(b2)
    b3 = BatchNormalization()(b3)
    b3 = LeakyReLU()(b3)
    b3 = Dropout(0.3)(b3)

    b4 = Conv2D(
        64, (3, 3), strides=(2, 2), kernel_initializer=initializer, padding="same"
    )(b3)
    b4 = BatchNormalization()(b4)
    b4 = LeakyReLU()(b4)
    b4 = Dropout(0.3)(b4)

    output = Conv2D(1, 4, strides=1, kernel_initializer=initializer, padding="same")(b4)
    return Model(inputs=input_image, outputs=output)


class Fit:
    def __init__(self, input_shape, output_shape, checkpoint_dir = './training_checkpoints/', checkpoint_ext = 'ckpt', window_size=5, load_checkpoint = None):
        self.window_size = window_size
        self.generator = UNet().get_model((self.window_size, *input_shape))
        self.discriminator = get_discriminator_model(output_shape)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_ext = checkpoint_ext
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, self.checkpoint_ext)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        if load_checkpoint:
            self.checkpoint.restore(self.checkpoint_dir + load_checkpoint).expect_partial()
            if self.checkpoint:
                print("\n\n %%%%%% Loaded Checkpoint %%%%%% \n\n")

        self.log_dir = "./logs/"
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit-2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator(target, training=True)
            disc_generated_output = self.discriminator(
                gen_output, training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step//48)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step//48)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step//48)
            tf.summary.scalar("disc_loss", disc_loss, step=step//48)

    def generator_loss(
        self,
        d_out,
        g_out,
        target,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        alpha=100,
    ):

        gan_loss = loss(tf.ones_like(d_out), d_out)
        l1_loss = tf.reduce_mean(tf.abs(target - g_out))
        total_gen_loss = gan_loss + (alpha * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(
        self, d_r_out, d_g_out, loss=keras.losses.BinaryCrossentropy(from_logits=True)
    ):
        real_loss = loss(tf.ones_like(d_r_out), d_r_out)
        generated_loss = loss(tf.zeros_like(d_g_out), d_g_out)
        return real_loss + generated_loss


if __name__ == "__main__":
    generator = UNet().get_model((5, 180, 320, 3))
    discriminator = get_discriminator_model([720, 1280, 3])
    print(generator.summary())
    print(discriminator.summary())
