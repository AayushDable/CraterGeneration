import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Activation, Conv2D, Flatten, LeakyReLU, Dense, Reshape, Conv2DTranspose, BatchNormalization
import matplotlib.pyplot as plt
import os
import numpy as np
from gan_toolset import show_fixed_images
from IPython.display import clear_output
from gan_toolset import save_generated_images
from gan_toolset import generate_and_save_image
from unilogger import ulog
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.models import load_model
import os
import copy

class UnrolledGAN:
    def __init__(self, image_shape, noise_dim, checkpoint_path, unrolling_steps=5, gen_lr=0.0003, disc_lr=0.0003):
        self.unrolling_steps = unrolling_steps
        self.noise_dim = noise_dim
        self.image_shape = image_shape
        self.checkpoint_path = checkpoint_path
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.temp_discriminator = self.build_discriminator()

    def build_generator(self):
        inputs = tf.keras.Input(shape=(self.noise_dim,),name='Input layer')
        x = Dense(4*4*512)(inputs)
        x = Reshape((4, 4, 512))(x)
        
        # 4x4x512 -> 8x8x256
        x = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',name='C2DT_layer_1')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 8x8x256 -> 16x16x128
        x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',name='C2DT_layer_2')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 16x16x128 -> 32x32x64
        x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',name='C2DT_layer_3')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 32x32x64 -> 64x64x32
        x = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same',name='C2DT_layer_5')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 64x64x32 -> 128x128x3
        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh',name='C2DT_layer_6')(x)

        generator = tf.keras.Model(inputs, x, name="generator")
        return generator

    def build_discriminator(self):
        inputs = Input(shape=self.image_shape)

        # 128x128x3 -> 64x64x32
        x = Conv2D(32, kernel_size=5, strides=2, padding='same')(inputs)
        # x = BatchNormalization()(x)
        # x = LayerNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 64x64x32 -> 32x32x64
        x = Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = LayerNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 32x32x64 -> 16x16x128
        x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = LayerNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 16x16x128 -> 8x8x256
        x = Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = LayerNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 8x8x256 -> 4x4x512
        x = Conv2D(512, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = LayerNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        
        # 4x4x512 -> 1x8192
        x = Flatten()(x)

        # Dense layer: 1x8192 -> 1x100
        x = Dense(100,name='Disc_dense_layer_1')(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(1,activation='sigmoid',name='Discriminator_final_layer')(x)

        discriminator = tf.keras.Model(inputs, x, name="encoder")
        return discriminator

    def _gen_loss(self, generated_output):
        gen_loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(generated_output), generated_output)
        return gen_loss

    def _disc_loss(self, real_output, generated_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(real_output), real_output)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(generated_output), generated_output)
        return real_loss + generated_loss

    @tf.function
    def _train_discriminator(self, real_data, noise):
        with tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
            real_output = self.discriminator(real_data, training=True)
            generated_output = self.discriminator(generated_data, training=True)
            disc_loss = self._disc_loss(real_output, generated_output)

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def _train_generator(self, real_data, noise):
        for _ in range(self.unrolling_steps):
            self._train_discriminator(real_data, noise)

        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            generated_output = self.discriminator(generated_data, training=True)
            gen_loss = self._gen_loss(generated_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        return gen_loss

    def train(self, dataset, epochs, batch_size):
        for epoch in range(epochs):
            print(f'Epoch in progress:{epoch} out of {epochs}')
            for step, real_images in enumerate(dataset):
                noise = tf.random.normal([real_images.shape[0], self.noise_dim])
                disc_loss=self._train_discriminator(real_images, noise)

                self.temp_discriminator.set_weights(self.discriminator.get_weights())
                temp_opt = copy.deepcopy(self.disc_optimizer)

                gen_loss=self._train_generator(real_images, noise)
                self.discriminator.set_weights(self.temp_discriminator.get_weights())
                self.disc_optimizer = copy.deepcopy(temp_opt)

                if step % 100 == 0:
                    ulog.logger.info(f"Step {step}: Gen loss = {gen_loss}, Disc loss = {disc_loss}")
                    clear_output(wait=True)

            self.generator.save_weights(f'{self.checkpoint_path}/generator_weights_step_{epoch}.h5')

            generate_and_save_image(self.generator, self.noise_dim, f"{self.checkpoint_path}/UnrolledGAN_output/", f"generated_image_{epoch}.png")

            #save_generated_images(epoch, self.generator, self.noise_dim)  # Add this line to save images after every epoch

