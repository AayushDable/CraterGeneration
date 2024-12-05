import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, LeakyReLU, Dense, Reshape, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow_addons.layers import GroupNormalization
import matplotlib.pyplot as plt
import os
import numpy as np
from gan_toolset import show_fixed_images
from IPython.display import clear_output
from gan_toolset import save_generated_images
from gan_toolset import generate_and_save_image
from unilogger import ulog
from gan_toolset import load_MGGAN_pretrained_encoder

class MGGAN:
    def __init__(self,image_shape,noise_dim,encoder_path,checkpoint_path):
        self.noise_dim=noise_dim
        self.encoder_path = encoder_path
        self.image_shape = image_shape
        self.checkpoint_path = checkpoint_path
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.encoder= self.build_encoder()
        self.discriminator_g = self.build_discriminator_g()
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.disc_g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        
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
    
    def build_encoder(self):
        encoder=load_MGGAN_pretrained_encoder(self.encoder_path)
        
        for layer in encoder.layers:
            layer.trainable = False

        return encoder

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


    def build_discriminator_g(self):
        inputs = Input(shape=(1024))
        x = Dense(512)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        discriminator = tf.keras.Model(inputs, outputs)
        return discriminator

    @tf.function
    def train_step(self,real_images,batch_size):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as disc_g_tape:
            noise = tf.random.normal([batch_size, self.noise_dim])
            generated_images = self.generator(noise)
            #discriminator output for real and genearted images
            disc_real_output = self.discriminator(real_images)
            disc_generated_output = self.discriminator(generated_images)

            encoded_real_images =self.encoder(real_images)
            encoded_generated_images = self.encoder(generated_images)

            #discriminator loss for real images
            real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(disc_real_output), disc_real_output)
            #discriminator loss for genearted images
            generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(disc_generated_output), disc_generated_output)
            #discriminator's total loss
            disc_loss = real_loss + generated_loss

            disc_g_real_output = self.discriminator_g(encoded_real_images)
            disc_g_generated_output = self.discriminator_g(encoded_generated_images)

            disc_g_real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(disc_g_real_output), disc_g_real_output)
            disc_g_generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(disc_g_generated_output), disc_g_generated_output)

            disc_g_loss = disc_g_real_loss + disc_g_generated_loss

            #generator loss from discriminator
            gen_loss_from_disc = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(disc_generated_output), disc_generated_output)

            gen_loss_from_disc_g = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(disc_g_generated_output), disc_g_generated_output)
            
            gen_loss = gen_loss_from_disc + gen_loss_from_disc_g
        
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))


        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        disc_g_gradients = disc_g_tape.gradient(disc_g_loss, self.discriminator_g.trainable_variables)
        self.disc_g_optimizer.apply_gradients(zip(disc_g_gradients, self.discriminator_g.trainable_variables))

        return gen_loss, disc_loss,disc_g_loss

    def train(self, dataset, EPOCHS, batch_size,start_epoch=0):
        for epoch in range(start_epoch,EPOCHS):
            ulog.logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
            
            for step, real_images in enumerate(dataset):

                gen_loss, disc_loss,disc_g_loss = self.train_step(real_images, real_images.shape[0])

                if step % 100 == 0:
                    ulog.logger.info(f"Step {step}: Gen loss = {gen_loss}, Disc loss = {disc_loss}, Guidance Disc loss = {disc_g_loss}")
                    clear_output(wait=True)
            
            self.generator.save_weights(f'{self.checkpoint_path}/generator_weights_step_{epoch}.h5')

            generate_and_save_image(self.generator, self.noise_dim, f"{self.checkpoint_path}/MGGAN_output/", f"generated_image_{epoch}.png")
