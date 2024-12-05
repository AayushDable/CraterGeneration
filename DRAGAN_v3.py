import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, LeakyReLU, Dense, Reshape, Conv2DTranspose, BatchNormalization,MaxPooling2D
import matplotlib.pyplot as plt
import os
import numpy as np
from gan_toolset import show_fixed_images
from IPython.display import clear_output
from gan_toolset import save_generated_images
from tensorflow.keras import layers
from unilogger import ulog
from gan_toolset import generate_and_save_image
from tensorflow_addons.layers import GroupNormalization

class DRAGAN:
    def __init__(self, image_shape, noise_dim, latent_dim,image_folder,checkpoint_path):
        self.image_shape = image_shape
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        self.image_size = image_shape[0]
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.optimizer_G = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.optimizer_D = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.lambda_gp = 10.0
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        # Create Checkpoint objects
        # self.generator_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer_G, model=self.generator)
        # self.discriminator_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer_D, model=self.discriminator)

        self.checkpoint_path=checkpoint_path
        self.model_info_path=checkpoint_path
        self.image_folder=image_folder
        # Create CheckpointManager objects
        # self.generator_checkpoint_manager = tf.train.CheckpointManager(self.generator_checkpoint, f"{self.checkpoint_path}/generator_checkpoints", max_to_keep=None)
        # self.discriminator_checkpoint_manager = tf.train.CheckpointManager(self.discriminator_checkpoint, f"{self.checkpoint_path}/discriminator_checkpoints", max_to_keep=None)
        self.fixed_noise = tf.random.normal([1, latent_dim])
        self.epsilon = 0#1e-7

    # #Define the generator model
    # def build_generator(self):
    #     model = tf.keras.Sequential([
    #         layers.Dense(4*4*512, activation='relu', input_shape=(self.noise_dim,)),
    #         #4x4x512
    #         layers.Reshape((4, 4, 512)),
    #         #4x4x512 -> 8x8x256
    #         layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         #8x8x256 -> 16x16x128
    #         layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         #16x16x128 -> 32x32x64
    #         layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         #32x32x64 -> 64x64x32
    #         layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         #64x64x32 -> 128x128x3
    #         layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.Activation('tanh')
    #     ])
    #     return model

    # def build_discriminator(self):
    #     model = tf.keras.Sequential([
    #         layers.Input(shape=(self.image_size, self.image_size, 3)),
    #         #128x128x3 -> 64x64x32
    #         layers.Conv2D(32, kernel_size=5, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         MaxPooling2D((2,2),padding='same'),
    #         #64x64x32 -> 32x32x64
    #         layers.Conv2D(64, kernel_size=5, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         MaxPooling2D((2,2),padding='same'),
    #         #32x32x64 -> 16x16x128
    #         layers.Conv2D(128, kernel_size=5, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         MaxPooling2D((2,2),padding='same'),
    #         #16x16x128 -> 8x8x256
    #         layers.Conv2D(256, kernel_size=5, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         MaxPooling2D((2,2),padding='same'),
    #         #8x8x256 -> 4x4x512
    #         layers.Conv2D(512, kernel_size=5, padding='same'),
    #         #layers.BatchNormalization(),
    #         layers.LeakyReLU(alpha=0.2),
    #         MaxPooling2D((2,2),padding='same'),
    #         #4x4x512 -> 1x8192
    #         layers.Flatten(),
    #         layers.Dense(1),
    #     ])
    #     return model

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

    def generator_loss(self, fake_logits):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(tf.ones_like(fake_logits), fake_logits)

    def discriminator_loss(self, real_images, generated_images):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # Get the logits for the real images
        real_logits = self.discriminator(real_images, training=True)
        
        # Get the logits for the fake images
        fake_logits = self.discriminator(generated_images, training=True)
        
        # Calculate the standard GAN loss for the real and the fake images
        real_loss = bce(tf.ones_like(real_logits),real_logits)
        fake_loss = bce(tf.zeros_like(fake_logits),fake_logits)
        
        # Return the sum of real and fake losses
        return real_loss + fake_loss



    def gradient_penalty(self, real_data, perturbed_data):
        alpha = tf.random.uniform(shape=[real_data.shape[0]]+ [1]*(len(real_data.shape)-1), minval=0., maxval=1.)
        with tf.GradientTape() as tape:
            differences = perturbed_data - real_data
            interpolates = real_data + (alpha * differences)
            tape.watch(interpolates)
            interpolates_logits = self.discriminator(interpolates, training=True)
            gradients = tape.gradient(interpolates_logits, [interpolates])[0]
            gradients_sqr = tf.square(gradients)
            gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=tf.range(1, len(gradients_sqr.shape)))

            gradient_penalty = tf.reduce_mean((gradients_sqr_sum - 1.) ** 2)
        return gradient_penalty



    @tf.function
    def train_step(self, real_images, batch_size, noise):

        with tf.GradientTape() as tape_D, tf.GradientTape() as tape_G:
            generated_images = self.generator(noise, training=True)

            perturbation = tf.random.normal(shape=real_images.shape, mean=0., stddev=0.5)
            x_hat = real_images + 0.5 * tf.math.reduce_std(real_images) * perturbation

            # Compute gradient penalty
            gp = self.gradient_penalty(real_images, x_hat)

            loss_D = self.discriminator_loss(real_images, generated_images) + self.lambda_gp * gp

            e_g_z = self.discriminator(generated_images)
            loss_G = self.generator_loss(e_g_z)

        gradients_D = tape_D.gradient(loss_D, self.discriminator.trainable_variables)
        self.optimizer_D.apply_gradients(zip(gradients_D, self.discriminator.trainable_variables))

        gradients_G = tape_G.gradient(loss_G, self.generator.trainable_variables)
        self.optimizer_G.apply_gradients(zip(gradients_G, self.generator.trainable_variables))

        return loss_D, loss_G


    def train(self, dataset, EPOCHS, batch_size,start_epoch=0):
        for epoch in range(start_epoch,EPOCHS):
            ulog.logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
            
            for step, real_images in enumerate(dataset):
                noise = tf.random.normal([batch_size, self.noise_dim])
                loss_D, loss_G = self.train_step(real_images, batch_size, noise)

                if step % 100 == 0:
                    ulog.logger.info(f"Step {step}: loss_D = {loss_D}, loss_G = {loss_G}")
                    #show_fixed_images(self.generator,self.fixed_noise)
                    #clear_output(wait=True)

            #Save generated images to checkpoint path
            #save_generated_images(epoch, self.generator, self.noise_dim,self.checkpoint_path)
            
            self.generator.save_weights(f'{self.checkpoint_path}/generator_weights_step_{epoch}.h5')
            self.discriminator.save_weights(f'{self.checkpoint_path}/discriminator_weights_step_{epoch}.h5')
            
            generate_and_save_image(self.generator, self.noise_dim, "D:/GAN_tests/Checkpoints/DRAGAN_checkpoints/DRAGAN_output/", f"generated_image_{epoch}.png")
        
    def save_parameters(self, path):
        params = {
            'image_shape': self.image_shape,
            'noise_dim': self.noise_dim,
            'latent_dim': self.latent_dim,
            'optimizer_G': {
                'learning_rate': float(self.optimizer_G.learning_rate.numpy()),
                'name': self.optimizer_G.get_config()['name']
            },
            'optimizer_D': {
                'learning_rate': float(self.optimizer_D.learning_rate.numpy()),
                'name': self.optimizer_D.get_config()['name']
            },
            'lambda_gp': self.lambda_gp,
            'checkpoint_path': self.checkpoint_path,
            'image_folder': self.image_folder,
            'model_info_path': self.model_info_path
        }
        
        with open(path, 'w') as f:
            f.write(str(params))
