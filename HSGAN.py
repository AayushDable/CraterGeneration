import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Conv2D, Flatten, LeakyReLU, Dense, Reshape, Conv2DTranspose, BatchNormalization
import matplotlib.pyplot as plt
import os
import numpy as np
from gan_toolset import show_fixed_images
from IPython.display import clear_output
from gan_toolset import save_generated_images
from gan_toolset import generate_and_save_image
from unilogger import ulog
from tensorflow_addons.layers import GroupNormalization

class HSGAN:
    def __init__(self, image_shape, noise_dim, latent_dim,image_folder,checkpoint_path):
        self.image_shape = image_shape
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.image_size = image_shape[0]
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()
        
        self.optimizer_G = tf.keras.optimizers.RMSprop(learning_rate=0.0003)
        self.optimizer_E = tf.keras.optimizers.RMSprop(learning_rate=0.0003)
        self.lambda_gp = 10.0
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        # Create Checkpoint
        self.checkpoint_path=checkpoint_path
        self.model_info_path=checkpoint_path
        self.image_folder=image_folder

        self.fixed_noise = tf.random.normal([1, latent_dim])

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
        x = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',name='C2DT_layer_6')(x)
        x = Activation('tanh')(x)
        generator = tf.keras.Model(inputs, x, name="generator")
        return generator

    def build_encoder(self):
        inputs = Input(shape=self.image_shape)

        # If input is 64x64x3 -> 32x32x32 or 128x128x3 -> 64x64x32
        x = Conv2D(32, kernel_size=5, strides=2, padding='same')(inputs)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # If input is 32x32x32 -> 16x16x64 or 64x64x32 -> 32x32x64
        x = Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # If input is 16x16x64 -> 8x8x128 or 32x32x64 -> 16x16x128
        x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # If input is 8x8x128 -> 4x4x256 or 16x16x128 -> 8x8x256
        x = Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        #Only for 128x128 images, 8x8x256 -> 4x4x512
        x = Conv2D(512, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # If input is 4x4x256 -> 1x4096 or 4x4x512 -> 1x8192
        x = Flatten()(x)   

        # Dense layer2: -> 1x100
        x = Dense(100,name='Encoder_dense_layer_2')(x)

        encoder = tf.keras.Model(inputs, x, name="encoder")
        return encoder

    def generator_loss(self, e_g_z1, e_g_z2):
        term = self.mse_loss(e_g_z1, e_g_z2) - 1
        loss = tf.square(term)
        return loss

    def encoder_loss(self, x1, x2, z1, z2, x_hat):
        #encoded representations of real images
        e_x1 = self.encoder(x1)
        e_x2 = self.encoder(x2)
        #encoded representations of generated images
        e_g_z1 = self.encoder(self.generator(z1))
        e_g_z2 = self.encoder(self.generator(z2))
        #Mean square error between encoded representations of real images
        term1 = self.mse_loss(e_x1, e_x2) - 1
        #Mean square error between encoded representations of generated images
        term2 = self.mse_loss(e_g_z1, e_g_z2)
        #Gradient penalty term to be fed to the encoder
        term3 = self.gradient_penalty(x_hat) * self.lambda_gp
        #Mean square error between encoded representation of generated image and the original noise profile
        term4 = self.mse_loss(e_g_z1, z1) + self.mse_loss(e_g_z2, z2)

        #Encoder loss to be summation of all terms, as proprosed by HSGAN paper
        loss = tf.square(term1) + tf.square(term2) + term3 + term4
        return loss

    def gradient_penalty(self, x_hat):
        with tf.GradientTape() as tape_gp:
            tape_gp.watch(x_hat)
            e_x_hat = self.encoder(x_hat)
        
        #Calculate gradient of encoded image e_x_hat with change in x_hat
        gradients = tape_gp.gradient(e_x_hat, x_hat)
        #square the gradient
        gradients_sqr = tf.square(gradients)
        #sum gradient along width height and channels
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3])
        #calculate gradient penalty
        gradient_penalty = tf.square(tf.sqrt(gradients_sqr_sum) - 1)
            
        return tf.reduce_mean(gradient_penalty)


    @tf.function
    def train_step(self, real_images, batch_size):
        batch_size=real_images.shape[0]
        x1, x2 = real_images[:batch_size // 2], real_images[batch_size // 2:]
        x2 = real_images[batch_size//2:][:x1.shape[0]]
        batch_size=x1.shape[0]*2
        z1, z2 = tf.random.normal([batch_size // 2, self.noise_dim]), tf.random.normal([batch_size // 2, self.noise_dim])

        with tf.GradientTape() as tape_E:
            generated_images1 = self.generator(z1, training=True)
            generated_images2 = self.generator(z2, training=True)

            epsilon = tf.random.uniform([batch_size // 2, 1, 1, 1], 0, 1)

            #Create interpolated images for calculating gradient penalty
            x_hat1 = epsilon * x1 + (1 - epsilon) * generated_images1
            x_hat2 = epsilon * x2 + (1 - epsilon) * generated_images2
            x_hat = tf.concat([x_hat1, x_hat2], axis=0)

            # Compute gradient penalty
            gp1 = self.gradient_penalty(x_hat1)
            gp2 = self.gradient_penalty(x_hat2)

            loss_E = self.encoder_loss(x1, x2, z1, z2, x_hat) + gp1 + gp2

        gradients_E = tape_E.gradient(loss_E, self.encoder.trainable_variables)
        self.optimizer_E.apply_gradients(zip(gradients_E, self.encoder.trainable_variables))

        with tf.GradientTape() as tape_G:
            generated_images1 = self.generator(z1, training=True)
            generated_images2 = self.generator(z2, training=True)
            e_g_z1 = self.encoder(generated_images1)
            e_g_z2 = self.encoder(generated_images2)
            loss_G = self.generator_loss(e_g_z1, e_g_z2)

        gradients_G = tape_G.gradient(loss_G, self.generator.trainable_variables)
        self.optimizer_G.apply_gradients(zip(gradients_G, self.generator.trainable_variables))

        return loss_E, loss_G



    # @tf.function
    # def train_step(self, real_images, batch_size, noise_size):
    #     half_batch = batch_size // 2
    #     half_batch_noise_dim = [half_batch, self.noise_dim]

    #     x1, x2 = real_images[:half_batch], real_images[half_batch:]
    #     z1, z2 = tf.random.normal(half_batch_noise_dim), tf.random.normal(half_batch_noise_dim)

    #     epsilon = tf.random.uniform([half_batch, 1, 1, 1], 0, 1)
        
    #     with tf.GradientTape() as tape_E, tf.GradientTape() as tape_G:
    #         generated_images1 = self.generator(z1, training=True)
    #         generated_images2 = self.generator(z2, training=True)

    #         x_hat1 = epsilon * x1 + (1 - epsilon) * generated_images1
    #         x_hat2 = epsilon * x2 + (1 - epsilon) * generated_images2
    #         x_hat = tf.concat([x_hat1, x_hat2], axis=0)

    #         gp1 = self.gradient_penalty(x_hat1)
    #         gp2 = self.gradient_penalty(x_hat2)
            
    #         loss_E = self.encoder_loss(x1, x2, z1, z2,x_hat) + gp1 + gp2

    #         e_g_z1 = self.encoder(generated_images1)
    #         e_g_z2 = self.encoder(generated_images2)
    #         loss_G = self.generator_loss(e_g_z1, e_g_z2)

    #     gradients_E = tape_E.gradient(loss_E, self.encoder.trainable_variables)
    #     self.optimizer_E.apply_gradients(zip(gradients_E, self.encoder.trainable_variables))

    #     gradients_G = tape_G.gradient(loss_G, self.generator.trainable_variables)
    #     self.optimizer_G.apply_gradients(zip(gradients_G, self.generator.trainable_variables))

    #     return loss_E, loss_G


    def train(self, dataset, EPOCHS, batch_size,start_epoch=0):
        for epoch in range(start_epoch,EPOCHS):
            ulog.logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
            
            for step, real_images in enumerate(dataset):
                if real_images.shape[0]!=1:
                    loss_E, loss_G = self.train_step(real_images, batch_size)

                if step % 100 == 0:
                    ulog.logger.info(f"Step {step}: loss_E = {loss_E}, loss_G = {loss_G}")
                    #show_fixed_images(self.generator,self.fixed_noise)
                    clear_output(wait=True)

            #Save generated images to checkpoint path
            #save_generated_images(epoch, self.generator, self.noise_dim,self.checkpoint_path)
            
            self.generator.save_weights(f'{self.checkpoint_path}/generator_weights_step_{epoch}.h5')
            self.encoder.save_weights(f'{self.checkpoint_path}/encoder_weights_step_{epoch}.h5')

            generate_and_save_image(self.generator, self.noise_dim, f"{self.checkpoint_path}/HSGAN_output/", f"generated_image_{epoch}.png")

    def save_parameters(self, path):
        params = {
            'image_shape': self.image_shape,
            'noise_dim': self.noise_dim,
            'latent_dim': self.latent_dim,
            'optimizer_G': {
                'learning_rate': float(self.optimizer_G.learning_rate.numpy()),
                'name': self.optimizer_G.get_config()['name']
            },
            'optimizer_E': {
                'learning_rate': float(self.optimizer_E.learning_rate.numpy()),
                'name': self.optimizer_E.get_config()['name']
            },
            'lambda_gp': self.lambda_gp,
            'checkpoint_path': self.checkpoint_path,
            'image_folder': self.image_folder,
            'model_info_path': self.model_info_path
        }
        
        with open(path, 'w') as f:
            f.write(str(params))
