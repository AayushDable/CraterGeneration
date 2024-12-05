import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from gan_toolset import generate_and_show_images
from IPython.display import clear_output
from gan_toolset import show_fixed_images
import matplotlib.pyplot as plt
from gan_toolset import generate_and_save_image
import os
from unilogger import ulog
from tensorflow_addons.layers import GroupNormalization

class VAEGAN:
    def __init__(self, input_shape, latent_dim,checkpoint_path):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        # self.lth_layer_index = lth_layer_index
        self.image_size = input_shape[0]
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.discriminator = self.build_discriminator()

        # self.vae_optimizer = optimizers.Adam()
        # self.discriminator_optimizer = optimizers.Adam()
        self.vae_optimizer = optimizers.RMSprop(learning_rate=0.0003)
        self.discriminator_optimizer = optimizers.RMSprop(learning_rate=0.0003)

        self.checkpoint_path = checkpoint_path
        self.fixed_noise = tf.random.normal([1, latent_dim])

    def build_encoder(self):
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        # If input is 64x64x3 -> 32x32x32 or 128x128x3 -> 64x64x32
        x = layers.Conv2D(32, kernel_size=5, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # If input is 32x32x32 -> 16x16x64 or 64x64x32 -> 32x32x64
        x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # If input is 16x16x64 -> 8x8x128 or 32x32x64 -> 16x16x128
        x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # If input is 8x8x128 -> 4x4x256 or 16x16x128 -> 8x8x256
        x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        #Only for 128x128 images, 8x8x256 -> 4x4x512
        x = layers.Conv2D(512, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # If input is 4x4x256 -> 1x4096 or 4x4x512 -> 1x8192
        x = layers.Flatten()(x)

        #1x8192 -> 1x100
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)

        return Model(inputs, [z_mean, z_log_var])

    def build_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(4*4*512, activation='relu')(inputs)
        x = layers.Reshape((4, 4, 512))(x)
        
        # 4x4x512 -> 8x8x256
        x = layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(x)
        # x = layers.BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # 8x8x256 -> 16x16x128
        x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
        # x = layers.BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # 16x16x128-> 32x32x64
        x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
        # x = layers.BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)     

        # 32x32x64-> 64x64x32
        x = layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)
        # x = layers.BatchNormalization()(x)
        x = GroupNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)       

        # 64x64x32-> 128x128x3
        x = layers.Conv2DTranspose(self.input_shape[-1], kernel_size=5, strides=2, padding='same')(x)
        x_tilde = layers.Activation('tanh')(x)

        return Model(inputs, x_tilde)

    def build_discriminator(self):
        inputs = layers.Input(shape=self.input_shape)

        #128x128x3 -> 64x64x32
        x = layers.Conv2D(32, kernel_size=5, strides=2, padding='same')(inputs)
        x = GroupNormalization()(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        #64x64x32 -> 32x32x64
        x = layers.Conv2D(64, kernel_size=5,strides=2, padding='same')(x)
        x = GroupNormalization()(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        #32x32x64 -> 16x16x128
        x = layers.Conv2D(128, kernel_size=5,strides=2, padding='same')(x)
        x = GroupNormalization()(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        #16x16x128 -> 8x8x256
        x = layers.Conv2D(256, kernel_size=5,strides=2, padding='same')(x)
        x = GroupNormalization()(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        #8x8x256 -> 4x4x512
        x = layers.Conv2D(512, kernel_size=5,strides=2, padding='same')(x)
        x = GroupNormalization()(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)    

        #4x4x512 -> 1x8192
        x = layers.Flatten()(x)
        hidden_representation = layers.Dense(100, activation='relu')(x)
        # x = layers.BatchNormalization()(hidden_representation)
        x = layers.LeakyReLU(alpha=0.2)(hidden_representation) 
        x = layers.Dropout(0.3)(x)
        final_output = layers.Dense(1, activation='sigmoid')(x)

        return Model(inputs, [final_output, hidden_representation])

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    def compute_GAN_loss(self,x, z):
        # Compute the discriminator's output for real, generated, and reconstructed samples
        real_output = self.discriminator(x)[0]
        generated_output = self.discriminator(self.decoder(z))[0]
        reconstructed_output = self.discriminator(self.decoder(self.sampling((self.encoder(x)))))[0]

        # Compute the binary cross entropy for each part
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(real_output), real_output)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(generated_output), generated_output)
        reconstructed_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(reconstructed_output), reconstructed_output)

        # Compute the total loss
        GAN_loss = real_loss + generated_loss + reconstructed_loss

        return GAN_loss
    

    def compute_LDisLlike(self,real_images, decoded_images):
    # Obtain the discriminator outputs for real and decoded images
        _, dis_real_output = self.discriminator(real_images)
        _, dis_decoded_output = self.discriminator(decoded_images)

        # Compute the Gaussian log-likelihood
        LDisLlike = tf.reduce_mean(tf.square(dis_real_output - dis_decoded_output))

        return LDisLlike

    
    @tf.function
    def train_step(self, real_images, gamma):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:

            #Encode the real images, getting mean and variance vectors
            z_mean, z_log_var = self.encoder(real_images)

            #use the parameterization trick to get distirbution that represent z_mean,z_log_var
            z = self.sampling((z_mean, z_log_var))

            #Reconstructed images using decoder
            x_tilde = self.decoder(z)

            #This is the reconstruction loss but computed using lth layer of discriminator
            LDisLlike = self.compute_LDisLlike(real_images,x_tilde)

            #Calculate the kl_divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            #calculate the total GAN loss using real and fake images
            GAN_loss=self.compute_GAN_loss(real_images,z)

            #encoder loss computed using Lprior (kl_loss) and reconstruction loss (lDisLlike)
            enc_loss = tf.reduce_mean(kl_loss) + tf.reduce_mean(LDisLlike)

            #decoder loss
            dec_loss = gamma * tf.reduce_mean(LDisLlike) - GAN_loss

            

        enc_gradients = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        self.vae_optimizer.apply_gradients(zip(enc_gradients, self.encoder.trainable_variables))


        dec_gradients = dec_tape.gradient(dec_loss, self.decoder.trainable_variables)
        self.vae_optimizer.apply_gradients(zip(dec_gradients, self.decoder.trainable_variables))

        disc_gradients = disc_tape.gradient(GAN_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return enc_loss, dec_loss,GAN_loss

    def train(self, dataset, epochs, batch_size, gamma=1.0):
        for epoch in range(epochs):
            ulog.logger.info(f'Epoch {epoch + 1}/{epochs}')
            for step, real_images in enumerate(dataset):
                # Train the VAE
                enc_loss, dec_loss,GAN_loss = self.train_step(real_images, gamma)

                if step % 100 == 0:
                    ulog.logger.info(f'Step {step}, enc Loss: {enc_loss.numpy()}, dec loss: {dec_loss.numpy()}, Discriminator Loss: {GAN_loss.numpy()}')
                    #show_fixed_images(self.decoder,self.fixed_noise)
                    #clear_output(wait=True)
                    generate_and_save_image(self.decoder, self.latent_dim, "D:/GAN_tests/Checkpoints/VAEGAN_checkpoints/VAEGAN_output/", f"generated_image_{epoch}.png")

            self.decoder.save_weights(f'{self.checkpoint_path}/decoder_weights_step_{epoch}.h5')
            self.discriminator.save_weights(f'{self.checkpoint_path}/discriminator_weights_step_{epoch}.h5')


