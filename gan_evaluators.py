import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
import cv2
from scipy.spatial.distance import cosine
from itertools import combinations
from scipy.fftpack import fft2, fftshift
import tensorflow as tf
import concurrent.futures
from skimage.metrics import structural_similarity as ssim
from itertools import combinations
from scipy.stats import wasserstein_distance
from IPython.display import clear_output
from VAE_layer_reduced import VAE
from tqdm import tqdm
from scipy.linalg import sqrtm
from tensorflow.keras import layers, Model, optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from scipy.stats import entropy

# def ndb_score(real_images, generated_images,sample_size,noise_dim, K=5, significance_level=0.1):
    
#     #take 1st n=sample_size images from real and generated data
#     real_data = real_images[:sample_size]
#     generated_data = generated_images[:sample_size]

#     #convert to tensors
#     real_data = np.array(real_data)
#     generated_data = np.array(generated_data)

#     # Flatten the spatial dimensions of the images
#     real_data_flat = real_data.reshape(real_data.shape[0], -1)
#     generated_data_flat = generated_data.reshape(generated_data.shape[0], -1)

#     # Step 1: Divide the training samples into K bins using K-means clustering
#     kmeans = KMeans(n_clusters=K).fit(real_data_flat)
    
#     # Step 2: Allocate the generated samples to the closest bin
#     generated_bin_indices = kmeans.predict(generated_data_flat)

#     # Step 3: For each of the K bins, conduct a two-sample test between the bin's real and generated samples to obtain a z-score
#     statistically_different_bins = 0
#     for i in range(K):
#         p1 = sum(kmeans.labels_ == i)/len(kmeans.labels_)
#         n1 = sum(kmeans.labels_ == i) #real_bin_len
        
#         p2 = sum(generated_bin_indices == i)/len(generated_bin_indices)
#         n2 = sum(generated_bin_indices == i) #gen_bin_len

#         total_size = p1 + p2 #real_bin_size + generated_bin_size

#         total_len = n1 + n2 #real_bin_len + gen_bin_len

#         p = (p1 * n1 + p2 * n2) / (n1 + n2)
#         try:
#             se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
#             z_score = (p1 - p2) / se

#             # Calculate the two-tailed p-value from the z-score
#             p_value = 2 * norm.sf(np.abs(z_score)) # sf = 1 - cdf

#         except:
#             #this is the case where the generated dist did not cluster to
#             #the particular bin, meaning the bin is statistically diff.
#             p_value = 0
        
#         # If the z-score is smaller than the threshold, mark the bin as statistically different
#         #print(z_score)
#         if p_value < significance_level:
#             statistically_different_bins += 1
#         #print(z_score)
#     # Step 4: Count the total number of statistically different bins and divide by K (number of bins)
#     ndb = statistically_different_bins / K
#     #print(f'2_{statistically_different_bins}')
#     return ndb

def ndb_score(real_images, generated_images,sample_size,noise_dim, K=5, significance_level=0.01,min_bin_count=10):
    #take 1st n=sample_size images from real and generated data
    real_data = real_images[:sample_size]
    generated_data = generated_images[:sample_size]

    #convert to tensors
    real_data = np.array(real_data)
    generated_data = np.array(generated_data)

    # Flatten the spatial dimensions of the images
    real_data_flat = real_data.reshape(real_data.shape[0], -1)
    generated_data_flat = generated_data.reshape(generated_data.shape[0], -1)

    # Step 1: Divide the training samples into K bins using K-means clustering
    kmeans = KMeans(n_clusters=K).fit(real_data_flat)

    real_bin_indices = kmeans.labels_
    real_cluster_counts = np.bincount(real_bin_indices)
    
    # Step 2: Allocate the generated samples to the closest bin
    generated_bin_indices = kmeans.predict(generated_data_flat)
    generated_cluster_counts = np.bincount(generated_bin_indices, minlength=K)

    # Step 3: For each of the K bins, conduct a two-sample test between the bin's real and generated samples to obtain a z-score
    statistically_different_bins = 0
    for i in range(K):
        p1 = sum(kmeans.labels_ == i)/len(kmeans.labels_)
        n1 = sum(kmeans.labels_ == i) #real_bin_len
        
        p2 = sum(generated_bin_indices == i)/len(generated_bin_indices)
        n2 = sum(generated_bin_indices == i) #gen_bin_len

        total_size = p1 + p2 #real_bin_size + generated_bin_size

        total_len = n1 + n2 #real_bin_len + gen_bin_len

        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Check if there are no generated images in this bin
        if n2 == 0:
            if n1 > min_bin_count:
                statistically_different_bins += 1
        else:
            try:
                se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
                z_score = (p1 - p2) / se

                # Calculate the two-tailed p-value from the z-score
                p_value = 2 * norm.sf(np.abs(z_score)) # sf = 1 - cdf
                
                # If the z-score is smaller than the threshold, mark the bin as statistically different
                if p_value < significance_level:
                    statistically_different_bins += 1

            except:
                pass # Handle any other unexpected exceptions
        
    # Step 4: Count the total number of statistically different bins and divide by K (number of bins)
    ndb = statistically_different_bins / K

    return ndb

# def average_gradient_magnitude(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Compute the gradients in the x and y directions
#     sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

#     # Compute the gradient magnitude
#     gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

#     # Calculate the average gradient magnitude
#     avg_gradient_magnitude = np.mean(gradient_magnitude)

#     # # Normalize the score
#     # max_gradient_magnitude = 255 * np.sqrt(2)
#     # normalized_score = avg_gradient_magnitude / max_gradient_magnitude

#     return avg_gradient_magnitude

# def gen_images_sharpness(generator, num_images, noise_dim):
#     #Generate n(num_images) noise vectors of size (noise_dim)
#     noise = np.random.normal(0, 1, (num_images, noise_dim))
#     # Generate images using the generator model
#     generated_images = generator.predict(noise)
    
#     # Ensure that the images are in the range [0, 255]
#     # generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)
#     generated_images = (generated_images * 255).astype(np.uint8)  

#     # Calculate sharpness scores for each image and return the average sharpness score
#     sharpness_scores = [average_gradient_magnitude(image) for image in generated_images]
#     average_sharpness = np.mean(sharpness_scores)
    
#     return average_sharpness

# def original_images_sharpness(dataset):
#     sharpness_scores = []

#     for batch in dataset:
#         for image in batch.numpy():
#             # To ensure that the image is in the range [0, 255]
#             # image = np.clip(image, 0, 255).astype(np.uint8)
#             image = (image * 255).astype(np.uint8)  
#             sharpness_scores.append(average_gradient_magnitude(image))

#     average_sharpness = np.mean(sharpness_scores)
#     return average_sharpness

# Improved sharpness difference score
def sharpness_difference(real_images, generated_images,sample_size):
    #take 1st n=sample_size images from real and generated data
    original_images = real_images[:sample_size]
    generated_images = generated_images[:sample_size]

    #convert to tensors
    original_images = np.array(original_images)
    generated_images = np.array(generated_images)

    SD = []
    for original, generated in zip(original_images, generated_images):
        # Convert to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        original_grad_x = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
        original_grad_y = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
        generated_grad_x = cv2.Sobel(generated_gray, cv2.CV_64F, 1, 0, ksize=3)
        generated_grad_y = cv2.Sobel(generated_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient differences
        grad_diff_x = np.abs(original_grad_x - generated_grad_x)
        grad_diff_y = np.abs(original_grad_y - generated_grad_y)
        
        # Compute GRADSI,K
        gradsi_k = np.mean(grad_diff_x + grad_diff_y)
        
        # Compute SD
        sd = 10 * np.log10(np.max(original_gray) ** 2 / gradsi_k)
        SD.append(sd)
        
    return np.mean(SD)

from skimage.metrics import structural_similarity as compare_ssim

def calculate_average_ssim(real_images, generated_images,sample_size):
    #take 1st n=sample_size images from real and generated data
    real_images = real_images[:sample_size]
    generated_images = generated_images[:sample_size]

    #convert to tensors
    real_images = tf.stack(real_images)
    generated_images = tf.stack(generated_images)

    ssim_scores = []
    for real_image, generated_image in zip(real_images, generated_images):
        # Ensure the images are in the range [0, 1]
        real_image = real_image * 0.5 + 0.5
        generated_image = generated_image * 0.5 + 0.5
        # Calculate SSIM score for this pair of images
        ssim_score = compare_ssim(real_image.numpy(), generated_image.numpy(), multichannel=True, channel_axis=-1, data_range=1)
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)

# def birthday_paradox_test(generator, noise_dim, num_samples, similarity_threshold):
#     # Generate images
#     noise = np.random.normal(0, 1, (num_samples, noise_dim))
#     generated_images = generator.predict(noise)

#     # Normalize images to [0, 1] range
#     generated_images = (generated_images + 1) / 2.0

#     # Calculate pairwise cosine similarity
#     similarities = []
#     for img1, img2 in combinations(generated_images, 2):
#         img1_flat = img1.flatten()
#         img2_flat = img2.flatten()
#         similarity = 1 - cosine(img1_flat, img2_flat)
#         similarities.append(similarity)

#     # Count image pairs with similarity above the threshold
#     num_collisions = sum(1 for sim in similarities if sim >= similarity_threshold)

#     # Calculate the proportion of collisions
#     collision_proportion = num_collisions / len(similarities)

#     return collision_proportion
# import concurrent.futures

def compute_similarity(pair):
    img1, img2 = pair
    return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=1)

# def birthday_paradox_test_ssim(generator, noise_dim, num_samples, similarity_threshold):
#     # Generate images
#     noise = np.random.normal(0, 1, (num_samples, noise_dim))
#     generated_images = generator.predict(noise)

#     # Normalize images to [0, 1] range
#     generated_images = (generated_images + 1) / 2.0

#     # Calculate pairwise SSIM
#     pairs = list(combinations(generated_images, 2))

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         similarities = list(executor.map(compute_similarity, pairs))

#     # Count image pairs with similarity above the threshold
#     num_collisions = sum(1 for sim in similarities if sim >= similarity_threshold)

#     # Calculate the proportion of collisions
#     collision_proportion = num_collisions / len(similarities)

#     return collision_proportion

# def high_frequency_energy(dataset, generator, noise_dim, sample_size=100, threshold=0.5):
#     hf_energy_original = []
#     hf_energy_generated = []

#     count=0
#     for i, batch in enumerate(dataset):
#         for j, image in enumerate(batch):
#             if count >= sample_size:
#                 break
#             original_image = image.numpy()
#             original_image = (original_image + 1) / 2
#             original_image_gray = np.mean(original_image, axis=-1)
#             original_fft = fftshift(fft2(original_image_gray))
#             # Define a high-pass filter with the given threshold
#             rows, cols = original_fft.shape
#             crow, ccol = round(rows / 2), round(cols / 2)
#             threshold_pixels = round((threshold * rows) / 2)  # Convert threshold to pixels
#             mask = np.zeros((rows, cols), dtype=np.uint8)
#             mask[crow - threshold_pixels:crow + threshold_pixels, ccol - threshold_pixels:ccol + threshold_pixels] = 1
#             high_pass_filter = 1 - mask
#             # Apply the high-pass filter
#             original_fft_filtered = original_fft * high_pass_filter
#             # Calculate the high-frequency energy
#             hf_energy_original.append(np.sum(np.abs(original_fft_filtered) ** 2))
#             count += 1

#     for i in range(sample_size):
#         noise = tf.random.normal([1, noise_dim])
#         generated_image = generator(noise, training=False)
#         generated_image = (generated_image + 1) / 2
#         generated_image_gray = np.mean(generated_image, axis=-1)
#         generated_fft = fftshift(fft2(generated_image_gray))
#         generated_fft_filtered = generated_fft * high_pass_filter
#         hf_energy_generated.append(np.sum(np.abs(generated_fft_filtered) ** 2))

#     hf_energy_original_mean = np.mean(hf_energy_original)
#     hf_energy_generated_mean = np.mean(hf_energy_generated)

#     return hf_energy_original_mean, hf_energy_generated_mean


#Used for custom similarity metric
def euclidean_distance(tensor1, tensor2):
    diff = tf.subtract(tensor1, tensor2)
    squared_diff = tf.square(diff)
    distance = tf.sqrt(tf.reduce_sum(squared_diff))
    return distance
#Used for custom similarity metric
def cosine_similarity(tensor1, tensor2):
    tensor1_normalized = tf.nn.l2_normalize(tensor1, axis=-1)
    tensor2_normalized = tf.nn.l2_normalize(tensor2, axis=-1)
    similarity = tf.reduce_sum(tf.multiply(tensor1_normalized, tensor2_normalized))
    return similarity
#Used for custom similarity metric
def earth_movers_distance(tensor1, tensor2):
    tensor1 = tensor1.numpy()
    tensor2 = tensor2.numpy()
    emd = wasserstein_distance(tensor1, tensor2)
    return emd

#custom similarity metric
def evaluate_similarity(distance1, distance2, threshold):
    distance1=tf.squeeze(distance1)
    distance2=tf.squeeze(distance2)
    # relative_difference = euclidean_distance(distance1,distance2).numpy()
    relative_difference = cosine_similarity(distance1,distance2).numpy()
    #relative_difference = earth_movers_distance(distance1,distance2)
    #relative_difference = tf.abs(distance2 - distance1) / distance1
    #print(relative_difference)
    if relative_difference>threshold:
        condition_met=1
    else:
        condition_met=0

    # condition_met = tf.reduce_all(relative_difference <= threshold)

    if condition_met == 1:
        return 1,relative_difference
    else:
        return 0,relative_difference

def clustering_based_similarity_evaluator(generated_images,inception_model_path,sample_size,similarity_threshold):
    #vae = VAE(image_shape,noise_dim)
    #encoder=vae.encoder
    #encoder.load_weights(encoder_path)

    #take 1st n=sample_size images from real and generated data
    generated_images = generated_images[:sample_size]

    #convert to tensors
    generated_images = tf.stack(generated_images)

    encoder = inception_trained_model_loader(inception_model_path)
    #Create empty arrays to fill on later
    groups = []
    images_that_represent_groups = []
    distance_set = []
    images_with_assigned_groups= [[]]

    #Iterate over the required sample size, an images is generated every iteration
    for i in tqdm(range(sample_size)):
        #noise = tf.random.normal([1, noise_dim])
        image = generated_images[i]
        
        # Rescale the generated image from (-1,1) to (0,1)
        image = (image + 1.) / 2.

        #image is converted into an encoded representation,call this as distance2
        distance2 = encoder(tf.expand_dims(image,axis=0),training=False)

        #if it is the first image, add it as a new group 
        if i == 0:
            images_that_represent_groups.append(image)
            images_with_assigned_groups[0]=[image]
            distance_set.append(distance2)
            groups.append(1)

        #otherwise evaluate it against existing groups
        else:
            min_distance = float("inf")
            closest_group_idx = -1
            associated=0
            
            #A loop that is to check the current representation against all other representations in the
            #distance set
            for j, distance1 in enumerate(distance_set):

                #get similarity metric (1 or 0) and distance from evaluate similarity function
                similarity, distance = evaluate_similarity(distance1, distance2, similarity_threshold)

                #if is similar and distance is lesser than the older minimum distance
                if similarity==1 and distance < min_distance:
                    #set as the representation associated to a group
                    associated=1
                    #update the minimum distance with new distance
                    min_distance = distance
                    #store the index of the closest group
                    closest_group_idx = j

            #increment the number of images represented by that closest group
            groups[closest_group_idx] += 1

            #saves images in assigned groups for further lookup
            images_with_assigned_groups[closest_group_idx].append(image)
            
            #if the image not associated to anything, then make a new group 
            if associated==0:
                distance_set.append(distance2)
                groups.append(1)
                images_with_assigned_groups.append([image])
    
    return groups,images_that_represent_groups,distance_set,images_with_assigned_groups

# def CS_evaluator_original_images(dataset,encoder_path,fixed_noise,sample_size,image_shape,noise_dim,similarity_threshold):
#     vae = VAE(image_shape,noise_dim)
#     encoder=vae.encoder
#     encoder.load_weights(encoder_path)
    
#     #Create empty arrays to fill on later
#     groups = []
#     images_that_represent_groups = []
#     distance_set = []
#     images_with_assigned_groups= [[]]

#     all_images=[]
#     for batch in dataset:
#         for img in batch:
#             all_images.append(img)

#     #Iterate over the required sample size, an images is generated every iteration
#     for i in tqdm(range(len(all_images))):
#         #noise = tf.random.normal([1, noise_dim])
#         image = tf.expand_dims(all_images[i],axis=0)
        
#         #image is converted into an encoded representation,call this as distance2
#         distance2 = encoder(image,training=False)

#         #if it is the first image, add it as a new group 
#         if i == 0:
#             images_that_represent_groups.append(image)
#             images_with_assigned_groups[0]=[image]
#             distance_set.append(distance2)
#             groups.append(1)

#         #otherwise evaluate it against existing groups
#         else:
#             min_distance = float("inf")
#             closest_group_idx = -1
#             associated=0
            
#             #A loop that is to check the current representation against all other representations in the
#             #distance set
#             for j, distance1 in enumerate(distance_set):

#                 #get similarity metric (1 or 0) and distance from evaluate similarity function
#                 similarity, distance = evaluate_similarity(distance1, distance2, similarity_threshold)

#                 #if is similar and distance is lesser than the older minimum distance
#                 if similarity==1 and distance < min_distance:
#                     #set as the representation associated to a group
#                     associated=1
#                     #update the minimum distance with new distance
#                     min_distance = distance
#                     #store the index of the closest group
#                     closest_group_idx = j

#             #increment the number of images represented by that closest group
#             groups[closest_group_idx] += 1

#             #saves images in assigned groups for further lookup
#             images_with_assigned_groups[closest_group_idx].append(image)
            
#             #if the image not associated to anything, then make a new group 
#             if associated==0:
#                 distance_set.append(distance2)
#                 groups.append(1)
#                 images_with_assigned_groups.append([image])
    
#     return groups,images_that_represent_groups,distance_set,images_with_assigned_groups

# def high_frequency_energy(dataset, generator, noise_dim, sample_size=100, threshold=0.5):
#     hf_energy_original = []
#     hf_energy_generated = []

#     count=0
#     for i, batch in enumerate(dataset):
#         for j, image in enumerate(batch):
#             if count >= sample_size:
#                 break
#             original_image = image.numpy()
#             original_image = (original_image + 1) / 2
#             original_image_gray = np.mean(original_image, axis=-1)
#             original_fft = fftshift(fft2(original_image_gray))
#             # Define a high-pass filter with the given threshold
#             rows, cols = original_fft.shape
#             crow, ccol = int(rows / 2), int(cols / 2)
#             mask = np.zeros((rows, cols), dtype=np.uint8)
#             mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
#             high_pass_filter = 1 - mask
#             # Apply the high-pass filter
#             original_fft_filtered = original_fft * high_pass_filter
#             # Calculate the high-frequency energy
#             hf_energy_original.append(np.sum(np.abs(original_fft_filtered) ** 2))
#             count+=1
        
#         for i in range(sample_size):
#             noise = tf.random.normal([1, noise_dim])
#             generated_image = generator(noise, training=False)
#             generated_image = (generated_image + 1) / 2
#             generated_image_gray = np.mean(generated_image, axis=-1)
#             generated_fft = fftshift(fft2(generated_image_gray))
            
#             # Define a high-pass filter with the given threshold
#             rows, cols = generated_fft.shape
#             crow, ccol = int(rows / 2), int(cols / 2)
#             mask = np.zeros((rows, cols), dtype=np.uint8)
#             mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
#             high_pass_filter = 1 - mask
            
#             generated_fft_filtered = generated_fft * high_pass_filter
#             hf_energy_generated.append(np.sum(np.abs(generated_fft_filtered) ** 2))


#     hf_energy_original_mean = np.mean(hf_energy_original)
#     hf_energy_generated_mean = np.mean(hf_energy_generated)

#     return hf_energy_original_mean, hf_energy_generated_mean


def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def inception_trained_model_loader(inception_model_path):
    #inception_model_path = "C:/Users/Aayush/Dropbox/symbiosis/Two_Month_Research/Code/crater_detection_v2/train_data/images/train/train_test_set_for_inceptionV3/trained_model/crater_inceptionV3_20_epochs.h5"
    model = load_model(inception_model_path)
    # InceptionV3 model
    for layer in model.layers: 
        layer.trainable = False # Making sure to not retrain the model

    # Create a new model that outputs the activations from the last layer now
    inception_based_feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    return inception_based_feature_extractor

def inception_based_fid_calculation(real_images,generated_images,inception_model_path,sample_size):
    #take 1st n=sample_size images from real and generated data
    real_images = real_images[:sample_size]
    generated_images = generated_images[:sample_size]

    #convert to tensors
    real_images = tf.stack(real_images)
    generated_images = tf.stack(generated_images)

    # load the discriminator model
    model = inception_trained_model_loader(inception_model_path)

    # Rescale the generated images from (-1,1) to (0,1)
    generated_images = (generated_images + 1.) / 2.

    # calculate FID
    fid = calculate_fid(model, real_images, generated_images)
    return fid


# def inception_score(generator, inception_model_path, noise_set, sample_size):
#     inception_model = load_model(inception_model_path)
#     # InceptionV3 model

#     for layer in inception_model.layers: 
#         layer.trainable = False # Making sure to not retrain the model

#     # Reshape the noise and generate images
#     combined_noise = tf.concat(noise_set[0:sample_size], axis=0)
#     combined_noise = tf.reshape(combined_noise, (sample_size, noise_set.shape[2]))
#     generated_images = generator.predict(combined_noise)

#     # Rescale the generated images from (-1,1) to (0,1)
#     generated_images = (generated_images + 1.) / 2.

#     # Predict the class probabilities for the generated images
#     y_pred = inception_model.predict(generated_images)

#     # Compute the inception score
#     p_yx = np.asarray(y_pred)
#     p_y = np.mean(p_yx, axis=0)
#     entropy_diff = entropy(p_yx.transpose(), qk=p_y, base=np.e, axis=1)
#     i_score = np.exp(np.mean(entropy_diff))

#     return i_score



def inception_score_confidence_binary_classification(generated_images, inception_model_path,sample_size, n_split=10, eps=1E-16):

    #take 1st n=sample_size images from real and generated data
    generated_images = generated_images[:sample_size]

    #convert to tensors
    generated_images = tf.stack(generated_images)

    scores = []

    inception_model = load_model(inception_model_path)
    # InceptionV3 model

    for layer in inception_model.layers: 
        layer.trainable = False # Making sure to not retrain the model

    # Reshape the noise and generate images
    imgs= generated_images

    # Rescale the images from (-1,1) to (0,1)
    imgs = (imgs + 1.) / 2.

    n_part = imgs.shape[0] // n_split

    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = imgs[ix_start:ix_end]

        # Predict class probabilities for a subset of images
        yhat = inception_model.predict(subset)

        # Calculate the confidence as the mean predicted probability
        #the np.max i used here was flattening the array
        #this is a bad application from my side i should have used
        # .flatten() or .ravel() instead. But for the sake of not
        #unintentionally break anything else, i'm not changing it
        #However, this is a self-note to keep this in mind for later
        avg_confidence = np.mean(np.max(yhat, axis=1))
        scores.append(avg_confidence)

    # Average over all batches
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

def inception_score_confidence_categorical_classification(generated_images, inception_model_path,sample_size, n_split=10, eps=1E-16):

    #take 1st n=sample_size images from real and generated data
    generated_images = generated_images[:sample_size]

    #convert to tensors
    generated_images = tf.stack(generated_images)

    scores = []

    inception_model = load_model(inception_model_path)
    # InceptionV3 model

    for layer in inception_model.layers: 
        layer.trainable = False # Making sure to not retrain the model

    # Reshape the noise and generate images
    imgs= generated_images

    # Rescale the images from (-1,1) to (0,1)
    imgs = (imgs + 1.) / 2.

    n_part = imgs.shape[0] // n_split

    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = imgs[ix_start:ix_end]

        # Predict class probabilities for a subset of images
        yhat = np.array(inception_model.predict(subset))[:,2]
        print(yhat)
        # Calculate the confidence as the mean predicted probability
        avg_confidence = np.mean(yhat)
        scores.append(avg_confidence)

    # Average over all batches
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std