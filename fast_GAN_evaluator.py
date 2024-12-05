# from gan_evaluators import birthday_paradox_test
# from gan_evaluators import birthday_paradox_test_ssim
# from gan_evaluators import gen_images_sharpness
# from gan_evaluators import original_images_sharpness
# from gan_evaluators import high_frequency_energy
from gan_evaluators import sharpness_difference
from gan_evaluators import calculate_average_ssim
from gan_evaluators import ndb_score
from gan_evaluators import clustering_based_similarity_evaluator
from gan_evaluators import inception_based_fid_calculation
from gan_evaluators import inception_score_confidence_binary_classification
from gan_evaluators import inception_score_confidence_categorical_classification
from gan_toolset import load_saved_weights
from IPython.display import clear_output
import tensorflow as tf
import numpy as np
from scipy.stats import entropy

def fast_gan_evaluator(epoch_start,
                    epoch_end,
                    real_images,
                    noise,
                    generator,
                    discriminator,
                    noise_dim,
                    SSIM_step,
                    SD_step,
                    NDB_step,
                    CS_step,
                    FID_step,
                    IS_step,
                    ndb_ss,
                    SD_ss,
                    SSIM_ss,
                    inception_model_path,
                    CS_ss,
                    CS_thres,
                    FID_ss,
                    IS_ss,
                    checkpoint_path,
                    inception_score_type):
    # num_images = 500  # The total number of images you want to generate

    # # Shuffle real image dataset and take the first num_images
    # real_dataset_shuffled = dataset.shuffle(buffer_size=1000)
    # real_images = []
    # for img_batch in real_dataset_shuffled:
    #     for img in img_batch:
    #         if len(real_images) >= num_images:
    #             break
    #         real_images.append(img.numpy())
    #     if len(real_images) >= num_images:
    #         break

    ssim_scores=[]
    sharpness_diff=[]
    ndb_score_list=[]
    no_unique_groups_list=[]
    entropy_list=[]
    fid_score_list=[]
    IS_list=[]


    for epoch in range(epoch_start,epoch_end):
        flag_SSIM = 1 if (SSIM_step is not None and epoch % SSIM_step == 0) else 0
        flag_SD = 1 if (SD_step is not None and epoch % SD_step == 0) else 0
        flag_NDB = 1 if (NDB_step is not None and epoch % NDB_step == 0) else 0
        flag_CS = 1 if (CS_step is not None and epoch % CS_step == 0) else 0
        flag_FID = 1 if (FID_step is not None and epoch % FID_step == 0) else 0
        flag_IS = 1 if (IS_step is not None and epoch % IS_step == 0) else 0

        if epoch == 1000:
            epoch = 999

        #Load saved weights
        if flag_SSIM == 1 or flag_SD == 1 or flag_NDB == 1 or flag_CS == 1\
            or flag_FID == 1 or flag_IS == 1:

            load_saved_weights(generator,discriminator,epoch,checkpoint_path)

            # Generate num_images generated images
            # noise = tf.random.normal([num_images, noise_dim])

            #uncomment this if you want to test generated_images from the generator
            generated_images = generator(noise, training=False).numpy()

            #uncomment this if you want to get ground truth scores
            # generated_images = real_images.copy()

        if flag_SSIM==1:
            print(f'Running structural similarity test(fidelity) for Epoch {epoch}')
            #Calculate ssim_score
            ssim_score = calculate_average_ssim(real_images, generated_images,SSIM_ss)
            ssim_scores.append(ssim_score)
        
        if flag_SD==1:
            print(f'Running sharpness test(fidelity) for Epoch {epoch}')
            #Calculate generated data sharpness
            SD = sharpness_difference(real_images,generated_images,SD_ss)
            sharpness_diff.append(SD)

        if flag_NDB==1:
            print(f'Running NDB test(diversity) for Epoch {epoch}')
            score_ndb = ndb_score(real_images,generated_images,ndb_ss,noise_dim)
            ndb_score_list.append(score_ndb)

        if flag_CS==1:
            print(f'Running CS test(diversity) for Epoch {epoch}')
            groups,images1,distances,images2 = \
            clustering_based_similarity_evaluator(generated_images,inception_model_path,CS_ss,CS_thres)
            no_unique_groups_list.append(len(groups))
            # Convert the group sizes to a probability distribution
            group_distribution = groups / np.sum(groups)
            # Calculate the entropy of the distribution
            group_entropy = entropy(group_distribution)
            entropy_list.append(group_entropy)
        if flag_FID==1:
            print(f'Running FID test(diversity and fidelity) for Epoch {epoch}')
            fid_score=inception_based_fid_calculation(real_images,generated_images,inception_model_path,FID_ss)
            fid_score_list.append(fid_score)

        if flag_IS==1:
            print(f'Running IS confidence test(fidelity) for Epoch {epoch}')
            #If using binary classification for IS
            if inception_score_type == "binary":
                i_score=inception_score_confidence_binary_classification(generated_images,inception_model_path,IS_ss)[0]
            #If using categorical classification for IS
            elif inception_score_type == "categorical":
                i_score=inception_score_confidence_categorical_classification(generated_images,inception_model_path,IS_ss)[0]
            IS_list.append(i_score)

        clear_output(wait=True)

    return ssim_scores,sharpness_diff,ndb_score_list,no_unique_groups_list,entropy_list,fid_score_list,IS_list