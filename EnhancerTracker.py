#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this notebook is to be the final version of EnhancerTracker for publication
# 
# ### This notebook will take 2 input fasta files of 400 base pair length and output their probability of being a enhancer
# ### Optional, output will also have a Class Activation map of the triplet sequences.

# ### @author: Luis Solis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# ### @author: William Melendez, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# ### @author: Sayantan Paul, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# ### @author: Shantanu Hemantrao Fuke, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# ### @author: Dr. Hani Z. Girgis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
# 
# #### Date Created: 12-3-2024

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import load_model

from Nets import CustomConvLayer
from Metrics import specificity

from Bio import SeqIO
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys


# In[ ]:


output_cam_pdf = False


# In[ ]:


similar_sequences_file = sys.argv[1]
all_sequences_file     = sys.argv[2]

if len(sys.argv) == 4:
    if sys.argv[3] == '--cam':
       output_cam_pdf = True


# In[ ]:


'''
Input 1 will include the two confirmed enhancers from the same cell type and must be in fasta format
Input 2 will include all sequences that will be tested to see if they are similar to the first two sequences, this must also be in fasta format

Output will include
'''
human_indexer          = f'indexer.pkl'

triplet_model_file     = f'Models/conv_model.keras'
class_model_file       = f'Models/class_model.keras'
cam_model_file         = f'Models/cam_model.keras'

output_dir             = f'Output'

max_len = 400


# ### Load all models used for EnhancerTracker

# In[ ]:


conv_model = load_model(triplet_model_file, custom_objects={'CustomConvLayer': CustomConvLayer,'specificity': specificity})
class_model = load_model(class_model_file, custom_objects={'CustomConvLayer': CustomConvLayer,'specificity': specificity})
cam_model = load_model(cam_model_file, custom_objects={'CustomConvLayer': CustomConvLayer,'specificity': specificity})


# ### Load indexer used for encoding the sequences to numerical format the model understands 

# In[ ]:


with open(human_indexer, 'rb') as f:
    indexer = pickle.load(f)


# ### Parse input files and grab their names for output and CAM
# ### Encode the input sequences

# In[ ]:


similar_seq_list = list(SeqIO.parse(similar_sequences_file, "fasta"))
all_sequence_list = list(SeqIO.parse(all_sequences_file, "fasta"))


# In[ ]:


similar_name_list = []
all_name_list = []

for seq in similar_seq_list:
    similar_name_list.append(seq.id)

for seq in all_sequence_list:
    all_name_list.append(seq.id)


# In[ ]:


matrix1  = indexer.encode_list(similar_seq_list)
matrix2  = indexer.encode_list(all_sequence_list)


# ### Create a zero tensor with shape of input for the model
# ### Fill in the tensor with data from input files
# 
# #### Tensor has 3 channels, 1st and 2nd channel are for input1 sequence 1 and 2. Channel 3 is for the sequence that will be identified from input2.

# In[ ]:


batch_size   = matrix2.shape[0]
row_size     = 1
column_size  = matrix1.shape[1]
channel_size = 3

tensor  = np.zeros((batch_size, row_size, column_size, channel_size), dtype=np.int8)


# In[ ]:


tensor.shape


# In[ ]:


for i in range(batch_size):
    tensor[i, :, :, 0] = matrix1[0]
    tensor[i, :, :, 1] = matrix1[1]
    tensor[i, :, :, 2] = matrix2[i]


# ### Predict the tensor and write results to output file

# In[ ]:


output_prediction = conv_model.predict(tensor)


# In[ ]:


formatted_output = [f"{value[0]:.2f}" for value in output_prediction]


# In[ ]:


with open(f'{output_dir}/Model_Output.txt', 'w') as file:
    for name, percentage in zip(all_name_list, formatted_output):
        file.write(f"{name} {percentage}\n")


# ### Below is code for making the CAM model
# ### Cam model was based on code from Deep Learning with Python by Francois Chollet

# In[ ]:


def plot_CAM_map(heatmap_interpolated_list, output_dir, name_list, save_pdf):
    """
    This function generates a series of 1D heatmaps (color-maps) based on the provided input data and visualizes them in a single figure.
    The heatmaps are displayed in three rows, each representing a different channel.

    Inputs:
    - heatmap_interpolated_list (list): A list of 1D numpy arrays representing the heatmap data from the sequences. 
      Each array corresponds to a different channel to be visualized.
    - output_dir (str): The directory path where the output figure will be saved.
    - name_list (list): A list of strings representing the names or labels for each dataset. These will be used as titles 
      for the individual subplots.
    - save_pdf (bool): A boolean indicating whether the figure should be saved as a PDF.
    """
    
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 5))  # 3 rows, 1 column
    for i, heatmap_interpolated in enumerate(heatmap_interpolated_list):                  
        image = axs[i].matshow(heatmap_interpolated.reshape(1, -1), cmap='jet', aspect='auto', vmin=0, vmax=1)
        axs[i].set_yticks([])
        axs[i].xaxis.set_ticks_position('bottom') 
        axs[i].set_xlim(-0.5, len(heatmap_interpolated))

        axs[i].set_title(f'{name_list[i].split(":", 1)[1]}', fontsize=14)
        
        if i == 2:
            axs[i].set_xlabel('Nucleotide position', fontsize=14)
            axs[i].tick_params(axis='x', labelsize=14) 
        else:
            axs[i].set_xticks([])

        fig.colorbar(image, ax=axs[i])

        # Remove the box around the heat map
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        
    plt.tight_layout()

    # Only save the figure if save_pdf is True
    if save_pdf:
        plt.savefig(f'{output_dir}.pdf')  # Save the plot as a PDF

    plt.close(fig)


# In[ ]:


def calculate_cam(x_batch_sample):
    """
    This function computes Class Activation Maps (CAM) for a given batch of input samples.
   
    Inputs:
    - x_batch_sample: The input batch of samples for which CAMs will be calculated.
      It is passed through the `cam_model` to get the feature maps.

    Outputs:
    - heatmap_list (list): A list of heatmaps (numpy arrays), one for each feature map in the input batch. 
      Each heatmap corresponds to the importance of different regions of the input image with respect to the model's 
      predictions.
    """
    
    with tf.GradientTape(persistent=True) as tape:
        # Get the CAM model output
        cam_output_np = cam_model.predict(x_batch_sample, verbose=0)

        # Convert each array in the list to a TensorFlow tensor and watch them
        cam_output_tensors = [tf.convert_to_tensor(array, dtype=tf.float32) for array in cam_output_np]
        for tensor in cam_output_tensors:
            tape.watch(tensor)

        # Use the tensors as inputs to the class_model
        preds = class_model(cam_output_tensors)[0]
    
    # Calculate the gradients with respect to each of the cam_output_tensors
    grads_list = [tape.gradient(preds, tensor) for tensor in cam_output_tensors]

    # Dispose the tape manually since it's persistent
    del tape

    cam_output_arrays = [tensor.numpy() for tensor in cam_output_tensors]
    heatmap_list = []
    for j, grads in enumerate(grads_list):
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

        last_conv_layer_output = cam_output_arrays[j]
        
        for i in range(pooled_grads.shape[-1]):
            # last_conv_layer_output[:, :, :, i] *= (-1 * pooled_grads[i])
            last_conv_layer_output[:, :, :, i] *= pooled_grads[i]

        # Apply ReLU to the mean of the gradient-weighted features
        # heatmap = np.mean(last_conv_layer_output, axis=-1)
        
        heatmap = np.max(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)      
        heatmap_list.append(heatmap)

    return heatmap_list


# In[ ]:


def scale_array(an_array):
    '''
    This code was generated by ChatGPT
    '''
    min_val    = np.min(an_array)
    max_val    = np.max(an_array)
    scaled_arr = (an_array - min_val) / ((max_val - min_val) + np.finfo(np.float64).eps)
    
    return scaled_arr


# In[ ]:


def get_sequence(triplet_index):
    """
    Extract Sequence Name using IndexNow processing control: Enhancer
    """       
    seq_name_list = []
    seq_name_list.append(similar_name_list[0])
    seq_name_list.append(similar_name_list[1])
    seq_name_list.append(all_name_list[triplet_index])
 
    return seq_name_list


# ### The CAM only gets generated if output_cam_pdf is True
# ### The code will go through each sequence in input2 and calculate a cam and plot the heatmap
# ### The heatmap will then get outputed to the output file as a pdf

# In[ ]:


if output_cam_pdf:
    for batch_idx in range(tensor.shape[0]):
        batch = tensor[batch_idx]
        batch = np.expand_dims(batch, axis=1)
        
        heatmap_list = calculate_cam(batch)
    
        heatmap_interpolated_list = []
        for i, heatmap in enumerate(heatmap_list):
            heatmap = scale_array(heatmap)
            old_indices = np.linspace(0, heatmap.shape[2] - 1, num=heatmap.shape[2])
            new_indices = np.linspace(0, heatmap.shape[2] - 1, num=max_len)
            heatmap_interpolated = np.interp(new_indices, old_indices, heatmap[0, 0, :])
            heatmap_interpolated_list.append(heatmap_interpolated)
        name_list = get_sequence(batch_idx)
        plot_CAM_map(heatmap_interpolated_list,f'{output_dir}/{name_list[2]}_CAM',name_list, True)

