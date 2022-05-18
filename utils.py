import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
import json
from keras.utils import np_utils
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
import re

# ============================= SEEDING for REPLICATE RESULTS  ==================================================================
# seed random number generator
from numpy.random import seed
seed(1)

# Seed the behavior of the environment variable
os.environ['PYTHONHASHSEED'] = str(1)
# Seed numpy's instance in case you are using numpy's random number generator, shuffling operations, ...
np.random.seed(1)
### tf random seed
tf.random.set_seed(1234)



## ============================= GLOBAL Variables and PATH ======================================================================
TARGET_SEQ_START = '📗'
TARGET_SEQ_END = '🥕'

seq_length = 100

### Set upper bound of num. samples. One sample: a pair of (input, target). Both input and target are sliced from the large concatenated 
### string of source codes. Slice length = seq_length = 100. 
#Basically, we do not train the whole concatenated string of source codes, rather portion of 
### the concatenated string. To handle unknown characters during program generation, we introduce 'UNK' char

num_samples = 2000000

path_to_input = 'C:/Drive E/Fall 2020/RQ1.2/New folder/ISSTA/Files_input/File_2'
path_to_model = 'C:/Drive E/Fall 2020/RQ1.2/New folder/ISSTA/Seq2seq/log_dir/testmodel_v3'
path_to_output = 'C:/Drive E/Fall 2020/RQ1.2/New folder/ISSTA/Files_input//File_out/'
### train-val accuracy plot
plot_train_val = path_to_output + "accuracy_plot.pdf"
### csv file containing output after compilation
fout_csv = path_to_output + 'compiled_output.csv'

## ========================= NEURAL NETWORK HYPERPARAMETERS =======================================================================
## ======================== TIPS and TRICKS to ESTIMATE PARAMs for a GOOD MODEL by KARPATHY: https://github.com/karpathy/char-rnn

### batch size: Batch size = 64 means 64 samples from the training set will be used each time to update model's weight. Batch size usually power of 2 (2^x)
### epoch:      A complete pass on the whole training dataset to update model's weight.
### patience:   Early stopping of training if the validation loss does not decrease (or val accuracy does not increase) for 5 epoch.
### LSTM_latentdim: Also known as rnn_size-- number of neurons per layer. Usually power of 2. Increased latent_dim with various dropout rate and monitoring the validation loss is a good way to estimate model's behavior. 
### dropout_rate: Prevents overfiiting. Randomly ignore some nurons during training. Can be added 1) before feeding the data into LSTM and 2) before final output of LSTM through dense layer

train_batch_size = 64  # Batch size for training.
train_epochs =    25  # Number of epochs to train for.
train_patience =  5
train_valsplit =  0.3
LSTM_latent_dim  = 256
dropout_rate    = 0.2




### remove comment and empty lines

def remove_comment(fcontent):
  ## #=[\s\S]*?=# : for multi line comment. [\s\S] captures newline https://stackoverflow.com/questions/44246215/is-s-s-same-as-dot
  ### #.*\n : for single line comment
  #pattern1 = '#=[\s\S]*?=#|#.*\n'
  pattern1 = '#=[\s\S]*?=#'
  pattern2 = '#.*\n'
  
  ### 1st clean multiline comments. Then, clean single line comments taking input removing multine comments
  fclean = re.sub(pattern1, ' ',fcontent)
  fclean = re.sub(pattern2, '\n',fclean)
  

  return fclean





#### combine all julia files into a single string
def concate_files():
    source_ttrain = ""

    # recursively traverse directories and subdirectories
    for root,d_names, files_in_dir in os.walk(path_to_input):

        for filename in files_in_dir:
            #if file in ["a.jl", "b_monte_carlo_clean.jl", "a_clean.jl"]:
            if ".jl" in filename:
        
                fname_fullpath = os.path.join(root, filename)
                print(fname_fullpath)
                raw_text = open(fname_fullpath, 'r', encoding='utf-8').read().strip()
                raw_text = raw_text.lower()
                ## remove comment
                raw_text = remove_comment(raw_text)
                source_ttrain = source_ttrain  + raw_text

    return source_ttrain


#### The input to the encoder is a sequence of characters, each encoded as one-hot vectors with length of num_encoder_tokens.
### The decoder input is defined as a sequence of character one-hot encoded to binary vectors with a length of num_decoder_tokens.
### vectorize method returns 
# encoder : num of unique char, num_encoder_tokens, and max_encoder_seq_length, and input_texts = sliced texts into chunks of 100 char (seq_length)
## decoder : num of unique char, num_deoder_tokens, and max_decoder_seq_length, and target_texts= sliced texts into chunks of 100 char (seq_length). 
## In our case, max_decoder_seq_length = max_encoder_seq_length + 2
def vectorize(source_ttrain):

    n_chars = len(source_ttrain)
    print("source file char ", n_chars)

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    ## mark beginning of target sequence

    ## take min of num_samples - seq_length and n_char - seq_length
    train_chars = min(n_chars - seq_length, num_samples - seq_length)


    for i in range(0, train_chars, 1):
        input_text = source_ttrain[i:i + seq_length]
        ### target sequence is shifted one position, 
        target_text = source_ttrain[(i+1):(i+seq_length+1)]
        #target_text = source_ttrain[(i+seq_length)]
        
        ### append start and end symbol
        target_text = TARGET_SEQ_START + target_text + TARGET_SEQ_END
        
        input_texts.append(input_text)
        target_texts.append(target_text)
        
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    
    ### added on 10/15/2021. Encoder and decoder should have same vocab
    for char in input_characters:
        if char not in target_characters:
            target_characters.add(char)

    
    for char in target_characters:
        if char not in input_characters:
            input_characters.add(char)

    
    ### added 11/05/2021. Add UNK char to handle with unknown char input
    input_characters.add('unk')
    target_characters.add('unk')
    
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

    return(input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_texts, target_texts)


#### Input: sliced texts ( input_texts, target_texts,), num_encoder_tokens, num_decoder_tokens. This method converts sliced texts into one-hot encodeed
### vectors for encoder and decoder.
def tokenize_encoder_decoder(input_characters, target_characters, input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens):
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    ### use bool instead of float32.
    
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype= np.bool
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.bool #"float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.bool
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1
        ### add this line. Each input sentence is padded to be equal length. Fill rest of the padding with ' '    
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1
        decoder_target_data[i, t:, target_token_index[" "]] = 1

    return(encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index)

    


#### Subprocess call 

import io 

def write_programs(text, filename, mode, path):
    ### split filename and extension (.jl)
    name = filename[:-3]
    extension = filename[-3:]
    filename = path + name + '_' + mode + extension

    ### fname wo path
    fname_wo_path = name + '_' + mode + extension

    f = io.open(filename, 'w', encoding='utf8')#open(filename, 'w')
    command = 'julia ' + filename
    f.write(text)
    f.close()
    
    '''
    p1 = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    stderr = str(p1.stdout.read())
    print(stderr)
    if ('internal compiler error' in stderr):
        return True
    if ('error' in stderr):
        return False
    else:
        return True
    '''
