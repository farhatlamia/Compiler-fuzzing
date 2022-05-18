from utils import *
import pandas as pd
import random
random.seed(237)


## Load trained model.
def load_trained_model(input_token_index, target_token_index):
    # Restore the model and construct the encoder and decoder. inference models requires reference to elements of the model used for training.
    #model = tf.keras.models.load_model('C:/Drive E/Fall 2020/RQ1.2/New folder/ISSTA/Seq2seq/log_dir/testmodel_v2')
    model = tf.keras.models.load_model(path_to_model)
    ### latent dim 

    # encoder model is defined as taking the input layer from the encoder in the trained model (encoder_inputs) and outputting the hidden and cell state tensors (encoder_states).
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)



    ### The decoder requires the hidden and cell states from the encoder as the initial state of the newly defined encoder model. Because the decoder is a separate standalone model, 
    #these states will be provided as input to the model, and therefore must first be defined as inputs: decoder_state_input_h and decoder_states_inputs

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(LSTM_latent_dim,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(LSTM_latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    ## hidden and cell states can then be specified for use as the initial state of the decoder LSTM layer. On the first call, the hidden and cell states from the encoder will be used to initialize the decoder LSTM layer, provided as input to the model directly.
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)

    # On subsequent recursive calls to the decoder, the last hidden and cell state must be provided to the model
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    return(encoder_model, decoder_model, reverse_input_char_index, reverse_target_char_index)


def char_sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index, num_decoder_tokens, target_token_index, max_decoder_seq_length, diversity = 1):
    
    
    TARGET_SEQ_END = '🥕'
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, target_token_index["\t"]] = 1.0
    target_seq[0, 0, target_token_index[TARGET_SEQ_START]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        #sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token_index = char_sample(output_tokens[0, -1, :], diversity)
        sampled_char = reverse_target_char_index[sampled_token_index]
        #print("sample char in decoder ", sampled_char)

        ## added 11/05/2021: handle UNK
        sampled_char_original = sampled_char

        if sampled_char in ['unk', TARGET_SEQ_START, TARGET_SEQ_END]:
            sampled_char = ''


        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char_original == TARGET_SEQ_END or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    
    
    return decoded_sentence

def synthesis(source_ttrain, max_encoder_seq_length, num_encoder_tokens, input_token_index, encoder_model, decoder_model, reverse_target_char_index, num_decoder_tokens, target_token_index, max_decoder_seq_length, input_characters, gmode='g1', smode='nosample'):
    length = len(source_ttrain)
    prefix_start = random.randrange(length - seq_length)
    ## wade hable len 252
    
    text = ""

    if (gmode is 'g1'):


        #prefix_start = len(source_ttrain) - 252

        prefix = source_ttrain[prefix_start:prefix_start + seq_length]
        #print("prefix ", prefix, "start ", prefix_start, "len ", len(prefix))
        head = source_ttrain[0 : prefix_start]

       

        #### replace one slice of char (100 char) with 200 char
        generated_seq = ""
        count_ = 0
        while(count_< 200):
            
            prefix_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
            
            for t, char in enumerate(prefix):
                #print("t ", t, "char ", char)
                ### added 11/05/2021: handle UNK
                if(char not in input_characters):
                    char = 'unk'
                    
                prefix_seq[0, t, input_token_index[char]] = 1.
        
            prefix_seq[0, t+1:, input_token_index[" "]] = 1.

            if (smode is 'nosample'):
                decoded_sentence = decode_sequence(prefix_seq, encoder_model, decoder_model, reverse_target_char_index, num_decoder_tokens, target_token_index, max_decoder_seq_length, 1)

            
            generated_seq += decoded_sentence
            prefix = prefix[1:] + decoded_sentence
            ### prefix limit to 100 char len
            prefix = prefix[-seq_length:] 
            count_ = count_ + len(decoded_sentence)
        
            

        #print("Decoded sentence:", decoded_sentence[1:-1])
        #print("Decoded sentence:", generated_seq)
        #print("Suffix ", source_ttrain[prefix_start + seq_length: prefix_start + 2*seq_length])
        tail = source_ttrain[prefix_start + seq_length:]

        text = head + generated_seq + tail

    return text



### generate programs using trained model

def generate():
    #### get input files concatenated as a string
    source_files = concate_files()
    ## vectorize concatenated string
    input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_texts, target_texts = vectorize(source_files)

    ### tokenize encoder decoder 
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index = tokenize_encoder_decoder(input_characters, target_characters, input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens)

    ### load trained model and init encoder decoder
    encoder_model, decoder_model, reverse_input_char_index, reverse_target_char_index = load_trained_model(input_token_index, target_token_index)

    
    
    path_outfile = (path_to_output)
    mode = 'nosample'

    
    # recursively traverse directories and subdirectories
    for root,d_names, files_in_dir in os.walk(path_to_input):
    
        #if filename in ["a.jl", "b_monte_carlo_clean.jl", "a_clean.jl"]:
        for filename in files_in_dir:

            if ".jl" in filename:
                ## full pat fname
                fname_fullpath = os.path.join(root, filename)

                #print("=======FILENAME: ", fname_fullpath, " root ", root)
                raw_text = open(fname_fullpath, 'r', encoding='utf-8').read().strip()
                raw_text = raw_text.lower()
                raw_text = remove_comment(raw_text)
                       
                        
                text = synthesis(raw_text, max_encoder_seq_length, num_encoder_tokens, input_token_index, encoder_model, decoder_model, reverse_target_char_index, num_decoder_tokens, target_token_index, max_decoder_seq_length, input_characters)

                write_programs(text, filename, mode, path_outfile)


##### execute generated programns
from subprocess import Popen, PIPE, STDOUT



def verify_correctness():

    path = (path_to_output)
    os.chdir(path)

    ### save compiled output in a .csv file
    outdf = []

    for filename in os.listdir() :

        if ".jl" not in filename:
            continue

        command = 'julia ' + filename
        p1 = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        stderr = str(p1.stdout.read())
        ### lower case error message
        stderr = stderr.lower()
        print("Filename: ", filename, "\n")
        print("Error: ", stderr, "\n")

        if ('compiler' in stderr or 'internal' in stderr):
            #print(filename + " ", "True")
            outdf.append((filename, "internal compiler error"))

        if ('error' in stderr):
            #print(filename + " ", "False")
            outdf.append((filename, "Other errors"))
        else:
            #print(filename + " ", "True")
            outdf.append((filename, "Pass"))

    outdf = pd.DataFrame(outdf, columns = ['Filename', 'Compiled Output'])
    
    outdf.to_csv(fout_csv,  index = False)



if __name__ == "__main__":
   
    ### generate programs
    generate()

    ### compile generated programs and print pass/ error/ compiler error
    verify_correctness()




