from utils import *
import matplotlib.pyplot as plt

## an LSTM : |_ |_ |_|_| ....
#            |  |  | | |
##
##           T1 T2 T3 T4 ...
##### build_model takes 3 parameters as input:
### input to encoder = num_encoder_tokens. The cardinality of encoder LSTM for each timestep is one-hot-endoded vector of size num_encoder_tokens 
### input to decoder = num_decoder_tokens. The cardinality of decoder LSTM for each timestep is one-hot-endoded vector of size num_decoder_tokens 
#### Latent dim: Hidden state of LSTM: It has to be the same for endoder and decoder LSTm. The one-hot-endoded input vector is mapped into a vector of size latent_dim

### decoder_dense outputs a sequence char by char. Each char output is based on taking max prob. of num_decoder_tokens.
def build_model(num_encoder_tokens, num_decoder_tokens):
  

    ## Encoder LSTM is defined return state = True. Returns sequences and state.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    ### add Dropout to LSTM inputs. added 11/7/2021. https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
    
    encoder = keras.layers.LSTM(LSTM_latent_dim, dropout= dropout_rate, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard sequences = `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(LSTM_latent_dim, dropout= dropout_rate, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)




    ### decoder_dense: outputs char. 
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    ### add dropouts 
    dropout_output = keras.layers.Dropout(dropout_rate)
    decoder_outputs = dropout_output(decoder_outputs)

    # Define the model 
    #  model is defined with inputs for the encoder and the decoder and the output target sequence.
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

### save trained model in a directory. 'C:/Drive E/Fall 2020/RQ1.2/New folder/ISSTA/Seq2seq/log_dir/testmodel_v2'. 
### this trained model will be used to generate artificial programs.
## 1/5/2022: Fixed error: passes model as function param
def train_save_model(encoder_input_data, decoder_input_data, decoder_target_data, seq2seqmodel):
    
    

    seq2seqmodel.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    ### early stopping of training if validation loss is not decresesd for 5 epoch (patience)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= train_patience, restore_best_weights=True)

    res = seq2seqmodel.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size= train_batch_size,
        epochs= train_epochs,
        validation_split= train_valsplit,
        )

   
   
    # plot training-validattion  accuracy
    plt.plot(res.history['accuracy'])
    plt.plot(res.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    ## save plot as pdf
    plot_fname = plot_train_val
    plt.savefig(plot_fname)  
    #plt.show()

    # Save model

    #model.save('C:/Drive E/Fall 2020/RQ1.2/New folder/ISSTA/Seq2seq/log_dir/testmodel_v2')
    seq2seqmodel.save(path_to_model)


if __name__ == "__main__":
    #### get input files concatenated as a string
    source_files = concate_files()
    ## vectorize concatenated string
    input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_texts, target_texts = vectorize(source_files)

    ### tokenize encoder decoder 
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index = tokenize_encoder_decoder(input_characters, target_characters, input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens)

    ## build model
    
    model = build_model(num_encoder_tokens, num_decoder_tokens)

    #### train and save model
    ## 1/5/2022: pass model as function param
    train_save_model(encoder_input_data, decoder_input_data, decoder_target_data, model)




