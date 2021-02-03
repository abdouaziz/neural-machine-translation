import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import  Adam , RMSprop
from tensorflow.keras import layers , activations , models , preprocessing , utils
import pandas as pd
from my_utils import load_data , split_data 


data = load_data("fra.txt")
data[:100] 
eng_corpus , french_corpus =  split_data(data)


def encoder ():


    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(eng_corpus) 
    eng_word_index = tokenizer.word_index
    num_eng_tokens = len( eng_word_index )+1
    eng_sequences= tokenizer.texts_to_sequences(eng_corpus) 
    max_input_length_sequences =  max([len(x) for x in  eng_sequences]) 


    padded_eng_sequences = pad_sequences(eng_sequences , maxlen=max_input_length_sequences , padding='post' )
    encoder_input_data = np.array(padded_eng_sequences )  

    return eng_word_index , max_input_length_sequences ,encoder_input_data  , num_eng_tokens 

eng_word_index , max_input_length_sequences ,encoder_input_data  , num_eng_tokens  = encoder()

def decoder():


    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts( french_corpus) 

    french_word_index = tokenizer.word_index
    num_french_tokens = len( french_word_index )+1

    french_sequences = tokenizer.texts_to_sequences( french_corpus) 
    max_output_length_sequences =  max([len(x) for x in  french_sequences]) 



    padded_french_sequences = preprocessing.sequence.pad_sequences( french_sequences , maxlen=max_output_length_sequences, padding='post' )
    decoder_input_data = np.array( padded_french_sequences) 


    return french_word_index , max_output_length_sequences , decoder_input_data  , num_french_tokens , french_sequences


french_word_index , max_output_length_sequences , decoder_input_data  , num_french_tokens , french_sequences = decoder()



decoder_target_data = []
for token_seq in french_sequences:
    decoder_target_data.append(token_seq[1 : ]) 
    
padded_french_lines = preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length_sequences, padding='post' )
onehot_french_lines = utils.to_categorical( padded_french_lines , num_french_tokens )
decoder_target_data = np.array( onehot_french_lines )




encoder_inputs = tf.keras.layers.Input(shape=(max_input_length_sequences,))
encoder_embedding = tf.keras.layers.Embedding( num_eng_tokens, 256 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 256 , return_state=True , recurrent_dropout=0.2 , dropout=0.2 )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=(max_output_length_sequences,))
decoder_embedding = tf.keras.layers.Embedding( num_french_tokens, 256 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 256 , return_state=True , return_sequences=True , recurrent_dropout=0.2 , dropout=0.2)
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( num_french_tokens , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(loss='categorical_crossentropy'  , optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

model.summary()






def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 256,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 256 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model 


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = []
    for word in words:
        tokens_list.append( eng_word_index[ word ] ) 
        
    return pad_sequences( [tokens_list] , maxlen=max_input_length_sequences , padding='post')
