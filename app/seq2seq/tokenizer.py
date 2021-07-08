import tensorflow as tf 



 


def get_Tokenizer_eng(sents_en):

    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
    tokenizer_en.fit_on_texts(sents_en)
    data_en = tokenizer_en.texts_to_sequences(sents_en)
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding="post")

    return tokenizer_en ,data_en



def get_Tokenizer_fra(sents_fr_in , sents_fr_out):


    tokenizer_fr = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
    tokenizer_fr.fit_on_texts(sents_fr_in)
    tokenizer_fr.fit_on_texts(sents_fr_out)
    
    data_fr_in = tokenizer_fr.texts_to_sequences(sents_fr_in)
    data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding="post")
    data_fr_out = tokenizer_fr.texts_to_sequences(sents_fr_out)
    data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding="post")

    return tokenizer_fr , data_fr_in , data_fr_out




    




