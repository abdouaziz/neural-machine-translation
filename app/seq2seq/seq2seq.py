import os 
import tensorflow as tf
import shutil 
import pandas as pd 
import unicodedata
import re
import zipfile


 

def preprocess_sentence(sent):
    sent = "".join([c for c in unicodedata.normalize("NFD", sent) 
        if unicodedata.category(c) != "Mn"])
    sent = re.sub(r"([!.?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)
    sent = re.sub(r"\s+", " ", sent)
    sent = sent.lower()
    return sent


def download_and_read():
    en_sents, fr_sents_in, fr_sents_out = [], [], []
    local_file = os.path.join("fra.txt")
    with open(local_file, "r") as fin:
        for i, line in enumerate(fin):
            en_sent, fr_sent = line.strip().split('\t')[:2]
            en_sent = [w for w in preprocess_sentence(en_sent).split()]
            fr_sent = preprocess_sentence(fr_sent)
            fr_sent_in = [w for w in ("BOS " + fr_sent).split()]
            fr_sent_out = [w for w in (fr_sent + " EOS").split()]
            en_sents.append(en_sent)
            fr_sents_in.append(fr_sent_in)
            fr_sents_out.append(fr_sent_out)
            if i >= 30001 - 1:
                break
    return en_sents, fr_sents_in, fr_sents_out


def clean_up_logs(path_dir):
    checkpoint_dir = os.path.join(path_dir , "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


class Encoder(tf.keras.Model):
    def __init__(self,vocab_size, num_timesteps,embedding_dim, encoder_dim , *args, **kwargs):
        super(Encoder,self).__init__(*args, **kwargs)
        self.encoder_dim = encoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size , embedding_dim , input_length=num_timesteps)
        self.rnn  = tf.keras.layers.GRU(self.encoder_dim , return_sequences=False ,return_state=True)

    def call (self , input , state):
        x = self.embedding(input)
        x , state = self.rnn(x , initial_state=state) 

        return x , state

    def init_state(self,batch_size):
        return tf.zeros((batch_size , self.encoder_dim))


class Decoder (tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_timesteps,decoder_dim, *args, **kwargs):
        super(Decoder , self).__init__(*args, **kwargs)
        self.decoder_dim = decoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size ,embedding_dim ,input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(decoder_dim , return_sequences=True , return_state=True)

    def call (self , input , state):
        x = self.embedding(input)
        x , state = self.rnn(x , initial_state=state)
        return x , state 


def loss_fn (ytrue , ypred):
    scce =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask =  tf.math.logical_not(tf.math.equal(ytrue , 0))
    mask =  tf.cast(mask , dtype=tf.float32)
    loss = scce(ytrue , ypred , sample_weight=mask)

    return loss 


@tf.function
def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
        loss = loss_fn(decoder_out, decoder_pred)
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss



if __name__== "__main__":

  
    NUM_SENT_PAIRS = 30000
    EMBEDDING_DIM = 256
    ENCODER_DIM, DECODER_DIM = 1024, 1024
    BATCH_SIZE = 64
    NUM_EPOCHS = 30

    
    data_dir = "./data"
    checkpoint_dir = clean_up_logs(data_dir)

    # data preparation
    download_url = "http://www.manythings.org/anki/fra-eng.zip"
    sents_en, sents_fr_in, sents_fr_out = download_and_read()

    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
    tokenizer_en.fit_on_texts(sents_en)
    data_en = tokenizer_en.texts_to_sequences(sents_en)
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding="post")

    tokenizer_fr = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
    tokenizer_fr.fit_on_texts(sents_fr_in)
    tokenizer_fr.fit_on_texts(sents_fr_out)
    data_fr_in = tokenizer_fr.texts_to_sequences(sents_fr_in)
    data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding="post")
    data_fr_out = tokenizer_fr.texts_to_sequences(sents_fr_out)
    data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding="post")

    
    vocab_size_en = len(tokenizer_en.word_index)
    vocab_size_fr = len(tokenizer_fr.word_index)
    word2idx_en = tokenizer_en.word_index
    idx2word_en = {v:k for k, v in word2idx_en.items()}
    word2idx_fr = tokenizer_fr.word_index
    idx2word_fr = {v:k for k, v in word2idx_fr.items()}
    print("vocab size (en): {:d}, vocab size (fr): {:d}".format(vocab_size_en, vocab_size_fr))

    maxlen_en = data_en.shape[1]
    maxlen_fr = data_fr_out.shape[1]
    print("seqlen (en): {:d}, (fr): {:d}".format(maxlen_en, maxlen_fr))

    batch_size = BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
    dataset = dataset.shuffle(10000)
    test_size = NUM_SENT_PAIRS // 4
    test_dataset = dataset.take(test_size).batch(batch_size, drop_remainder=True)
    train_dataset = dataset.skip(test_size).batch(batch_size, drop_remainder=True)

    # check encoder/decoder dimensions
    embedding_dim = EMBEDDING_DIM
    encoder_dim, decoder_dim = ENCODER_DIM, DECODER_DIM

    encoder = Encoder(vocab_size_en+1, embedding_dim, maxlen_en, encoder_dim)
    decoder = Decoder(vocab_size_fr+1, embedding_dim, maxlen_fr, decoder_dim) 

    
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

    num_epochs = NUM_EPOCHS
    eval_scores = []

    for e in range(num_epochs):
        encoder_state = encoder.init_state(batch_size)

        for batch, data in enumerate(train_dataset):

            encoder_in, decoder_in, decoder_out = data
        # print(encoder_in.shape, decoder_in.shape, decoder_out.shape)
            loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)
            print("loss ",loss)
    
            print("Epoch: {}, Loss: {:.4f}".format(e + 1, loss.numpy()))

            if e % 10 == 0:

                checkpoint.save(file_prefix=checkpoint_prefix)
    
 
 
    # eval_scores.append(eval_score)
 








 
