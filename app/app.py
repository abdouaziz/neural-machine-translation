import numpy as np
import os 
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction 
from seq2seq.preprocessing import download_and_read
from data.download import get_path_name
from seq2seq.tokenizer import get_Tokenizer_eng , get_Tokenizer_fra

 

class Encoder (tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps,embedding_dim, encoder_dim, *args, **kwargs):
        super(Encoder,self).__init__(*args, **kwargs)
        self.encoder_dim =  encoder_dim
        self.embedding  =tf.keras.layers.Embedding(vocab_size , embedding_dim, input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(encoder_dim, return_sequences=False , return_state=True)

    def call(self,x , state):
        x =  self.embedding(x)
        x, state =  self.rnn(x)
        return x , state 


    def init_state(self, batch_size):

        return tf.zeros((batch_size, self.encoder_dim))


class Decoder(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, num_timesteps, decoder_dim, *args, **kwargs):
        super(Decoder , self).__init__(*args, **kwargs)
        self.decoder_dim =  decoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size , embedding_dim, input_length=num_timesteps)
        self.rnn = tf.keras.layers.GRU(decoder_dim , return_sequences=True , return_state=True)


    def call(self , x , state):
        x = self.embedding(x)
        x , state =  self.rnn(x)

        return x , state



def loss_fn (ytrue ,ypred):

    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(ytrue,0))
    mask = tf.cast(mask , tf.int64)
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


def predict(encoder, decoder, batch_size,sents_en, data_en, sents_fr_out,word2idx_fr, idx2word_fr):
    random_id = np.random.choice(len(sents_en))
    print("input    : ",  " ".join(sents_en[random_id]))
    print("label    : ", " ".join(sents_fr_out[random_id]))

    encoder_in = tf.expand_dims(data_en[random_id], axis=0)
    decoder_out = tf.expand_dims(sents_fr_out[random_id], axis=0)

    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state

    decoder_in = tf.expand_dims(tf.constant([word2idx_fr["BOS"]]), axis=0)
    pred_sent_fr = []
    while True:
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
        decoder_pred = tf.argmax(decoder_pred, axis=-1)
        pred_word = idx2word_fr[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(pred_word)
        if pred_word == "EOS":
            break
        decoder_in = decoder_pred
    
    print("predicted: ", " ".join(pred_sent_fr))


def evaluate_bleu_score(encoder, decoder, test_dataset,word2idx_fr, idx2word_fr):

    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for encoder_in, decoder_in, decoder_out in test_dataset:
        encoder_state = encoder.init_state(batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)

        # compute argmax
        decoder_out = decoder_out.numpy()
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()

        for i in range(decoder_out.shape[0]):
            ref_sent = [idx2word_fr[j] for j in decoder_out[i].tolist() if j > 0]
            hyp_sent = [idx2word_fr[j] for j in decoder_pred[i].tolist() if j > 0]
            # remove trailing EOS
            ref_sent = ref_sent[0:-1]
            hyp_sent = hyp_sent[0:-1]
            bleu_score = sentence_bleu([ref_sent], hyp_sent, 
                smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)

    return np.mean(np.array(bleu_scores))



if __name__ =="__main__":

    
    NUM_SENT_PAIRS = 300
    EMBEDDING_DIM = 256
    ENCODER_DIM, DECODER_DIM = 1024, 1024
    BATCH_SIZE = 64
    NUM_EPOCHS = 30




    checkpoint_dir ="./dataset/checkpoit"

      


    sents_en, sents_fr_in, sents_fr_out= download_and_read()

    tokenizer_en ,data_en = get_Tokenizer_eng(sents_en)
 
    tokenizer_fr , data_fr_in , data_fr_out = get_Tokenizer_fra(sents_fr_in, sents_fr_out) 

    vocab_size_en = len(tokenizer_en.word_index)
    vocab_size_fr = len(tokenizer_fr.word_index)
    word2idx_en = tokenizer_en.word_index

    idx2word_en = {v:k for k, v in word2idx_en.items()}
    word2idx_fr = tokenizer_fr.word_index
    idx2word_fr = {v:k for k, v in word2idx_fr.items()}

   
    maxlen_en = data_en.shape[1]
    maxlen_fr = data_fr_out.shape[1]
    
    batch_size = BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
    dataset = dataset.shuffle(10000)


    test_size = NUM_SENT_PAIRS // 4
    test_dataset = dataset.take(test_size).batch(batch_size, drop_remainder=True)
    train_dataset = dataset.skip(test_size).batch(batch_size, drop_remainder=True)


 
    embedding_dim = EMBEDDING_DIM
    encoder_dim, decoder_dim = ENCODER_DIM, DECODER_DIM

    encoder = Encoder(vocab_size_en+1, embedding_dim, maxlen_en, encoder_dim)
    decoder = Decoder(vocab_size_fr+1, embedding_dim, maxlen_fr, decoder_dim)
 
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    num_epochs = NUM_EPOCHS
    eval_scores = []

    for e in range(num_epochs):
        encoder_state = encoder.init_state(batch_size)
      

        for batch, data in enumerate(train_dataset):

            encoder_in, decoder_in, decoder_out = data
  
            loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)
    
        print("Epoch: {}, Loss: {:.4f}".format(e + 1, loss.numpy()))

        if e % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    
        predict(encoder, decoder, batch_size, sents_en, data_en,sents_fr_out, word2idx_fr, idx2word_fr)

        eval_score = evaluate_bleu_score(encoder, decoder, test_dataset, word2idx_fr, idx2word_fr)
        print("Eval Score (BLEU): {:.3e}".format(eval_score))

    
   