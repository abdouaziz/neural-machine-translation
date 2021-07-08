from seq2seq.preprocessing import download_and_read
import tensorflow as tf 



NUM_SENT_PAIRS = 30000
EMBEDDING_DIM = 256
ENCODER_DIM, DECODER_DIM = 1024, 1024
BATCH_SIZE = 64
NUM_EPOCHS = 30

 

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
