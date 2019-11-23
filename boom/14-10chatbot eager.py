from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

path_train_inp =r"D:\Akulaku\stanford-tensorflow-tutorials\assignments\chatbot\processed\train.enc"
path_train_targ =r"D:\Akulaku\stanford-tensorflow-tutorials\assignments\chatbot\processed\train.dec"

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# en_sentence = u"May I borrow this book?"
# sp_sentence = u"¿Puedo tomar prestado este libro?"
# print(preprocess_sentence(en_sentence))
# print(preprocess_sentence(sp_sentence).encode('utf-8'))


def create_dataset(path_train_inp,path_train_targ, num_examples):
    inp = open(path_train_inp,'r').read().strip().split('\n')
    targ = open(path_train_targ, 'r').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(inp[i]),preprocess_sentence(targ[i])] for i in range(len(inp)) if i<num_examples]
    return zip(*word_pairs)

# en, sp = create_dataset(path_to_file, None)
# print(en[-1])
# print(sp[-1])

def max_length(tensor):
    return max(len(t) for t in tensor)

buckets = [(10, 10), (15, 15), (20, 20)]
data_set = [[] for _ in buckets]

def tokenize(inp_lang,targ_lang):
  lang_tokenizer_inp = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer_targ = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer_inp.fit_on_texts(inp_lang)
  lang_tokenizer_targ.fit_on_texts(targ_lang)
  inp_tensor = lang_tokenizer_inp.texts_to_sequences(inp_lang)
  targ_tensor = lang_tokenizer_targ.texts_to_sequences(targ_lang)
  for i in range(len(inp_tensor)):
    for bucket_id, (inp_size, targ_size) in enumerate(buckets):
        if len(inp_tensor[i]) < inp_size and len(targ_tensor[i]) < targ_size:
            data_set[bucket_id].append([inp_tensor[i],targ_tensor[i]])
            break

  for bucket_id in range(len(buckets)):
      data_set[bucket_id] = list(zip(*data_set[bucket_id]))
      data_set[bucket_id][0] = tf.keras.preprocessing.sequence.pad_sequences(data_set[bucket_id][0],padding='post',maxlen=buckets[bucket_id][0])
      data_set[bucket_id][1] = tf.keras.preprocessing.sequence.pad_sequences(data_set[bucket_id][1],padding='post',maxlen=buckets[bucket_id][1])

  return data_set, lang_tokenizer_inp,lang_tokenizer_targ

def load_dataset(path_train_inp,path_train_targ, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path_train_inp,path_train_targ, num_examples)

    data_set, inp_lang_tokenizer,targ_lang_tokenizer = tokenize(inp_lang,targ_lang)

    return data_set, inp_lang_tokenizer, targ_lang_tokenizer

# Try experimenting with the size of that dataset
num_examples = 30000
train_set, inp_lang, targ_lang = load_dataset(path_train_inp,path_train_targ, num_examples)

BUFFER_SIZE = [len(input_tensor_train[0]) for input_tensor_train in train_set]
BATCH_SIZE = 64
steps_per_epoch = [len(input_tensor_train[0])//BATCH_SIZE for input_tensor_train in train_set]
print(BUFFER_SIZE,steps_per_epoch)
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
print(vocab_inp_size,vocab_tar_size)
# exit(0)

dataset = [tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE[index])
           for index,(input_tensor_train, target_tensor_train) in enumerate(train_set)]
dataset = [i.batch(BATCH_SIZE, drop_remainder=True) for i in dataset]        # 最后不够64个时丢弃


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints/chatbot1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for bucket_id in range(len(buckets)):
        for (batch, (inp, targ)) in enumerate(dataset[bucket_id].take(steps_per_epoch[bucket_id])):        # take会无限循环
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch==steps_per_epoch[bucket_id]:
                break
            if batch % 100 == 0:
                print('Epoch {} bucket {} Batch {} Loss {:.4f}'.format(epoch + 1,bucket_id,batch,batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch[0]+1))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

max_length_targ = [len(i[1][0]) for i in train_set]
max_length_inp = [len(i[0][0]) for i in train_set]

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)        # 预处理
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]  # 分割单词为整数

    bucket_id=None
    for index, (inp_size, targ_size) in enumerate(buckets):
        if len(inputs)<inp_size:
            bucket_id=index
            break
    attention_plot = np.zeros((max_length_targ[bucket_id], max_length_inp[bucket_id]))

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],     # 按照最大长度填充post令牌
                                                           maxlen=max_length_inp[bucket_id],
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)   #

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)   # 编码整数为向量

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)        #  开始令牌

    for t in range(max_length_targ[bucket_id]):        # 遍历最大长度
        predictions, dec_hidden, attention_weights = decoder(dec_input,     # dec_input是整数，里面通过embed转向量
                                                             dec_hidden,    #
                                                             enc_out)       # enc_out为向量，与dec_input转化后的向量叠加，输入到gru单元中，用于预测下一个结果

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()        # predictions为整数列表，大小为tokenize识别到的所有单词数量targ_lang.word_index，取最大的整数，即为最相似的单词对应的整数

        result += targ_lang.index_word[predicted_id] + ' '      # 找到对应的语言

        if targ_lang.index_word[predicted_id] == '<end>':       # 当前predicted_id对应end返回，不管post令牌了
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)           # 根据新的predicted_id来输入

    return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

