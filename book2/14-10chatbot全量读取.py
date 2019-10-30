from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
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

buckets = [(10, 10),(15, 15)]
data_set = [[] for _ in buckets]
number_word = None

def tokenize(inp_lang,targ_lang):
  lang_tokenizer_inp = tf.keras.preprocessing.text.Tokenizer(filters='',num_words=number_word,oov_token="unk")
  lang_tokenizer_targ = tf.keras.preprocessing.text.Tokenizer(filters='',num_words=number_word,oov_token="unk")
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
    inp_lang,targ_lang = create_dataset(path_train_inp,path_train_targ, num_examples)

    data_set, inp_lang_tokenizer,targ_lang_tokenizer = tokenize(inp_lang,targ_lang)

    return data_set, inp_lang_tokenizer, targ_lang_tokenizer

# Try experimenting with the size of that dataset
num_examples = 20000
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

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    for batch_idx in np.array_split(rnd_idx,range(batch_size,len(X),batch_size)):
        if len(batch_idx)!=batch_size:
            # yield X[:batch_size],y[:batch_size]     # 最后一批不够size了，取开头部分
            raise StopIteration
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

w = tf.get_variable('proj_w', [units,vocab_tar_size ])
b = tf.get_variable('proj_b', [vocab_tar_size])

def seq2seq(encoder_inputs, decoder_inputs):
    single_cell = tf.contrib.rnn.GRUCell(embedding_dim)
    tem_cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(3)])
    output_projection = (w, b)
    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(       # 函数自带单词嵌入
        encoder_inputs, decoder_inputs, tem_cell,
        num_encoder_symbols=vocab_inp_size,
        num_decoder_symbols=vocab_tar_size,
        embedding_size=embedding_dim,feed_previous=True,output_projection=None)    # 使用output_projection后，前一个的输出会*w+b，下面的bucket函数要用softmax_loss_function

encoder_inputs = []
decoder_inputs = []
weights = []

for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],name="encoder{0}".format(i)))
# encoder_inputs = tf.placeholder(tf.int32,shape=[None,n_steps])
# encoder_inputs = tf.unstack(tf.transpose(encoder_inputs, perm=[1, 0]))
for i in range(buckets[-1][1]+1):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],name="decoder{0}".format(i)))
    weights.append(tf.placeholder(tf.float32, shape=[None],name="weight{0}".format(i)))

targets = decoder_inputs[1:]


def sampled_loss(logits, labels):
    labels = tf.reshape(labels, [-1, 1])
    return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                      biases=b,
                                      inputs=logits,
                                      labels=labels,
                                      num_sampled=512,
                                      num_classes=vocab_tar_size)

outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
    encoder_inputs,
    decoder_inputs,
    targets,
    weights,
    buckets,
    seq2seq,
    softmax_loss_function=None,
)

updates = []
opt = tf.train.AdamOptimizer()
for b in range(len(buckets)):
    updates.append(opt.minimize(losses[b]))

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
checkpoint_dir = './training_checkpoints/chatbot1/ckpt'

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    total_step = 0
    EPOCHS = 10
    saver = tf.train.Saver()
    for epoch in range(EPOCHS):
        start = time.time()
        for bucket_id in range(len(buckets)):
            for (batch, (inp, targ)) in enumerate(shuffle_batch(train_set[bucket_id][0],train_set[bucket_id][1],BATCH_SIZE)):
                input_feed = {}
                inp_tran = np.transpose(inp)
                targ_tran = np.transpose(targ)
                encoder_size, decoder_size = buckets[bucket_id]

                batch_masks = []
                for length_id in range(decoder_size):
                    batch_mask = np.ones(BATCH_SIZE, dtype=np.float32)
                    for batch_id in range(BATCH_SIZE):
                        # we set mask to 0 if the corresponding target is a PAD symbol.
                        # the corresponding decoder is decoder_input shifted by 1 forward.
                        if length_id < decoder_size - 1:
                            target = targ[batch_id][length_id + 1]
                        if length_id == decoder_size - 1 or target == 0:
                            batch_mask[batch_id] = 0.0
                    batch_masks.append(batch_mask)

                for l in range(encoder_size):
                    input_feed[encoder_inputs[l].name] = inp_tran[l]
                for l in range(decoder_size):
                    input_feed[decoder_inputs[l].name] = targ_tran[l]
                    input_feed[weights[l].name] = batch_masks[l]

                last_target = decoder_inputs[decoder_size].name
                input_feed[last_target] = np.zeros([BATCH_SIZE], dtype=np.int32)

                sess.run(updates[bucket_id],input_feed)
                if batch == steps_per_epoch[bucket_id]:
                    break
                if batch % 100 == 0:
                    print('Epoch {} bucket {} Batch {} Loss {:.4f}'.format(epoch + 1,bucket_id, batch, losses[bucket_id].eval(feed_dict=input_feed)))

        if (epoch + 1) % 2 == 0:
            save_path = saver.save(sess, checkpoint_dir)
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

def evaluate(sentence):
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_dir)
        sentence = preprocess_sentence(sentence)        # 预处理
        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]  # 分割单词为整数

        bucket_id=None
        for index, (inp_size, targ_size) in enumerate(buckets):
            if len(inputs)<inp_size:
                bucket_id=index
                break
        encoder_size, decoder_size = buckets[bucket_id]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],     # 按照最大长度填充post令牌
                                                               maxlen=buckets[bucket_id][0],
                                                               padding='post')
        targ = np.array([1 for _ in range(decoder_size)]).reshape(1,-1)
        inputs_tran = np.transpose(inputs)
        targ_tran = np.transpose(targ)
        input_feed = {}
        batch_masks = []
        for length_id in range(decoder_size):
            batch_mask = np.ones(1, dtype=np.float32)
            for batch_id in range(1):
                # we set mask to 0 if the corresponding target is a PAD symbol.
                # the corresponding decoder is decoder_input shifted by 1 forward.
                if length_id < decoder_size - 1:
                    target = targ[batch_id][length_id + 1]
                if length_id == decoder_size - 1 or target == 0:
                    batch_mask[batch_id] = 0.0
            batch_masks.append(batch_mask)
        for l in range(encoder_size):
            input_feed[encoder_inputs[l].name] = inputs_tran[l]
        for l in range(decoder_size):
            input_feed[decoder_inputs[l].name] = targ_tran[l]
            input_feed[weights[l].name] = batch_masks[l]
        last_target = decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([1], dtype=np.int32)
        result = ''
        output_feed=[]
        for t in range(decoder_size):        # 遍历最大长度
            output_feed.append(outputs[bucket_id][t])
        predictions = sess.run(output_feed, input_feed)
        for prediction in predictions:
            predicted_id = np.argmax(prediction[0])
            result += targ_lang.index_word[predicted_id] + ' '      # 找到对应的语言
            if targ_lang.index_word[predicted_id] == '<end>':       # 当前predicted_id对应end返回，不管post令牌了
                break
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

# evaluate(u'hace mucho frio aqui.')
# evaluate(u'esta es mi vida.')
# evaluate(u'¿todavia estan en casa?')
# wrong translation
# evaluate(u'trata de averiguarlo.')
evaluate(u'Ven a conocer a mi amigo.')