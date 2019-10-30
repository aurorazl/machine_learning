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


def create_dataset(path,num_examples=None):
    vocab = {}
    with open(path,'r') as f:
        for i in f:
            for token in preprocess_sentence(i).split():
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    sorted_vocab.insert(0,"UNK")
    sorted_vocab.insert(0,"PAD")
    return {sorted_vocab[i]:i for i in range(len(sorted_vocab))}

number_word = 12000
def sentence2id(vocab, line):
    vect=[]
    for token in preprocess_sentence(line).split():
        i = vocab.get(token,vocab['UNK'])
        if i>=number_word:
            vect.append(vocab['UNK'])
        else:
            vect.append(i)
    return vect

def pad_input(input_, size):
    return input_ + [0] * (size - len(input_))

buckets = [(15, 15)]

inp_vocab_dict = create_dataset(path_train_inp)
targ_vocab_dict = create_dataset(path_train_targ)

BATCH_SIZE = 64
embedding_dim = 128
units = 1024
vocab_inp_size = number_word
vocab_tar_size = number_word
# print(vocab_inp_size,vocab_tar_size)

def shuffle_batch(bucket_id, batch_size):
    X_batch = []
    y_batch = []
    cnt = 0
    with open(path_train_inp,'r') as f_inp:
        with open(path_train_targ, 'r') as f_targ:
            for inp,targ in zip(f_inp,f_targ):   # while inp and targ f.readline()有可能遇到一行为空的情况
                inp_tensor = sentence2id(inp_vocab_dict,inp)
                targ_tensor = sentence2id(targ_vocab_dict,targ)
                if len(inp_tensor)<buckets[bucket_id][0] and len(targ_tensor)<buckets[bucket_id][1]:
                    X_batch.append(pad_input(inp_tensor,batch_size))
                    y_batch.append(pad_input(targ_tensor,batch_size))
                    cnt += 1
                if cnt == batch_size:
                    yield X_batch,y_batch
                    cnt = 0
                    X_batch = []
                    y_batch = []

w = tf.get_variable('proj_w', [units,vocab_tar_size])
b = tf.get_variable('proj_b', [vocab_tar_size])

def seq2seq(encoder_inputs, decoder_inputs):
    single_cell = tf.contrib.rnn.GRUCell(embedding_dim)
    # single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=0.8)
    # tem_cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(3)])
    output_projection = (w, b)
    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(       # 函数自带单词嵌入
        encoder_inputs, decoder_inputs, single_cell,
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

def train():
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        EPOCHS = 10
        saver = tf.train.Saver()
        for epoch in range(EPOCHS):
            start = time.time()
            for bucket_id in range(len(buckets)):
                for (batch, (inp, targ)) in enumerate(shuffle_batch(bucket_id,BATCH_SIZE)):
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

                    if batch % 100 == 0:
                        loss = losses[bucket_id].eval(feed_dict=input_feed)
                        print('Epoch {} bucket {} Batch {} Loss {:.4f}'.format(epoch + 1,bucket_id, batch,loss))
                        if loss<1e-4:
                            saver.save(sess, checkpoint_dir)
                            return

            if (epoch + 1) % 2 == 0:
                saver.save(sess, checkpoint_dir)
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
train()

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

# evaluate(u'What color ?')
# evaluate(u'esta es mi vida.')
# evaluate(u'¿todavia estan en casa?')
# wrong translation
# evaluate(u'trata de averiguarlo.')
# evaluate(u'Ven a conocer a mi amigo.')