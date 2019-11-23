import tensorflow as tf  # 0.12
import os
import numpy as np
import math
import random
import copy

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vec = './data/word2vec/train_question_encode.vec'
train_decode_vec = './data/word2vec/train_answer_decode.vec'
test_encode_vec = './data/word2vec/test_question_encode.vec'
test_decode_vec = './data/word2vec/test_answer_decode.vec'

# 词汇表大小5000
vocabulary_encode_size = 470
vocabulary_decode_size = 470

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(question_path, answer_path, max_size=None):
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(question_path, mode="r") as question_file:
        with tf.gfile.GFile(answer_path, mode="r") as answer_file:
            question, answer = question_file.readline(), answer_file.readline()
            counter = 0
            while question and answer and (not max_size or counter < max_size):
                counter += 1
                question_ids = [int(x) for x in question.split()]
                answer_ids = [int(x) for x in answer.split()]
                answer_ids.append(EOS_ID)
                for bucket_id, (question_size, answer_size) in enumerate(buckets):
                    if len(question_ids) < question_size and len(answer_ids) < answer_size:
                        data_set[bucket_id].append([question_ids, answer_ids])
                        break
                question, answer = question_file.readline(), answer_file.readline()
                print("question: {}, answer: {}".format(question, answer))
    return data_set


class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forward_only=False):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None

        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, labels, logits, num_samples, self.target_vocab_size)

            softmax_loss_function = sampled_loss
        single_cell = tf.contrib.rnn.GRUCell(size)
        if use_lstm:
            single_cell = tf.contrib.rnn.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            tem_cell = copy.deepcopy(cell)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, tem_cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))

        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)

            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, answer_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        参数:
          session: tensorflow 会话.
          encoder_inputs: 问题向量列表
          decoder_inputs: 回答向量列表
          answer_weights: 答案权重列表
          bucket_id: 桶编号which bucket of the model to use.
          forward_only: 前向或反向运算标志位
        返回:
        	一个由梯度范数组成的三重范数（如果不使用反向传播,则为无）。
    平均困惑度和输出
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            answer_weights disagrees with bucket size for the specified bucket_id.
        """
        # 问答匹配桶尺寸
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(answer_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(answer_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, answer_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.answer_weights[l].name] = answer_weights[l]

        # Since our answers are decoder inputs shifted by one, we need one more.
        last_answer = self.decoder_inputs[decoder_size].name
        input_feed[last_answer] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self,data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: 词向量列表,如[[[4,4],[5,6,8]]]
          bucket_id: 桶编号,值取自桶对话占比
        Returns:
          The triple (encoder_inputs, decoder_inputs, answer_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        # 问题和答案的数据量:桶的话数buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        encoder_size, decoder_size = self.buckets[bucket_id]
        # 生成问题和答案的存储器
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            # 从问答数据集中随机选取问答
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # 问题末尾添加PAD_ID并反向排序
            encoder_pad = [word_to_vec.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # 答案添加GO_ID和PAD_ID
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([word_to_vec.GO_ID] + decoder_input +
                                  [word_to_vec.PAD_ID] * decoder_pad_size)

        # 问题,答案,权重批量数据
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # 批量问题
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        # 批量答案
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

            # 答案权重即Attention机制
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # 若答案为PAD则权重设置为0,因为是添加的ID
                # 其他的设置为1
                if length_idx < decoder_size - 1:
                    answer = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or answer == word_to_vec.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        print("encoder inputs: {}, decoder inputs: {}, answer weights: {}".format(encoder_inputs, decoder_inputs,
                                                                                  answer_weights))

model = Seq2SeqModel(source_vocab_size=vocabulary_encode_size, target_vocab_size=vocabulary_decode_size,
                                   buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
                                   batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.97,
                                   forward_only=False)

# 使用GPU配置
# # config = tf.ConfigProto()
# # config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory
if __name__ == "__main__":

    with tf.Session() as sess:

        # with tf.Session(config=config) as sess:
        # 检查是否有已存在的训练模型
        # 有模型则获取模型轮数,接着训练
        # 没有模型则从开始训练
        ckpt = tf.train.get_checkpoint_state('./models')
        if ckpt != None:
            train_turn = ckpt.model_checkpoint_path.split('-')[1]
            print("model path: {}, train turns: {}".format(ckpt.model_checkpoint_path, train_turn))
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            total_step = int(train_turn)
        else:
            sess.run(tf.global_variables_initializer())
            total_step = 0

        train_set = read_data(train_encode_vec, train_decode_vec)
        test_set = read_data(test_encode_vec, test_decode_vec)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
        # print("train bucket sizes: {}".format(train_bucket_sizes))
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in
                               range(len(train_bucket_sizes))]

        loss = 0.0
        # total_step = int(train_turn)
        previous_losses = []
        # 一直训练，每过一段时间保存一次模型
        while True:
            random_number_01 = np.random.random_sample()
            # get minimum i as bucket id when value > randmom value
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, answer_weights = model.get_batch(train_set, bucket_id)
            # print("encoder inputs: {}, decoder inputs: {}, answer weights: {}".format(encoder_inputs, decoder_inputs, answer_weights))
            gradient_norm, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, answer_weights, bucket_id,
                                                     False)
            print("gradient norm: {}, step loss: {}".format(gradient_norm, step_loss))
            loss += step_loss / 500
            total_step += 1

            print("total step: {}".format(total_step))
            if total_step % 500 == 0:
                print("global step: {}, learning rate: {}, loss: {}".format(model.global_step.eval(),
                                                                            model.learning_rate.eval(), loss))

                # 如果模型没有得到提升，减小learning rate
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # 保存模型
                checkpoint_path = "./models/chatbot_seq2seq.ckpt"
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                loss = 0.0
                # 使用测试数据评估模型
                for bucket_id in range(len(buckets)):
                    if len(test_set[bucket_id]) == 0:
                        continue
                    encoder_inputs, decoder_inputs, answer_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, answer_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("bucket id: {}, eval ppx: {}".format(bucket_id, eval_ppx))