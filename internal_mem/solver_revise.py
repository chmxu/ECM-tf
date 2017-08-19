import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys

class ECMSolver(object):
    def __init__(self, model, data, val_data, **kwargs):

        self.model = model
        self.data = data
        self.val_data = val_data
        self.word_to_idx = kwargs.pop('word2idx', None)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self._start = self.word_to_idx['<START>']
        self._null = self.word_to_idx['<NULL>']
        self.idx2word = dict([(v,k) for k,v in self.word_to_idx.items()])

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self):
        # train/val dataset
        n_examples = len(self.data['questions'])
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        posts = np.array(self.data['trans_questions'])
        emotion_category = np.array(self.data['questions_emotion'])
        response = np.array(self.data['trans_answers'])

        logits_list, loss, train_op, memory_norm, memory_norm_list = self.model.build_model()
        print("model built")
        tf.get_variable_scope().reuse_variables()

        # train op
        # with tf.name_scope('optimizer'):

        print("The number of epoch: %d" % self.n_epochs)
        print("Data size: %d" % n_examples)
        print("Batch size: %d" % self.batch_size)
        print("Iterations per epoch: %d" % n_iters_per_epoch)


        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                posts = posts[rand_idxs]
                emotion_category = emotion_category[rand_idxs]
                response = response[rand_idxs]

                for i in range(n_iters_per_epoch):
                    posts_batch = posts[i * self.batch_size:(i + 1) * self.batch_size]
                    emotion_batch = np.reshape(emotion_category[i * self.batch_size:(i + 1) * self.batch_size],
                                    [-1, 1])
                    response_batch = response[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.posts: posts_batch,
                                 self.model.emotion_category: emotion_batch,
                                 self.model.response: response_batch}
                    _, l, logits_batch, memory_norm_batch, memory_norm_batch_list = sess.run([train_op, loss, logits_list, memory_norm, memory_norm_list], feed_dict)
                    # write summary for tensorboard visualization

                    if (i + 1) % self.print_every == 0:
                        response_example = [np.argmax(np.squeeze(elem[0]), axis=0) for elem in logits_batch]
                        lists = []
                        for idx in response_example:
                            if "END" not in str(self.idx2word[idx]):
                                lists.append(idx)
                            else:
                                break
                        response_example = ' '.join([str(self.idx2word[idx]) for idx in lists])
                        response_truth = [str(self.idx2word[idx]) for idx in response_batch[0]]
                        while True:
                            if response_truth[-1] == '<NULL>':
                                response_truth.pop()
                            else:
                                break
                        response_truth = ' '.join(response_truth)
                        print("epoch %d iteration %d loss %f" % (e, i, l))
                        print("Ground truth response is %s" % response_truth)
                        print("Trained response is %s" % response_example)
                        print("The final norm of internal memory is %f" % memory_norm_batch)
                        # print("The change of the norm of internal memory is")
                        # print(memory_norm_batch_list)
                        sys.stdout.flush()

                if (e + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
                    print("model-%s saved." % (e + 1))

    def apply(self, t):
        n_examples = len(self.data['questions'])
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        posts = np.array(self.data['trans_questions'])
        emotion_category = np.array(self.data['questions_emotion'])
        test_decoder_outputs = self.model.build_sampler(t)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        responses = []
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for i in range(n_iters_per_epoch):
                posts_batch = posts[i * self.batch_size:(i + 1) * self.batch_size]
                emotion_batch = np.reshape(emotion_category[i * self.batch_size:(i + 1) * self.batch_size],
                                           [-1, 1])
                feed_dict = {self.model.posts: posts_batch,
                             self.model.emotion_category: emotion_batch}
                decoder_outputs = sess.run([test_decoder_outputs], feed_dict)
                decoder_outputs = decoder_outputs[0]
                length = len(decoder_outputs[0])
                for i in range(length):
                    response = [np.argmax(np.squeeze(elem[i]), axis=0) for elem in decoder_outputs]
                    lists = []
                    for idx in response:
                        if "END" not in str(self.idx2word[idx]):
                            lists.append(idx)
                        else:
                            break
                    response = ' '.join([str(self.idx2word[idx]) for idx in lists])
                    responses.append(response)
            return responses

    def test(self):
        n_examples = len(self.data['questions'])
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        posts = np.array(self.data['trans_questions'])
        emotion_category = np.array(self.data['questions_emotion'])
        test_decoder_outputs = self.model.build_sampler(t=0)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        responses = []
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for i in range(n_iters_per_epoch):
                posts_batch = posts[i * self.batch_size:(i + 1) * self.batch_size]
                emotion_batch = np.reshape(emotion_category[i * self.batch_size:(i + 1) * self.batch_size],
                                           [-1, 1])
                feed_dict = {self.model.posts: posts_batch,
                             self.model.emotion_category: emotion_batch}
                decoder_outputs = sess.run([test_decoder_outputs], feed_dict)
                decoder_outputs = decoder_outputs[0]
                length = len(decoder_outputs[0])
                for i in range(length):
                    response = [np.argmax(np.squeeze(elem[i]), axis=0) for elem in decoder_outputs]
                    lists = []
                    for idx in response:
                        if "END" not in str(self.idx2word[idx]):
                            lists.append(idx)
                        else:
                            break
                    response = ' '.join([str(self.idx2word[idx]) for idx in lists])
                    responses.append(response)
        new_data = {"questions": self.data['questions'], "answers": responses,
                    "ground_truth": self.data['answers'], "emotion_category": self.data['questions_emotion']}
        return new_data

