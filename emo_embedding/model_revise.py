import tensorflow as tf
from tensorflow.python.framework import ops
import math
import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops


class ECM_Model(object):
	def __init__(self, max_length_posts, max_length, emotion_num, word_to_idx, dim_embed=100, dim_hidden=256,
				 embedding_matrix=None, emo_embed=100, dtype=tf.float32, learning_rate=0.01):
		self.word_to_idx = word_to_idx
		self.idx_to_word = {i: w for w, i in word_to_idx.items()}
		self.dim_embed = dim_embed
		self.dim_hidden = dim_hidden
		self.max_length = max_length
		self.dtype = dtype
		self.emotion_num = emotion_num
		self.emotion_embed = emo_embed
		self.vocab_num = len(word_to_idx)
		self.learning_rate = learning_rate
		if embedding_matrix != None:
			self.embedding_matrix = np.load(embedding_matrix)
		else:
			self.embedding_matrix = None
		self.posts = tf.placeholder(tf.int32, [None, max_length_posts])
		self.emotion_category = tf.placeholder(tf.int32, [None, 1])
		self.response = tf.placeholder(tf.int32, [None, max_length])
		self.response_onehot = tf.one_hot(self.response, depth=self.vocab_num)
		self._start = word_to_idx['<START>']
		self._null = word_to_idx['<NULL>']
		self._end = word_to_idx['<END>']
		with tf.variable_scope('word_embedding'):
			self.w = tf.get_variable('w', [self.vocab_num, self.dim_embed],
		                    initializer=self.get_initializer(2))

		self.weight_initializer = tf.orthogonal_initializer()
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(
			minval=-1.0, maxval=1.0)

	def get_initializer(self, n):
		return tf.random_uniform_initializer(minval=-1.0 * math.sqrt(3.0 / n), maxval=math.sqrt(3.0 / n))

	def word_embedding(self, inputs, reuse=False):
		with tf.variable_scope('word_embedding', reuse=reuse):
			if self.embedding_matrix == None:
				w = tf.get_variable('w', [self.vocab_num, self.dim_embed],
									initializer=self.get_initializer(2))
			else:
				w = tf.Variable(self.embedding_matrix, name='w', dtype=tf.float32)
			x = tf.nn.embedding_lookup(w, inputs, name='word_vector')
			return x

	def emotion_embedding(self, inputs, reuse=False):
		with tf.variable_scope('emotion_embedding', reuse=reuse):
			w = tf.get_variable('w', [self.emotion_num, self.emotion_embed],
								initializer=self.get_initializer(2))
			x = tf.nn.embedding_lookup(w, inputs, name='emotion_vector')
			return x

	def _batch_norm(self, x, mode='train', name=None):
		return tf.contrib.layers.batch_norm(inputs=x,
		                                    decay=0.95,
		                                    center=True,
		                                    scale=True,
		                                    is_training=(mode == 'train'),
		                                    updates_collections=None,
		                                    scope=(name + 'batch_norm'))

	def build_model(self):
		posts = self.posts
		batch_size = tf.shape(posts)[0]
		response = self.response
		posts_embedding = tf.nn.embedding_lookup(self.w, posts)
		posts_embedding = self._batch_norm(
			posts_embedding, mode='train', name='posts')
		posts_list = tf.unstack(posts_embedding, axis=1)
		response_embedding = tf.nn.embedding_lookup(self.w, response)
		response_list = tf.unstack(response, axis=1)
		emotion_embedding = tf.tile(tf.expand_dims(tf.reshape(self.emotion_embedding(self.emotion_category),
		                               [batch_size, self.emotion_embed]), axis=1), [1, self.max_length, 1])
		start_embedding = tf.tile(tf.nn.embedding_lookup(self.w, [self._start]), [batch_size, 1])

		with variable_scope.variable_scope("attention_seq2seq", dtype=self.dtype) as scope:
			dtype = scope.dtype
			stack = []
			for i in range(2):
				stack.append(tf.contrib.rnn.GRUCell(num_units=self.dim_hidden))
			cell = tf.contrib.rnn.MultiRNNCell(cells=stack)
			encoder_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.8)
			encoder_outputs, h = tf.contrib.rnn.static_rnn(
				encoder_cell,
				posts_list,
				dtype=dtype
			)
			top_states = [
				tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
			]
			attention_states = array_ops.concat(top_states, 1)

			decoder_inputs_mat = tf.concat([tf.expand_dims(start_embedding, axis=1), response_embedding[:, 0:self.max_length-1, :]], axis=1)
			decoder_inputs_mat = tf.concat([decoder_inputs_mat, emotion_embedding], axis=2)
			decoder_inputs_mat = self._batch_norm(
				decoder_inputs_mat, mode='train', name='posts')
			decoder_inputs = tf.unstack(decoder_inputs_mat, axis=1)
			decoder_outputs, decoder_states = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs, h, attention_states, cell,
			                                            num_heads=1,
			                                            output_size=self.vocab_num)
			loss_weights = [tf.to_float(tf.not_equal(label, self._null)) for label in response_list]
			loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_outputs, response_list, loss_weights,
			                                                    self.vocab_num)
			train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

			return decoder_outputs, loss, train_op

	def build_sampler(self, t):
		posts = self.posts
		batch_size = tf.shape(posts)[0]
		posts_embedding = tf.nn.embedding_lookup(self.w, posts)
		posts_list = tf.unstack(posts_embedding, axis=1)
		emotion_embedding = tf.reshape(self.emotion_embedding(self.emotion_category, reuse=(t != 0)),
		                               [-1, self.emotion_embed])
		emotion_embedding_expand = tf.tile(tf.expand_dims(emotion_embedding, axis=1), [1, self.max_length, 1])
		start_embedding = tf.tile(tf.nn.embedding_lookup(self.w, [self._start]), [batch_size, 1])

		with variable_scope.variable_scope("attention_seq2seq", dtype=self.dtype, reuse=(t != 0)) as scope:
			dtype = scope.dtype
			stack = []
			for i in range(2):
				stack.append(tf.contrib.rnn.GRUCell(num_units=self.dim_hidden))
			cell = tf.contrib.rnn.MultiRNNCell(cells=stack)
			encoder_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.8)
			encoder_outputs, h = tf.contrib.rnn.static_rnn(
				encoder_cell,
				posts_list,
				dtype=dtype
			)
			top_states = [
				tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
			]
			attention_states = array_ops.concat(top_states, 1)

			decoder_inputs_mat = tf.tile(tf.expand_dims(start_embedding, axis=1), [1, self.max_length, 1])
			decoder_inputs_mat = tf.concat([decoder_inputs_mat, emotion_embedding_expand], axis=2)
			decoder_inputs = tf.unstack(decoder_inputs_mat, axis=1)
			test_decoder_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs,
			                                                        h,
			                                                        attention_states,
			                                                        cell,
			                                                        num_heads=1,
			                                                        output_size=self.vocab_num,
			                                                        loop_function=lambda prev, _: tf.concat([tf.nn.embedding_lookup(self.w, math_ops.argmax(prev, 1)), emotion_embedding], axis=1))
			return test_decoder_outputs