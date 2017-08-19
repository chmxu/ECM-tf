import tensorflow as tf
import math
import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

linear = core_rnn_cell_impl._linear
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
		self.learning_rate = learning_rate
		self.emotion_embed = emo_embed
		self.vocab_num = len(word_to_idx)
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

		self.weight_initializer = tf.orthogonal_initializer()
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(
			minval=-1.0, maxval=1.0)
		with tf.variable_scope('word_embedding'):
			self.w = tf.get_variable('w', [self.vocab_num, self.dim_embed],
		                    initializer=self.get_initializer(2))


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

	def emotion_embedding(self, inputs, reuse=False):  # This is the internal memory, or the initial embedding.
		with tf.variable_scope('emotion_embedding', reuse=reuse):
			w = tf.get_variable('w', [self.emotion_num, self.emotion_embed],
								initializer=self.get_initializer(2))
			x = tf.nn.embedding_lookup(w, inputs, name='emotion_vector')
			return x

	def get_internal_weight(self, reuse=False, mode="read", decoder_input=None, attn=None, state=None):
		with tf.variable_scope("internal_weight", reuse=reuse):
			if mode == "read":
				W_r = tf.get_variable('W_r', [self.emotion_embed, self.dim_embed + 2 * self.dim_hidden],
				                      initializer=self.get_initializer(2))
				return tf.nn.sigmoid(tf.matmul(tf.concat([decoder_input, attn, state], axis=1), tf.transpose(W_r)))
			else:
				W_w = tf.get_variable('W_w', [self.emotion_embed, self.dim_hidden])
				return tf.nn.sigmoid(tf.matmul(state, tf.transpose(W_w)))

	def read_gate(self, state, decoder_input, attn, t):
		return self.get_internal_weight(reuse=(t != 0), decoder_input=decoder_input, attn=attn, state=state)

	def write_gate(self, state, t):
		return self.get_internal_weight(reuse=(t != 0), mode="write", state=state)

	def attention_decoder(self,
	                      decoder_inputs,
	                      internal_memory,
	                      initial_state,
	                      attention_states,
	                      cell,
	                      output_size=None,
	                      num_heads=1,
	                      loop_function=None,
	                      dtype=None,
	                      scope=None,
	                      initial_state_attention=False,
	                      feed_forward=False):

		if not decoder_inputs:
			raise ValueError("Must provide at least 1 input to attention decoder.")
		if num_heads < 1:
			raise ValueError("With less than 1 heads, use a non-attention decoder.")
		if attention_states.get_shape()[2].value is None:
			raise ValueError("Shape[2] of attention_states must be known: %s" %
			                 attention_states.get_shape())
		if output_size is None:
			output_size = cell.output_size

		with variable_scope.variable_scope(
						scope or "attention_decoder", dtype=dtype) as scope:
			dtype = scope.dtype

			batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
			attn_length = attention_states.get_shape()[1].value
			if attn_length is None:
				attn_length = array_ops.shape(attention_states)[1]
			attn_size = attention_states.get_shape()[2].value

			# To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
			hidden = array_ops.reshape(attention_states,
			                           [-1, attn_length, 1, attn_size])
			hidden_features = []
			v = []
			attention_vec_size = attn_size  # Size of query vectors for attention.
			for a in xrange(num_heads):
				k = variable_scope.get_variable("AttnW_%d" % a,
				                                [1, 1, attn_size, attention_vec_size])
				hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
				v.append(
					variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

			state = initial_state

			def attention(query):
				"""Put attention masks on hidden using hidden_features and query."""
				ds = []  # Results of attention reads will be stored here.
				if nest.is_sequence(query):  # If the query is a tuple, flatten it.
					query_list = nest.flatten(query)
					for q in query_list:  # Check that ndims == 2 if specified.
						ndims = q.get_shape().ndims
						if ndims:
							assert ndims == 2
					query = array_ops.concat(query_list, 1)
				for a in xrange(num_heads):
					with variable_scope.variable_scope("Attention_%d" % a):
						y = linear(query, attention_vec_size, True)
						y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
						# Attention mask is a softmax of v^T * tanh(...).
						s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
						                        [2, 3])
						a = nn_ops.softmax(s)
						# Now calculate the attention-weighted vector d.
						d = math_ops.reduce_sum(
							array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
						ds.append(array_ops.reshape(d, [-1, attn_size]))
				return ds

			outputs = []
			prev = None
			batch_attn_size = array_ops.stack([batch_size, attn_size])
			attns = [
				array_ops.zeros(
					batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
			]
			for a in attns:  # Ensure the second shape of attention vectors is set.
				a.set_shape([None, attn_size])
			if initial_state_attention:
				attns = attention(initial_state)
			memory_norm = [tf.norm(internal_memory, ord='fro', axis=(0, 1))]
			for i, inp in enumerate(decoder_inputs):
				if i > 0:
					variable_scope.get_variable_scope().reuse_variables()
				# If loop_function is set, we use it instead of decoder_inputs.
				if loop_function is not None and prev is not None:
					with variable_scope.variable_scope("loop_function", reuse=True):
						inp = loop_function(prev, i)
				if feed_forward is True and prev is not None:
					prev_symbol = math_ops.argmax(prev, 1)
					inp = tf.nn.embedding_lookup(self.w, prev_symbol)
				# Merge input and previous attentions into one vector of the right size.
				input_size = inp.get_shape().with_rank(2)[1]
				if input_size.value is None:
					raise ValueError("Could not infer input size from input: %s" % inp.name)
				g_r = self.read_gate(state[-1], decoder_input=tf.squeeze(inp), attn=attns[0], t=i)
				memory = g_r * internal_memory
				x = linear(tf.concat([inp, attns[0], memory], axis=1), input_size, True)
				# Run the RNN.
				cell_output, state = cell(x, state)
				g_w = self.write_gate(state[-1], t=i)
				internal_memory = g_w * internal_memory
				memory_norm.append(tf.norm(internal_memory, ord='fro', axis=(0, 1)))
				# Run the attention mechanism.
				if i == 0 and initial_state_attention:
					with variable_scope.variable_scope(
							variable_scope.get_variable_scope(), reuse=True):
						attns = attention(state)
				else:
					attns = attention(state)

				with variable_scope.variable_scope("AttnOutputProjection"):
					output = linear([cell_output] + attns, output_size, True)
				if loop_function is not None:
					prev = output
				if feed_forward is True:
					prev = output
				outputs.append(output)
		memory_norm = tf.convert_to_tensor(memory_norm)
		return outputs, state, internal_memory, memory_norm

	def build_model(self):
		posts = self.posts
		batch_size = tf.shape(posts)[0]
		response = self.response
		posts_embedding = tf.nn.embedding_lookup(self.w, posts)
		posts_list = tf.unstack(posts_embedding, axis=1)
		response_embedding = tf.nn.embedding_lookup(self.w, response)
		response_list = tf.unstack(response, axis=1)
		emotion_embedding = tf.reshape(self.emotion_embedding(self.emotion_category),
		                               [-1, self.emotion_embed])
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
			# decoder_inputs_mat = tf.concat([decoder_inputs_mat, emotion_embedding], axis=2)
			decoder_inputs = tf.unstack(decoder_inputs_mat, axis=1)
			decoder_outputs, decoder_states, final_memory, memory_norm_list = self.attention_decoder(decoder_inputs,
			                                                         emotion_embedding,
			                                                         h,
			                                                         attention_states,
			                                                         cell,
			                                                         num_heads=1,
			                                                         output_size=self.vocab_num)
			loss_weights = [tf.to_float(tf.not_equal(label, self._null)) for label in response_list]
			loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_outputs, response_list, loss_weights,
			                                                    self.vocab_num)
			memory_norm = tf.norm(final_memory, ord='fro', axis=(0, 1))
			loss += memory_norm
			train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
			return decoder_outputs, loss, train_op, memory_norm, memory_norm_list

	def build_sampler(self, t):
		posts = self.posts
		batch_size = tf.shape(posts)[0]
		posts_embedding = tf.nn.embedding_lookup(self.w, posts)
		posts_list = tf.unstack(posts_embedding, axis=1)
		emotion_embedding = tf.reshape(self.emotion_embedding(self.emotion_category, reuse=(t != 0)),
		                               [-1, self.emotion_embed])
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
			# decoder_inputs_mat = tf.concat([decoder_inputs_mat, emotion_embedding], axis=2)
			decoder_inputs = tf.unstack(decoder_inputs_mat, axis=1)
			test_decoder_outputs, _, _, _ = self.attention_decoder(decoder_inputs,
			                                                        emotion_embedding,
			                                                        h,
			                                                        attention_states,
			                                                        cell,
			                                                        num_heads=1,
			                                                        output_size=self.vocab_num,
			                                                        feed_forward=True)
			return test_decoder_outputs