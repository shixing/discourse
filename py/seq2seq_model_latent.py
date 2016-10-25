# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

from tensorflow.models.rnn.translate import data_utils
import seq2seq
import env


def compute_post_Q_L(log_p_y_gv_z_x, log_p_z_gv_x, previous_post_z, num_z):
    """ Compute P(z|x,y), Q and L given logP(z|x) and logP(y|z,x)
    
    Args:
        log_p_y_gv_z_x: logP(y|z,x), of shape [batch_size * num_z]
        log_p_z_gv_x: logP(z|x), of shape [batch_size * num_z]
    """
    
    log_p_y_z_gv_x = log_p_y_gv_z_x + log_p_z_gv_x
    log_p_y_z_gv_x_reshape = array_ops.reshape(log_p_y_z_gv_x, [-1, num_z])

    # unlabeled L
    uL = math_ops.reduce_sum(tf.log(math_ops.reduce_sum(tf.exp(log_p_y_z_gv_x_reshape), 1)))

    post_z = tf.nn.softmax(log_p_y_z_gv_x_reshape)
    post_z = array_ops.reshape(post_z,[-1])
    # unlabeled Q
    post_z_copy = tf.stop_gradient(post_z)
    uQ = math_ops.reduce_sum(log_p_y_z_gv_x * post_z_copy)

    # labeled Q
    lQ = math_ops.reduce_sum(log_p_y_z_gv_x * previous_post_z)
    
    # labeled L
    lL = lQ

    return post_z, uL, uQ, lL, lQ

def compute_post_Q_L_avg(log_p_y_gv_z_x, log_p_z_gv_x, previous_post_z, num_z):
    """ Compute P(z|x,y), Q and L given logP(z|x) and logP(y|z,x)
    
    Args:
        log_p_y_gv_z_x: logP(y|z,x), of shape [batch_size * num_z]
        log_p_z_gv_x: logP(z|x), of shape [batch_size * num_z]
    """
    log_p_z_uniform = tf.constant(np.log(1.0/num_z), shape=log_p_z_gv_x.get_shape(), dtype=log_p_z_gv_x.dtype)
    log_p_y_z_gv_x = log_p_y_gv_z_x + log_p_z_uniform
    log_p_y_z_gv_x_reshape = array_ops.reshape(log_p_y_z_gv_x, [-1, num_z])

    # unlabeled L
    uL = math_ops.reduce_sum(tf.log(math_ops.reduce_sum(tf.exp(log_p_y_z_gv_x_reshape), 1)))

    post_z = tf.nn.softmax(log_p_y_z_gv_x_reshape)
    post_z = array_ops.reshape(post_z,[-1])
    # unlabeled Q
    post_z_copy = tf.stop_gradient(post_z)
    uQ = math_ops.reduce_sum(log_p_y_z_gv_x * post_z_copy)

    # labeled Q
    lQ = math_ops.reduce_sum(log_p_y_z_gv_x * previous_post_z)
    
    # labeled L
    lL = lQ

    return post_z, uL, uQ, lL, lQ




class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 num_z, 
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.
        
        Args:
        source_vocab_size: size of the source vocabulary.
        target_vocab_size: size of the target vocabulary.
        num_z: size of the hidden states.
        buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
        batch_size % num_z = 0
        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.
        use_lstm: if true, we use LSTM cells instead of GRU cells.
        num_samples: number of samples for sampled softmax.
        forward_only: if set, we do not construct the backward pass in the model.
        dtype: the data type to use to store internal variables.
        """

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_z = num_z
        self.buckets = buckets
        self.batch_size = batch_size
        assert(self.batch_size % self.num_z == 0)
        self.real_batch_size = int(self.batch_size / self.num_z)
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        
        # create np_hidden_input
        self.np_hidden_input = np.array(range(self.num_z) * int(self.real_batch_size))

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, hidden_input, do_decode):
            return seq2seq.embedding_rnn_seq2seq_latent(
                encoder_inputs,
                decoder_inputs,
                hidden_input,
                cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                num_z = self.num_z,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)

        
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                    name="weight{0}".format(i)))

        self.hidden_input = tf.placeholder(tf.int32, shape = [batch_size], name = "z")
        self.previous_post_z = tf.placeholder(tf.float32, shape = [batch_size], name = "previous_post_z")

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        self.outputs, self.losses, self.log_p_zs = seq2seq.model_with_buckets_latent(self.encoder_inputs, self.decoder_inputs, self.hidden_input, targets,self.target_weights, buckets, lambda x, y, z: seq2seq_f(x, y, z, False),softmax_loss_function=softmax_loss_function, per_example_loss = True)

        # for post_z, Q and L 
        self.post_zs = []
        self.uQs = []
        self.uLs = []
        self.lQs = []
        self.lLs = []
        self.log_p_y_gv_zs = []
        for i in xrange(len(self.outputs)):
            output = self.outputs[i]
            loss = self.losses[i]
            log_p_y_gv_z = -loss
            log_p_z = self.log_p_zs[i]
            self.log_p_y_gv_zs.append(log_p_y_gv_z)

            if env.config.getboolean("model","withpz"):
                cpql = compute_post_Q_L
            else:
                cpql = compute_post_Q_L_avg
            
            post_z,uL,uQ,lL,lQ = cpql(log_p_y_gv_z, log_p_z, self.previous_post_z, self.num_z)
            self.post_zs.append(post_z)
            self.uQs.append(uQ)
            self.uLs.append(uL)
            self.lQs.append(lQ) 
            self.lLs.append(lL)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms_u = []
            self.updates_u = []
            self.gradient_norms_l = []
            self.updates_l = []
            if env.config.getboolean("model","withAdagrad"):
                opt = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            for b in xrange(len(buckets)):
                gradients = tf.gradients(-self.uQs[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
                self.gradient_norms_u.append(norm)
                self.updates_u.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

            
            for b in xrange(len(buckets)):
                gradients = tf.gradients(-self.lQs[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
                self.gradient_norms_l.append(norm)
                self.updates_l.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())


    def batch_step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, labeled = False, true_hidden_inputs = None, forward_only = False, fetch_more = False):
        '''
        EM step for labeled data and unlabeled data
        '''

        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"" %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"" %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"" %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # for hidden_input and previous_post_z
        input_feed[self.hidden_input.name] = self.np_hidden_input
        
        
        # for previous_post_z
        if labeled:
            post_z = np.zeros((len(true_hidden_inputs),))
            for i in xrange(0,len(true_hidden_inputs)):
                post_z[i] = 1.0 if i % self.num_z == true_hidden_inputs[i][0] else 0.0            
            input_feed[self.previous_post_z.name] = post_z

        # Output feed: depends on whether we do a backward step or not.

        if labeled:
            output_feed = [self.lQs[bucket_id], self.lLs[bucket_id]]
        else:
            output_feed = [self.uQs[bucket_id], self.uLs[bucket_id]]

        if not forward_only:
            if labeled:
                output_feed = [self.updates_l[bucket_id], self.gradient_norms_l[bucket_id] ] + output_feed
            else:
                output_feed = [self.updates_u[bucket_id], self.gradient_norms_u[bucket_id] ] + output_feed
        if fetch_more:
            output_feed += [ self.log_p_zs[bucket_id], self.log_p_y_gv_zs[bucket_id], self.post_zs[bucket_id]]
        
        outputs = session.run(output_feed, input_feed)
    
        log_p_z, log_p_y_gv_z, post_z, norm, L, Q = None, None, None, None, None, None
        
        if forward_only:
            Q = outputs[0]
            L = outputs[1]
        else:
            norm = outputs[1]
            Q = outputs[2]
            L = outputs[3]
        if fetch_more:
            log_p_z = outputs[-3]
            log_p_y_gv_z = outputs[-2]
            post_z = outputs[-1]
        
        return log_p_z, log_p_y_gv_z, post_z, L, norm, Q

    

    def get_batch(self, data, bucket_id, start_id = None, num_z = 1):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
          start_id: if not None, creat the batch start from a certain index.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights, hiddens) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, zs = [], [], []

        # check start_id
        if start_id != None and start_id + self.real_batch_size > len(data[bucket_id]):
            return None, None, None, None

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        withReverse = env.config.getboolean("model","withReverse")

        for i in xrange(self.real_batch_size):
            if start_id == None:
                encoder_input, decoder_input, z = random.choice(data[bucket_id])
            else:
                encoder_input, decoder_input, z = data[bucket_id][start_id + i]

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            if withReverse:
                encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            else:
                encoder_inputs.append(list(encoder_pad + encoder_input))
            

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                [data_utils.PAD_ID] * decoder_pad_size)
            zs.append(z)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.real_batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.real_batch_size)], dtype=np.int32))

          # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.real_batch_size, dtype=np.float32)
            for batch_idx in xrange(self.real_batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        def expand_n(arr,n):
            if len(arr.shape) == 1:
                expand = np.zeros((arr.shape[0] * n,))
            else:
                expand = np.zeros((arr.shape[0] * n, arr.shape[1]))

            for i in xrange(arr.shape[0]):
                for j in xrange(n):
                    expand[i * n + j] = arr[i]
            return expand

        zs = expand_n(np.array(zs), num_z)

        # expand num_z 
        for l in xrange(encoder_size):
            batch_encoder_inputs[l] = expand_n(batch_encoder_inputs[l],num_z)
        for l in xrange(decoder_size):
            batch_decoder_inputs[l] = expand_n(batch_decoder_inputs[l],num_z)
            batch_weights[l] = expand_n(batch_weights[l],num_z)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, zs
