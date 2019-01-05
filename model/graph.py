#encoding utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Toolkit for Writing System Modeling
# Copyright (c) 2018, Dengliang Shi & Pingping Chen
# Author: Dengliang Shi, dengliang.shi@yahoo.com
#         Pingping Chen, pingping.chen@gmail.com
# Apache 2.0 License
# --------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries


# Third-party Libraries
import numpy as np
import tensorflow as tf

# User Defined Modules


# ------------------------------------------------------------Global--------------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
class Graph(object):
    """Graph for writing system model.
    """
    def __init__(self):
        """Initialize this module.
        """
        self.vocab_size    = 0           # size of word or character vocabulary.
        self.batch_size    = 0           # size of a training batch.
        self.seq_length    = 0           # length of a input sequence.
        self.unit_num      = 0           # number of units in lstm hidden layers.
        self.layer_num     = 0           # number of lstm hidden layers.
        self.embedding_dim = 0           # dimension of embeddings for words or characters.
        self.ikeep_prob    = 0           # keep probability for dropout at input layer.
        self.hkeep_prob    = 0           # keep probability for dropout at hidden layer.
        self.grad_cutoff   = 0           # value for gradients cutoff.

    def init_model(self, vocab_size, args, is_train = False, is_reuse = False):
        """Initialize recurrent language model.
        :Param vocab_size: size of word or character vocabulary.
        :Param args      : arguments for writing system model.
        :Param is_train  : if this model is for training
        :Param is_reuse  : if reuse the graph.
        """
        # get configuration parameters
        self.vocab_size    = vocab_size
        self.embidding_num = embidding_num
        self.batch_size    = args.batch_size
        self.seq_length    = args.seq_length
        self.unit_num      = args.unit_num
        self.layer_num     = args.layer_num
        self.embedding_dim = args.embedding_dim
        self.ikeep_prob    = args.input_keep_prob
        self.hkeep_prob    = args.hidden_keep_prob
        self.grad_cutoff   = args.grad_cutoff
        # create the whole graph of lstm-rnnlm
        self._create_graph(is_train, is_reuse)
        # set up a model saver to save model during training
        self.model_saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)

    def _lstm_layers(self, input_tensor, unit_num, layer_num, keep_prob = 0.5,
        is_train = False, is_reuse = False):
        """Long-short term memory (LSTM) recurrent neural network layer.
        :Param input_tensor: batch of input data, [batch_size, seq_len, embedding_dim].
        :Param unit_num    : the size of hidden layer.
        :Param layer_num   : number of hidden layers.
        :Param keep_prob   : keep probabilty for dropout, default is 0.5.
        :Param is_train    : if create graph for training, default is False.
        :Param is_reuse    : if reuse this graph, default is False.
        """
        with tf.variable_scope('LSTM', reuse = is_reuse) as scope:
            lstm_cells = []
            batch_size = input_tensor.shape[0]
            # create lstm cells for lstm hidden layers
            for i in range(layer_num):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(unit_num, forget_bias = 1.0)
                # apply dropout to hidden layers except the last one  if training 
                if is_train and (keep_prob < 1) and (i < layer_num - 1):
                    wrapper_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                        output_keep_prob = keep_prob)
                    lstm_cells.append(wrapper_cell)
                else:
                    lstm_cells.append(lstm_cell)
            # multiple lstm hidden layers
            multi_cells = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple = True)
            # inital state for hidden layer
            init_state = multi_cells.zero_state(batch_size, dtype = tf.float32)
            # final output and state of hidden layer
            output, final_state = tf.nn.dynamic_rnn(inputs = input_tensor,
                cell = multi_cells, initial_state = init_state, dtype = tf.float32)
        return init_state, final_state, output

    def _create_graph(self, is_train = False, is_reuse = False):
        """Create the whole graph for language model.
        :Param is_train  : if this model is for training
        :Pararm is_reuse : if reuse the graph.
        """
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope('LSTMLM', reuse = is_reuse, initializer = initializer) as scope:
            # place a holder for input vectors
            self.input_holder = tf.placeholder(shape = [self.batch_size,
                self.seq_length], name = 'input_holder', dtype = tf.int32)
            # place a holder for target token index
            self.target_hoder = tf.placeholder(shape = [self.batch_size,
                self.seq_length], name = 'target_holder', dtype = tf.int32)
            # create embedding lookup table for tokens
            embeddings = tf.get_variable(shape = [self.vocab_size,
                embedding_dim], name = 'embeddings', dtype = tf.float32)
            input_tensor = tf.nn.embedding_lookup(embeddings, input_holder)
            # apply dropout on input layer
            if is_train and self.input_keep_prob < 1:
                self.input_tensor = tf.nn.dropout(self.input_tensor, self.ikeep_prob)
            # hidden layers of language model
            self.init_state, self.final_state, lstm_output = self._lstm_layers(self.input_tensor,
                self.unit_num, self.layer_num, self.hkeep_prob, is_train, is_reuse)
            # weight for output layer of language model
            weight = tf.get_variable(shape = [self.unit_num, self.vocab_size],
                name = 'weight', dtype = tf.float32)
            # bias terms for output layer of language model
            bias = tf.get_variable(shape = [self.vocab_size], name = 'bias', dtype = tf.float32)
            # reshape output of hidden layers to [batch_size * seq_len, hidden_size]
            reshape_state = tf.reshape(lstm_output, [-1, self.unit_num])
            # the unnormalized probability
            self.logits = tf.matmul(reshape_state, weight) + bias
            # calculate the softmax loss of model
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits = [self.logits, ], 
                targets = [tf.reshape(self.target_hoder, [-1])], weights = [tf.ones([self.batch_size * self.seq_length])])
            self.cost = tf.reduce_sum(self.loss) / self.batch_size
            if is_train: self.learn_rate, self.train_op = self._train_op(self.cost, self.grad_cutoff)

    def _train_op(self, batch_cost, grad_cutoff):
        """Training operations for training the whole model.
        :Param batchcost  : value of cost function on current training batch.
        :Param grad_cutoff: vaule for gradients cutoff.
        """
        with tf.variable_scope('Train-OP') as scope:
            learn_rate = tf.Variable(0.0, trainable = False)
            # optimize the model using SGD method
            optimizer = tf.train.GradientDescentOptimizer(learn_rate)
            train_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(batch_cost, train_vars), grad_cutoff)
            global_step = tf.Variable(0, name = 'global_step')
            train_op = optimizer.apply_gradients(grads_and_vars = zip(grads, train_vars),
                global_step = global_step, name = 'apply_gradients')
        return learn_rate, train_op

    def save_model(self, session, ckpt_path, global_step):
        """Save the whole model under given path.
        :Param session    : instance of tensorflow session.
        :Param ckpt_path  : path for saving model.
        :Param global_step: the step at which the model will be save.
        """
        # create folder for check points if does not exist
        if not os.path.exists(ckpt_path): os.mkdir(ckpt_path)
        # save ckeckpoints under given path
        checkpoint = os.path.join(ckpt_path, 'wsm.ckpt')
        self.model_saver.save(session, checkpoint, global_step = global_step)

    def load_model(self, session, ckpt_path):
        """Load in model from given path.
        :Param session    : instance of tensorflow session.
        :Param ckpt_path  : path for saving model.
        """
        ckpt_state =  tf.train.get_checkpoint_state(ckpt_path)
        if not (ckpt_state and tf.train.checkpoint_exists(ckpt_state.model_checkpoint_path)):
            raise Exception('model does not exists.')
        self.model_saver.restore(session, ckpt_state.model_checkpoint_path)