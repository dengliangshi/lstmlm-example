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
import os

# Third-party Libraries
import numpy as np
import tensorflow as tf

# User Defined Modules
from vocab import Vocab
from .graph import Graph

# ------------------------------------------------------------Global--------------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
class Model(object):
    """This module is responsible for training or evaluating writing system model.
    """
    def __init__(self, is_train = True):
        """Initialize this module.
        :Param is_train: if create this model for training.
        """
        self.batch_size  = 0            # size of batches.
        self.epoch_num   = 0            # maximum number of training epoch.
        self.learn_rate  = 0            # initial learn rate for training.
        self.seq_length  = 0            # maximum length of sequence.
        self.output_path = ''           # path for saving output files.
        self.is_train =  is_train       # if create this model for training.
        # create an instance of vocabulary
        self._vocab = Vocab()
        # create a model for training if necessary
        if self.is_train: self.train_model = Graph()
        # create a model for evaluation 
        self.eval_model = Graph()

    def init_model(self, logger, args):
        """Initialize recurrent language model.
        :Param logger: logger for deal with information during processing.
        :Param args  : arguments for writing system model.
        """
        self.logger = logger
        # get configuration parameters
        self.batch_size  = args.batch_size
        self.seq_length  = args.seq_length
        self.epoch_num   = args.epoch_num
        self.learn_rate  = args.learn_rate
        # create root path for saving output files if not exists
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        self.output_path = args.output_path
        # create path for saving output files for current writing system
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        # build up vocabulary from training dataset
        vocab_size = self._vocab.init_vocab(logger, args)
        # create graph for training, validation and test
        if self.is_train:
            self.logger.info('create models for training, validation and testing.')
            self.train_model.init_model(vocab_size, args, True, False)
            self.eval_model.init_model(vocab_size, args, False, True)
        else:
            self.logger.info('create model for evaluation.')
            self.eval_model.init_model(vocab_size, args, False, False)
        # path for saving model
        self.ckpt_path = os.path.join(self.output_path, 'ckpt')

    def _eval_model(self, session, model, batches, seq_length):
        """Evaluate trained writing system model on given dataset.
        :Param session   : instance of tensorflow session.
        :Param model     : trained model to be evaluted.
        :Param vocab     : instance of vocabulary.
        :Param seq_length: the maximum length of sequence.
        """
        total_cost = []

        pre_state = session.run(model.init_state)
        for input_vectors, target_indices in batches:
            feed_dict = {model.input_holder: input_vectors,
                model.target_hoder: target_indices}
            for index, (h, c) in enumerate(model.init_state):
                feed_dict[c] = pre_state[index].c
                feed_dict[h] = pre_state[index].h
            pre_state, cost = session.run(
                fetches = [model.final_state, model.cost], 
                feed_dict = feed_dict
            )
            total_cost.append(cost / seq_length)
        return np.exp(np.mean(total_cost))
    
    def train(self):
        """Train language model on given training dataset.
        """
        pre_ppl    = 1e+4        # ppl on validation dataset at previous epoch
        adjust_num = False       # if learning rate has been cutted off

        # raise warining if this model is not created for training
        if not self.is_train: raise Exception('this model is not created for trianing')
        self.logger.info('start training language model.')
        with tf.Session() as session:
            # initialize the whole graph
            session.run(tf.global_variables_initializer())
            current_learn_rate = self.learn_rate
            for epoch in range(self.epoch_num):
                total_cost = []
                # adjust learning rate during trianing
                session.run(tf.assign(self.train_model.learn_rate, current_learn_rate))
                # get training batches from pre-processed files
                train_batches = self._vocab.get_batches(self.batch_size, self.seq_length, 'train')
                pre_state = session.run(self.train_model.init_state)
                for input_tensor, target_indices in train_batches:
                    feed_dict = {self.train_model.input_holder: input_tensor,
                        self.train_model.target_hoder: target_indices}
                    for index, (h, c) in enumerate(self.train_model.init_state):
                        feed_dict[c] = pre_state[index].c
                        feed_dict[h] = pre_state[index].h
                    pre_state, cost, _ = session.run(fetches = [self.train_model.final_state,
                        self.train_model.cost, self.train_model.train_op], feed_dict = feed_dict)
                    total_cost.append(cost / self.seq_length)
                valid_batches = self._vocab.get_batches(self.batch_size, self.seq_length, 'valid')
                ppl = self._eval_model(session, self.eval_model, valid_batches, self.seq_length)
                self.logger.info('training loss: %.2f, PPL on validation dataset: %.2f'
                    % (np.mean(total_cost), ppl))
                if ppl > pre_ppl:
                    current_learn_rate = self.learn_rate / 2
                    self.train_model.load_model(session, self.ckpt_path)
                    adjust_num += 1
                else:
                    self.train_model.save_model(session, self.ckpt_path, epoch)
                    pre_ppl = ppl
                # apply early stop strategy
                if adjust_num > 1: break
            # evaluate trained model on test date set
            test_batches = self._vocab.get_batches(self.batch_size, self.seq_length, 'test')
            ppl = self._eval_model(session, self.eval_model, test_batches, self.seq_length)
            self.logger.info('PPL on test dataset: %.2f' % ppl)

    def evaluate(self):
        """Evaluate saved model on given test dataset.
        """
        total_cost = []

        self.logger.info('start evaluating writing system model.')
        with tf.Session() as session:
            # initialize the whole graph for evaluation model
            session.run(tf.global_variables_initializer())
            self.logger.info('load in whole model from checkpoint path')
            self.eval_model.load_model(session, self.ckpt_path)
            # evaluate model on test dataset
            test_batches = self.vocab.get_batches(self.batch_size, self.seq_length, 'test')
            ppl = self._eval_model(session, self.eval_model, test_batches, self.seq_length)
            self.logger.info('PPL on test dataset: %.2f' % ppl)