#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import argparse

# Third-party Libraries


# User Define Modules


# --------------------------------------------------------Global Strings----------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
parser = argparse.ArgumentParser(prog = 'python main.py', formatter_class = argparse.ArgumentDefaultsHelpFormatter,
	description = 'Arguments for language model')
parser.add_argument('--data_path',        type = str,   default = './data',
    help = 'path for data files.')
parser.add_argument('--output_path',      type = str,   default = './result',
    help = 'path for saving output files.')
parser.add_argument('--vocab_size',       type = int,   default = 40000,
    help = 'specify maximum size of vocabulary.')
parser.add_argument('--bos_mark',         type = str,   default = '[',
    help = 'mark for begin of a sentence.')
parser.add_argument('--eos_mark',         type = str,   default = ']',
    help = 'mark for end of a sentence.')
parser.add_argument('--oov_mark',         type = str,   default = '|',
    help = 'mark for token out of vocabulary.')
parser.add_argument('--pad_mark',         type = str,   default = '~',
    help = 'mark for paddiing in a sequence.')
parser.add_argument('--batch_size',       type = int,   default = 16,
    help = 'size of training batches.')
parser.add_argument('--seq_length',       type = int,   default = 32,
    help = 'length of input sequences for model.')
parser.add_argument('--epoch_num',        type = int,   default = 20,
    help = 'maximum number of epoch for training.')
parser.add_argument('--embedding_dim',    type = int,   default = 300,
    help = 'dimension of word or character embeddings.')
parser.add_argument('--layer_num',        type = int,   default = 2,
    help = 'number of lstm hidden layers.')
parser.add_argument('--unit_num',         type = int,   default = 256,
    help = 'number of units in lstm hidden layer.')
parser.add_argument('--grad_cutoff',      type = int,   default = 5,
    help = 'threshold for gradient cutoff.')
parser.add_argument('--learn_rate',       type = float, default = 1,
    help = 'learning rate for training model.')
parser.add_argument('--ikeep_prob',  type = float, default = 1,
    help = 'keep probability for dropout in input layer.')
parser.add_argument('--hkeep_prob', type = float, default = 1,
    help = 'keep probability for dropout in hidden layers.')