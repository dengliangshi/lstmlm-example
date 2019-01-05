#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import os
import codecs

# Third-party Libraries
import numpy as np

# User Define Modules


# --------------------------------------------------------Global Strings----------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
class Vocab(object):
    """Build up word vocabulary from training dataset.
    """
    def __init__(self):
        """Initialize vocabulary.
        """
        self.vocab_size = 0           # the size of vocabulary.
        self.seq_length = 0           # the maximum length of sequence.
        self.bos_mark   = '['         # end mark of a sentence.
        self.eos_mark   = ']'         # end mark of a sentence.
        self.oov_word   = '|'         # mark for words out of vocabulary.
        self.pad_mark   = '~'         # mark for paddings for sequence.
        self.token2id   = {}          # map token to index.
        self.id2token   = {}          # map index to token.
    
    def _collect_token(self, train_file):
        """Build up vocabulary from training dataset.
        :Param train_file: the path of training file.
        """
        vocab = {}           # tokens and their frequency from dataset

        input_file = codecs.open(train_file, 'r', 'utf-8')
        for sentence in input_file:
            for token in sentence.strip().split():
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
        input_file.close()
        return vocab

    def _build_vocab(self, vocab_file):
        """Build up vocabulary from training dataset.
        :Param vocab_file: path of file for saving vocabulary.
        """
        token2id = {}

        self.logger.info('build up vocabulary from training dataset.')
        train_file = os.path.join(self.data_path, 'train.txt')
        vocab = self._collect_token(train_file)
        token2id[self.pad_mark] = 0
        token2id[self.bos_mark] = 1
        token2id[self.eos_mark] = 2
        token2id[self.oov_word] = 3
        self._assign_index(vocab, token2id, self.vocab_size, 4)
        self._save_vocab(vocab_file, token2id)

    def _assign_index(self, vocab, item2id, vocab_size, item_num):
        """Assign each item in vocabulary with an unique index.
        :Param vocab     : items and their frequency from dataset.
        :Param item2id   : map items to their index.
        :Param vocab_szie: specify the size of target vocabulary.
        :Param item_num  : count the number of items.
        """
        sorted_vocab = sorted(vocab.items(),
            key = lambda x: x[1], reverse = True)
        if vocab_size > 0:
            sorted_vocab = sorted_vocab[:vocab_size]
        for item, _ in sorted_vocab:
            if item in item2id:
                continue
            item2id[item] = item_num
            item_num += 1
    
    def _save_vocab(self, vocab_file, token2id):
        """Save vocabulary into given file.
        :Param vocab_file: the path of file for saving vocabulary.
        :Param token2id  : dict for mapping from token to index.
        """
        output_file = codecs.open(vocab_file, 'w', 'utf-8')
        for token, index in token2id.items():
            output_file.write('%s\t%d\n' % (token, index))
        output_file.close()

    def _load_vocab(self, vocab_file, token2id, id2token):
        """Load in vocabulary from existing file.
        :Param vocab_file: the path of file for vocabulary.
        :Param token2id  : dict for mapping from token to index.
        :Param id2token  : dict for mapping from index to token.
        """
        vocab_size = 0       # count the size of vocabualry

        input_file = codecs.open(vocab_file, 'r', 'utf-8')
        for line in input_file:
            token, index = line.strip().split('\t')
            token2id[token] = int(index)
            id2token[int(index)] = token
            vocab_size += 1
        input_file.close()
        return vocab_size
    
    def init_model(self, logger, args):
        """Initialize parameters for vocabulary.
        :Param logger: logger for deal with information during processing.
        :Param args  : configration parameters using tensorflow flags.
        """
        self.logger = logger
        self.logger.info('initialize vocabulary using given arguments.')
        self.vocab_size  = args.vocab_size
        self.oov_word    = args.oov_mark
        self.bos_mark    = args.bos_mark
        self.eos_mark    = args.eos_mark
        self.pad_mark    = args.pad_mark
        self.data_path   = args.data_path
        self.output_path = args.output_path
        # build up or load in vocabulary
        vocab_file = os.path.join(self.output_path, 'vocab.txt')
        if not os.path.isfile(vocab_file):
            self._build_vocab(vocab_file)
        self.logger.info('load in vocabulary from existing file.')
        self.vocab_size = self._load_vocab(vocab_file, self.token2id, self.id2token)
        return self.vocab_size

    def get_batches(self, batch_size, seq_length, data_type):
        """Get batches from specified dataset for model.
        :Param batch_size: size of each data batch.
        :Param seq_length: length of each sequence in batch.
        :Param data_type : target dataset, training, validation or test.
        """
        file_name = ('%s.txt' % data_type)
        batch_file = os.path.join(self.output_path, file_name)
        input_file = codecs.open(batch_file, 'r', 'utf-8')
        index_vector = np.array([int(x)
            for x in input_file.read().strip().split()])
        batch_num = int(len(index_vector) / (batch_size * seq_length))
        end_index = batch_num * batch_size * seq_length
        input_vector = index_vector[:end_index]
        output_vector = np.copy(input_vector)
        output_vector[:-1] = input_vector[1:]
        output_vector[-1] = input_vector[0]
        input_batch = np.split(input_vector.reshape(
            batch_size, -1), batch_num, 1)
        output_batch = np.split(output_vector.reshape(
            batch_size, -1), batch_num, 1)
        for index in range(batch_num):
            yield input_batch[index], output_batch[index]
        input_file.close()