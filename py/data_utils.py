""" Utilities for data preparing """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from tensorflow.python.platform import gfile

from env import *

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def basic_tokenizer(sentence):
    words = []
    words = sentence.split()
    return words

def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False, with_start_vocab = True):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
    vocabulary_path: path where the vocabulary will be created.
    data_paths: data files that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
    if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):

        vocab = {}
        for data_path in data_paths:
            print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
            with gfile.GFile(data_path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        word = w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
        if with_start_vocab:                    
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab_list = sorted(vocab, key=vocab.get, reverse=True)        
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
      rev_vocab = []
      with gfile.GFile(vocabulary_path, mode="rb") as f:
          rev_vocab.extend(f.readlines())
      rev_vocab = [line.strip() for line in rev_vocab]
      vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
      return vocab, rev_vocab
  else:
      raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]



def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, n_sense = 4, tokenizer=None, latent = False):
  """Get WMT data into data_dir, create vocabularies and tokenize data.
  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = os.path.join(data_dir,'train')
  dev_path = os.path.join(data_dir,'dev')  
  test_path = os.path.join(data_dir, 'test')
  
  # Create vocabularies of the appropriate sizes.
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
  create_vocabulary(en_vocab_path, [train_path + ".arg1",train_path + ".arg2"], en_vocabulary_size, tokenizer)
  sense_vocab_path = os.path.join(data_dir,'vocab{}.sense'.format(n_sense))
  create_vocabulary(sense_vocab_path, [train_path + ".rl"], n_sense, tokenizer, with_start_vocab = False)
  
  # Create token ids for the training data.
  fr_train_ids_path = train_path + (".ids%d.arg2" % en_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.arg1" % en_vocabulary_size)
  sense_train_ids_path = train_path + (".ids{}.rl".format(n_sense))
  data_to_token_ids(train_path + ".arg2", fr_train_ids_path, en_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".arg1", en_train_ids_path, en_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".rl", sense_train_ids_path, sense_vocab_path)

  # Create token ids for the development data.
  fr_dev_ids_path = dev_path + (".ids%d.arg2" % en_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.arg1" % en_vocabulary_size)
  sense_dev_ids_path = dev_path + (".ids{}.rl".format(n_sense))

  data_to_token_ids(dev_path + ".arg2", fr_dev_ids_path, en_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".arg1", en_dev_ids_path, en_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".rl", sense_dev_ids_path, sense_vocab_path)
  
  # Create token ids for the test data.
  fr_test_ids_path = test_path + (".ids%d.arg2" % en_vocabulary_size)
  en_test_ids_path = test_path + (".ids%d.arg1" % en_vocabulary_size)
  sense_test_ids_path = test_path + (".ids{}.rl".format(n_sense))

  data_to_token_ids(test_path + ".arg2", fr_test_ids_path, en_vocab_path, tokenizer)
  data_to_token_ids(test_path + ".arg1", en_test_ids_path, en_vocab_path, tokenizer)
  data_to_token_ids(test_path + ".rl", sense_test_ids_path, sense_vocab_path)

  # for type

  return (en_train_ids_path, fr_train_ids_path, sense_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path, sense_dev_ids_path,
          en_test_ids_path, fr_test_ids_path, sense_test_ids_path,
          en_vocab_path, sense_vocab_path )
      

if __name__ == "__main__":
    prepare_wmt_data(dir_data,1000,1000,tokenizer = basic_tokenizer)
