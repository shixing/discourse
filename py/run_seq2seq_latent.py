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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
#from tensorflow.models.rnn.translate import data_utils
#from tensorflow.models.rnn.translate import seq2seq_model
import seq2seq_model_latent as seq2seq_model
from munkres import Munkres, print_matrix
import pandas as pd
import configparser
import env

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(10, 10),(20, 20),(30, 30),(40, 40),(60,60),(100,100)]


def read_data(source_path, target_path, hidden_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
        hidden_path: path to the file with token-ids for the hidden variable z; 
        max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            with tf.gfile.GFile(hidden_path,mode="r") as hidden_file:
                source, target, z = source_file.readline(), target_file.readline(), hidden_file.readline()
                counter = 0
                while source and target and z and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    z_id = [int(x) for x in z.split()]
                    target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids, z_id])
                            break
                    source, target, z = source_file.readline(), target_file.readline(), hidden_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float32
    model = seq2seq_model.Seq2SeqModel(
        env.config.getint("model","en_vocab_size"),
        env.config.getint("model","en_vocab_size"),
        env.config.getint("model","num_z"),
        _buckets,
        env.config.getint("model","size"),
        env.config.getint("model","num_layers"),
        env.config.getfloat("model","max_gradient_norm"),
        env.config.getint("model","batch_size"),
        env.config.getfloat("model","learning_rate"),
        env.config.getfloat("model","learning_rate_decay_factor"),
        forward_only=forward_only)

    ckpt = tf.train.get_checkpoint_state(env.config.get("model","train_dir"))
    if (not env.config.getboolean("model","fromScratch")) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def show_all_variables():
    all_vars = tf.all_variables()
    for var in all_vars:
        print(var.name)


def train():
    np.set_printoptions(suppress=True)


    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.
    print("Preparing WMT data in %s" % env.config.get("model","data_dir"))
    en_train, fr_train, type_train, en_dev, fr_dev, type_dev, en_test, fr_test, type_test, _, _ = data_utils.prepare_wmt_data(env.config.get("model","data_dir"), env.config.getint("model","en_vocab_size"),latent = True, n_sense = env.config.getint("model","num_z"))

    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:

        # Create model.
        print("Creating %d layers of %d units." % (env.config.getint("model","num_layers"), env.config.getint("model","size")))
        model = create_model(sess, False)
        
        show_all_variables()
        


        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % env.config.getint("model","max_train_data_size"))
        dev_set = read_data(en_dev, fr_dev, type_dev)
        train_set = read_data(en_train, fr_train, type_train, max_size = None)
        #test_set = read_data(en_test, fr_test, type_test, env.config.getint("model",.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = int(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

        dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
        dev_total_size = int(sum(dev_bucket_sizes))



        # set env.config.getint("model",.steps_per_checkpoint = half/ epoch

        batch_size = env.config.getint("model","batch_size")
        num_z = env.config.getint("model","num_z")
        n_epoch = env.config.getint("model","n_epoch")
        steps_per_epoch = int(train_total_size / batch_size)
        steps_per_dev = int(dev_total_size / batch_size)
        
        steps_per_checkpoint = steps_per_dev * 4
        total_steps = steps_per_epoch * n_epoch

        # reports
        print(_buckets)
        print("Train:")
        print("total: {}".format(train_total_size))
        print("buckets: ", train_bucket_sizes)
        print("Dev:")
        print("total: {}".format(dev_total_size))
        print("buckets: ", dev_bucket_sizes)
        print()
        print("Steps_per_epoch:", steps_per_epoch)
        print("Total_steps:", total_steps)
        print("Steps_per_checkpoint:", steps_per_checkpoint)

        with_labeled_data = True
        isSGD = False

        # This is the training loop
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        his = []
        local_alpha = 0.05
        low_ppx = 10000000
        low_ppx_step = 0

        while current_step < total_steps:

            # for training data
            if with_labeled_data: 
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in xrange(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])
                start_time = time.time()

                encoder_inputs, decoder_inputs, target_weights, hiddens = model.get_batch(train_set, bucket_id, num_z = num_z)

                _,_,_,L,norm,Q = model.batch_step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, labeled = True, true_hidden_inputs = hiddens)
                step_time += (time.time() - start_time) / steps_per_checkpoint
                loss += (-L) / steps_per_checkpoint / batch_size
                current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                print("--------------------","TRAIN",current_step,"-------------------")
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                train_ppx = perplexity
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(env.config.get("model","train_dir"), "translate.ckpt")
                if env.config.getboolean('model',"saveCheckpoint"):
                    print("Saving model....")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                
                
                
                
                
                # dev data
                print("--------------------","DEV",current_step,"-------------------")
                Q, L, cost, accuracy, eval_ppx = evaluate(sess, model, dev_set, _buckets, name="dev", show_stat = True, show_basic = True, show_sample = True)

                his.append([current_step, Q, L, cost, accuracy, train_ppx, eval_ppx])
                if eval_ppx < low_ppx:
                    low_ppx = eval_ppx
                    low_ppx_step = current_step

                sys.stdout.flush()
                # Decrease learning rate if current eval ppl is larger
                if len(previous_losses) > 5 and eval_ppx > max(previous_losses[-5:]):
                    break
                    #sess.run(model.learning_rate_decay_op)
                previous_losses.append(eval_ppx)
                
                # increase alpha
                if env.config.getboolean("model","withAlpha"):
                    if local_alpha + 0.1 <= 1.0:
                        local_alpha += 0.1
                        with tf.variable_scope('', reuse = True) as scope:
                            alpha = tf.get_variable("embedding_rnn_seq2seq_latent/alpha")
                            sess.run(alpha.assign([local_alpha]))
                    print("alpha", local_alpha)
                    print()

    
    low_index = 0
    low_ppx = 1000000000
    for i in xrange(len(his)):
        ep = his[i][-1]
        if low_ppx > ep :
            low_ppx = ep
            low_index = i

    theone = his[low_index]
    print(theone[0], "{:2f}/{:2f}".format(theone[-2],theone[-1]), theone[-3])

    df = pd.DataFrame(his)
    df.columns=["step","Q","L","cost", "Accuracy", "Train_ppx","Eval_ppx"]
    df.to_csv(os.path.join(env.config.get("model","train_dir"),"log.csv"))

def precision_and_recall(gold_z,predict_z,num_z, omit_0 = False):
    prefix = ""
    counts = np.zeros((num_z,3)) # [#correct, #pre, #gold]
    for i in xrange(len(gold_z)):
        g = gold_z[i]
        p = predict_z[i]
        counts[g][2] += 1
        counts[p][1] += 1
        if g == p:
            counts[g][0] += 1
    if omit_0:
        counts = counts[1:,:]
        prefix = "#"

    print(prefix+"counts:")
    print(counts)
    print(prefix+"Pre Recall F1")
    prf = np.zeros((num_z,3))
    
    if omit_0:
        prf = prf[1:,:]

    prf[:,0] = counts[:,0] / counts[:,1]
    prf[:,1] = counts[:,0] / counts[:,2]
    prf[:,2] = 2 * prf[:,0] * prf[:,1] / (prf[:,0] + prf[:,1])
    print(prf)

    macro_f1 = np.average(prf[:,2])
    accuracy = np.sum(counts[:,0]) / np.sum(counts[:,2])
    print(prefix+"Macro F1",macro_f1 * 100.0)
    print(prefix+"Accuracy", accuracy)
    


def evaluate(sess, model, data_set, _buckets, name = 'test',show_sample = False, show_stat = False,  show_basic = False):
    # Run evals on development set and print their perplexity.
    start_id = 0
    sum_eval_loss = 0
    sum_eval_Q = 0
    n_steps_dev = 0

    num_z = env.config.getint("model","num_z")
    batch_size = env.config.getint("model","batch_size")
    withCompact = env.config.getboolean("model","withCompact")
    n = 0
    accuracy = 0
    predict_z = []
    gold_z = []
    post_z_avg = np.zeros((num_z, num_z))

    
    for bucket_id in xrange(len(_buckets)):
        if len(data_set[bucket_id]) == 0:
            #print("  eval: empty bucket %d" % (bucket_id))
            continue
        
        n += len(data_set[bucket_id])
        print("Evaluating {}th bucket {} size: {}".format(bucket_id, _buckets[bucket_id], len(data_set[bucket_id])))
        start_id = 0
        while True:
            encoder_inputs, decoder_inputs, target_weights, hiddens = model.get_batch(data_set, bucket_id, start_id = start_id, num_z = num_z)
            if encoder_inputs == None:
                break
            start_id += batch_size

            log_p_z, log_p_y_gv_z, post_z, L, _, Q = model.batch_step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, forward_only = True, fetch_more = True)                
            sum_eval_loss += -L / batch_size
            sum_eval_Q += Q / batch_size
            n_steps_dev += 1

            if num_z == 5: 
                temp = np.reshape(post_z,[-1,num_z])
                temp[:,0] = 0.0
                predict_z += list(np.argmax(temp, axis = 1))
            else:
                predict_z += list(np.argmax(np.reshape(post_z,[-1,num_z]), axis = 1))


            if withCompact:
                gold_z += [x[0] for x in hiddens]
                for i in xrange(0,len(hiddens)):
                    z_type = int(hiddens[i][0])
                    post_z_avg[z_type] += post_z.reshape([-1,num_z])[i]
                    r = np.random.rand()
                    if show_sample and r < 0.002:
                        print("z",z_type)
                        #print(np.array(encoder_inputs).T[i])
                        #print(np.array(decoder_inputs).T[i])         
                        print("p(z|x)", np.exp(log_p_z.reshape([-1,num_z])[i]))
                        print("p(y|z,x)",np.exp(log_p_y_gv_z.reshape([-1,num_z])[i]))
                        print("p(z|x,y)",post_z.reshape([-1,num_z])[i])


            else:
                for i in xrange(0,len(hiddens), num_z):
                    gold_z.append(hiddens[i][0])
            
                for i in xrange(0,len(hiddens),num_z):
                    z_type = int(hiddens[i][0])
                    post_z_avg[z_type] += post_z.reshape([-1,num_z])[int(i/num_z)]
                    r = np.random.rand()
                    if show_sample and r < 0.002:
                        print("z",z_type)
                        #print(np.array(encoder_inputs).T[i])
                        #print(np.array(decoder_inputs).T[i])         
                        print("p(z|x)", np.exp(log_p_z.reshape([-1,num_z])[int(i/num_z)]))
                        print("p(y|z,x)",np.exp(log_p_y_gv_z.reshape([-1,num_z])[int(i/num_z)]))
                        print("p(z|x,y)",post_z.reshape([-1,num_z])[int(i/num_z)])

    post_z_avg = post_z_avg / np.sum(post_z_avg, axis = 1).reshape([-1,1])

    eval_ppx = math.exp(float(sum_eval_loss/n_steps_dev)) if sum_eval_loss/n_steps_dev < 300 else float("inf")
    Q = sum_eval_Q / n_steps_dev
    L = - sum_eval_loss / n_steps_dev


    #print(gold_z)
    #print(predict_z)
    #print(len(gold_z))
    #print(len(predict_z))

    if num_z == 5:
        for i in xrange(len(gold_z)):
            if gold_z[i] == 0:
                predict_z[i] = 0



    cost, indexes, matrix = get_matching_accuracy(gold_z, predict_z, num_z)
    accuracy = cost*1.0/n
    if show_stat:
        for row, column in indexes:
            value = matrix[row][column]
            print('(%d, %d) -> %d' % (row, column, value))
        print('total cost: %d' % cost)
        #print('Accuracy: {:2f}'.format(cost*1.0 / n ))
        precision_and_recall(gold_z,predict_z,num_z)
        if num_z == 5:
            precision_and_recall(gold_z,predict_z,num_z, True)
            
        print()
        
    if show_stat:
        print("p(z|x,y); row = real_z")
        print(post_z_avg)
        print()

    if show_basic:
        print(name,": perplexity %.4f" % (eval_ppx))
        print(name,": Q %.4f" % (Q))
        print(name,": L %.4f" % (L))
        print()

    return Q, L, cost, accuracy, eval_ppx
    



def get_matching_accuracy(ts, predict_zs, num_z):
    matrix = np.zeros((num_z,num_z))
    for i in xrange(len(ts)):
        t = ts[i]
        z = predict_zs[i]
        matrix[t][z] += 1

    cost_matrix = np.sum(matrix) - matrix
    m = Munkres()
    indexes = m.compute(cost_matrix)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
    return total, indexes, matrix




def main(_):
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(sys.argv[1])
    env.config = config
    gpu_name = env.config.get("model", "gpu_name")
    with tf.device(gpu_name):
        print("on GPU: ", gpu_name)
        train()

if __name__ == "__main__":
  tf.app.run()
