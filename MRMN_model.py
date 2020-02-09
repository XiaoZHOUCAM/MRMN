#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import gzip
import json
import numpy as np
import random
from collections import Counter
import numpy as np
import operator
import timeit
import time
import datetime
import argparse
import sys


class MRMN:
    def __init__(self, num_users, num_items, args, mode=1):
        self.num_users = num_users
        self.num_items = num_items
        self.graph = tf.Graph()
        self.args = args
        self.stddev = self.args.std
        # self.initializer = tf.random_normal_initializer(stddev=self.stddev)
        self.initializer = tf.random_uniform_initializer(minval=-self.stddev,
                                                         maxval=self.stddev)
        self.attention = None
        self.selected_memory = None
        self.num_mem = self.args.num_mem
        self.mode = 1
        
    def get_list_feed_dict(self,batch,mode='training'):
        if(mode=='training'):
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
            item_neg_input = [x[2] for x in batch]
            type_batch = [x[3] for x in batch]
            feed_dict = {
                self.user_input:user_input,
                self.item_input:item_input,
                self.item_input_neg:item_neg_input,
                self.input_type:type_batch,
                self.dropout:self.args.dropout
            }
        else:
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
            feed_dict = {
                self.user_input:user_input,
                self.item_input:item_input,
                self.dropout:1
             }
        feed_dict[self.learn_rate] = self.args.learn_rate
        return feed_dict
 
    def build_list_inputs(self):
        self.user_input =  tf.placeholder(tf.int32, shape=[None],name='user')
        self.item_input =  tf.placeholder(tf.int32, shape=[None],name='item')
        self.item_input_neg = tf.placeholder(tf.int32, shape=[None],
                                                    name='item_neg')
        self.input_type = tf.placeholder(tf.float32, shape=[None,5],name='type')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.label = tf.placeholder(tf.float32, shape=[None],name='labels')
        self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        self.batch_size = tf.shape(self.user_input)[0]


    def composition_layer(self, user_emb, item_emb, dist='L2', reuse=None,
                            selected_memory=None):
        energy = item_emb - (user_emb + selected_memory)
        if('L2' in dist):
            final_layer = -tf.sqrt(tf.reduce_sum(tf.square(energy), 1) + 1E-3)
        elif('L1' in dist):
            final_layer = -tf.reduce_sum(tf.abs(energy), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer = tf.reshape(final_layer,[-1,1])
        return final_layer
    
    def _build_list_network(self):
        self.build_list_inputs()
        stddev = self.stddev
        with tf.variable_scope('embedding_layer'):
            with tf.device('/cpu:0'):
                self.user_embeddings = tf.get_variable('user_emb',[self.num_users+1, self.args.embedding_size],initializer=self.initializer)
                self.item_embeddings = tf.get_variable('item_emb',[self.num_items+1, self.args.embedding_size],initializer=self.initializer)
            self.user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            self.item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
            self.item_emb_neg = tf.nn.embedding_lookup(self.item_embeddings,self.item_input_neg)

            if(self.args.constraint):
                self.user_emb = tf.clip_by_norm(self.user_emb, 1.0, axes=1)
                self.item_emb = tf.clip_by_norm(self.item_emb, 1.0, axes=1)
                self.item_emb_neg = tf.clip_by_norm(self.item_emb_neg, 1.0, axes=1)
            self.user_item_key = tf.Variable(
                            tf.random_normal(
                            [self.args.embedding_size, self.num_mem],
                            stddev=stddev))
            self.user_item_key1 = tf.Variable(
                            tf.random_normal(
                            [self.args.embedding_size, self.num_mem],
                            stddev=stddev))
            self.user_item_key2 = tf.Variable(
                            tf.random_normal(
                            [self.args.embedding_size, self.num_mem],
                            stddev=stddev))
            self.user_item_key3 = tf.Variable(
                            tf.random_normal(
                            [self.args.embedding_size, self.num_mem],
                            stddev=stddev))
            self.user_item_key4 = tf.Variable(
                            tf.random_normal(
                            [self.args.embedding_size, self.num_mem],
                            stddev=stddev))
            self.memories = tf.Variable(
                            tf.random_normal(
                            [self.num_mem, self.args.embedding_size],
                            stddev=stddev))
            
            _key = tf.multiply(self.user_emb, self.item_emb)
            self.key_attention = tf.matmul(_key,self.user_item_key)
            self.key_attention1 = tf.matmul(_key,self.user_item_key1)
            self.key_attention2 = tf.matmul(_key,self.user_item_key2)          
            self.key_attention3 = tf.matmul(_key,self.user_item_key3)
            self.key_attention4 = tf.matmul(_key,self.user_item_key4)
            self.selected_memory = tf.matmul(self.key_attention, self.memories)
            self.selected_memory1 = tf.matmul(self.key_attention1, self.memories)
            self.selected_memory2 = tf.matmul(self.key_attention2, self.memories)
            self.selected_memory3 = tf.matmul(self.key_attention3, self.memories) 
            self.selected_memory4 = tf.matmul(self.key_attention4, self.memories)
            final_layer = self.composition_layer(self.user_emb, self.item_emb,
                                selected_memory=self.selected_memory)
            final_layer1 = self.composition_layer(self.user_emb, self.item_emb,
                                selected_memory=self.selected_memory1)
            final_layer2 = self.composition_layer(self.user_emb, self.item_emb,
                                selected_memory=self.selected_memory2)
            final_layer3 = self.composition_layer(self.user_emb, self.item_emb,
                                selected_memory=self.selected_memory3)
            final_layer4 = self.composition_layer(self.user_emb, self.item_emb,selected_memory=self.selected_memory4)
            self.predict_op = final_layer
            final_layer_neg = self.composition_layer(self.user_emb, self.item_emb_neg,
                        reuse=True, selected_memory=self.selected_memory)
            #self.tmp_cost = tf.squeeze(final_layer_neg - final_layer) +  self.input_type
            self.tmp_cost = tf.squeeze(final_layer_neg - final_layer) +  self.input_type[:,0]
            self.tmp_cost1 = tf.squeeze(final_layer_neg - final_layer1) +  self.input_type[:,1]
            self.tmp_cost2 = tf.squeeze(final_layer_neg - final_layer2) +  self.input_type[:,2]
            self.tmp_cost3 = tf.squeeze(final_layer_neg - final_layer3) +  self.input_type[:,3]
            self.tmp_cost4 = tf.squeeze(final_layer_neg - final_layer4) +  self.input_type[:,4]
            self.cost = tf.reduce_sum(tf.nn.relu(self.tmp_cost)) + tf.reduce_sum(tf.nn.relu(self.tmp_cost1)) + tf.reduce_sum(tf.nn.relu(self.tmp_cost2)) + tf.reduce_sum(tf.nn.relu(self.tmp_cost3)) + tf.reduce_sum(tf.nn.relu(self.tmp_cost4))
            
            if(self.args.l2_reg>0):
                vars = tf.trainable_variables()
                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.args.l2_reg
                self.cost += lossL2
            if(self.args.opt=='SGD'):
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            elif(self.args.opt=='Adam'):
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            elif(self.args.opt=='Adadelta'):
                self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            elif(self.args.opt=='Adagrad'):
                self.opt = tf.train.AdagradOptimizer(learning_rate=self.learn_rate,
                                        initial_accumulator_value=0.9)
            elif(self.args.opt=='RMS'):
                self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate,
                                        decay=0.9, epsilon=1e-6)
            elif(self.args.opt=='Moment'):
                self.opt = tf.train.MomentumOptimizer(self.args.learn_rate, 0.9)
            tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)
            gradients = self.opt.compute_gradients(self.cost)
            self.gradients = gradients
            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                grad = tf.clip_by_value(grad, -10, 10, name=None)
                return tf.clip_by_norm(grad, self.args.clip_norm)
            if(self.args.clip_norm>0):
                clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
            else:
                clipped_gradients = [(grad,var) for grad,var in gradients]

            # grads, _ = tf.clip_by_value(tf.gradients(self.cost, tvars),-10,10)
            self.optimizer = self.opt.apply_gradients(clipped_gradients)
            self.train_op = self.optimizer

                
    def _build_network(self):
        ''' Builds Computational Graph
        '''

        self.build_inputs()
        self.target = tf.expand_dims(self.label,1)
        stddev = self.stddev

        with tf.variable_scope('embedding_layer'):
            with tf.device('/cpu:0'):
                self.user_embeddings = tf.get_variable('user_emb',[self.num_users+1,
                                                        self.args.embedding_size],
                                                        initializer=self.initializer)
                self.item_embeddings = tf.get_variable('item_emb',[self.num_items+1,
                                                        self.args.embedding_size],
                                                        initializer=self.initializer)
                self.user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
                self.item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
                if(self.args.constraint):
                    self.user_emb = tf.clip_by_norm(self.user_emb, 1.0, axes=1)
                    self.item_emb = tf.clip_by_norm(self.item_emb, 1.0, axes=1)

                if('PAIR' in self.args.rnn_type):
                    self.user_emb_neg = tf.nn.embedding_lookup(self.user_embeddings,
                                                    self.user_input_neg)
                    self.item_emb_neg = tf.nn.embedding_lookup(self.item_embeddings,
                                                    self.item_input_neg)
                    if(self.args.constraint):
                        self.user_emb_neg = tf.clip_by_norm(self.user_emb_neg, 1.0, axes=1)
                        self.item_emb_neg = tf.clip_by_norm(self.item_emb_neg, 1.0, axes=1)

        self.user_item_key = tf.Variable(
                            tf.random_normal(
                            [self.args.embedding_size, self.num_mem],
                            stddev=stddev))
        self.memories = tf.Variable(
                            tf.random_normal(
                            [self.num_mem, self.args.embedding_size],
                            stddev=stddev))
        _key = tf.multiply(self.user_emb, self.item_emb)
        self.key_attention = tf.matmul(_key, self.user_item_key)

        # print(self.key_attention)
        self.key_attention = tf.nn.softmax(self.key_attention)

        if(self.mode==1):
            self.selected_memory = tf.matmul(self.key_attention, self.memories)
            print(self.selected_memory)
        elif(self.mode==2):
            self.key_attention = tf.expand_dims(self.key_attention, 1)
            self.selected_memory = self.key_attention * self.memories
            self.selected_memory = tf.reduce_sum(self.selected_memory, 2)

        self.attention = self.key_attention

        final_layer = self.composition_layer(self.user_emb, self.item_emb,
                                selected_memory=self.selected_memory)
        if('PAIR' in self.args.rnn_type):
            final_layer_neg = self.composition_layer(self.user_emb_neg, self.item_emb_neg,
                        reuse=True, selected_memory=self.selected_memory)
            self.predict_op_neg = final_layer_neg

        self.predict_op = final_layer
        # Define loss and optimizer
        with tf.name_scope("train"):
            margin = self.args.margin

            if('PAIR' in self.args.rnn_type):
                self.cost = tf.reduce_sum(tf.nn.relu(margin - final_layer + final_layer_neg))
            else:
                self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=self.target, logits=final_layer))

            if(self.args.l2_reg>0):
                vars = tf.trainable_variables()
                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.args.l2_reg
                self.cost += lossL2

            if(self.args.opt=='SGD'):
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            elif(self.args.opt=='Adam'):
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            elif(self.args.opt=='Adadelta'):
                self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            elif(self.args.opt=='Adagrad'):
                self.opt = tf.train.AdagradOptimizer(learning_rate=self.learn_rate,
                                        initial_accumulator_value=0.9)
            elif(self.args.opt=='RMS'):
                self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate,
                                        decay=0.9, epsilon=1e-6)
            elif(self.args.opt=='Moment'):
                self.opt = tf.train.MomentumOptimizer(self.args.learn_rate, 0.9)
            tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)
            gradients = self.opt.compute_gradients(self.cost)
            self.gradients = gradients
            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                grad = tf.clip_by_value(grad, -10, 10, name=None)
                return tf.clip_by_norm(grad, self.args.clip_norm)
            if(self.args.clip_norm>0):
                clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
            else:
                clipped_gradients = [(grad,var) for grad,var in gradients]

            # grads, _ = tf.clip_by_value(tf.gradients(self.cost, tvars),-10,10)
            self.optimizer = self.opt.apply_gradients(clipped_gradients)
            self.train_op = self.optimizer

        self.post_step = []
        model_stats()
