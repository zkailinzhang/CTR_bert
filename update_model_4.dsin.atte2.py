#-*- coding:utf-8 -*-

import tensorflow as tf
from keras import backend as K
import keras
import random
import os
import pandas as pd
import numpy as np
from queue import Queue  
from typing import List, Tuple
from threading import Thread

from rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import GRUCell
from modules import embedding,positional_encoding, multihead_attention,\
    feedforward,label_smoothing
from data_iter import DataIter, FIELD, HbaseDataIterUpdate
from map2int import TO_MAP, MAP
from transport_model import trans_model
from utils import *
from Dice import dice

from deepctr.layers.sequence import (AttentionSequencePoolingLayer, BiasEncoding,
                               BiLSTM, Transformer)


import sys

import datetime

import csv
import logging
import json
import traceback
import fcntl

flags = tf.app.flags
#flags.DEFINE_integer()
flags.DEFINE_float("learning_rate",0.001,"lr []")
flags.DEFINE_float("decay_rate",0.8,"lr dacay rate")
flags.DEFINE_integer("decay_step",10000,"lr decay step")

flags.DEFINE_string("logfile",'logs.dsin.s.out',"log file to save")
flags.DEFINE_boolean("is_training",True,"True for training, False for testing")
flags.DEFINE_string("csvfile","test_metric_day",'csv file to save test metric')
flags.DEFINE_integer("batch_size",128,'batch sizes ')


FLAGS = flags.FLAGS #y以上还可以放在之后

#__file__
FLAGS.csvfile = FLAGS.csvfile + '_dsin_4.csv' 
PATH = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string("train_cnt_file",os.path.join(PATH,"train_cnt_dsin_4.npy"),'file train cnt')

#__file__
logging.basicConfig(filename=os.path.join(PATH,FLAGS.logfile),filemode='a', # w
format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y %H:%M:%S",
level=logging.DEBUG)


AD_BOUND = 10000
USER_BOUND = 10000000
USER_SUM = 10000
AD_SUM = 100000

CITY_SUM = 5000


EMBEDDING_DIM = 128
ATTENTION_SIZE = 128
ABILITY_DIM = 5


AD_IMG_VALUE_DIM = 40
AD_IMG_LABEL_DIM = 20

HIDDEN_SIZE =128

USER_API_LEN = 10
USER_API_SUM_A = 100
USER_API_SUM_B = 200
USER_API_SUM_C = 800

#文件
FILE_USER_API_A = os.path.join(PATH,"userapi_a.json")
FILE_USER_API_B = os.path.join(PATH,"userapi_b.json")
FILE_USER_API_C = os.path.join(PATH,"userapi_c.json")

class BaseModel(object):
    def __init__(self):
        pass

    def build_inputs(self):
        """
        base input !!!
        :return:
        """
        with self.graph.as_default():
            with tf.name_scope('Inputs'):
                #？？
                self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
                self.lr = tf.placeholder(tf.float64, [])
                #用户ID mid？  广告id  具体样式
                self.uid_ph = tf.placeholder(tf.int32, [None, ], name="uid_batch_ph")
                self.mid_ph = tf.placeholder(tf.int32, [None, ], name="mid_batch_ph")
                
                self.mobile_ph = tf.placeholder(tf.int32, [None, ], name="mobile_batch_ph")

                self.province_ph = tf.placeholder(tf.int32, shape=[None, ], name="province_ph")
                self.city_ph = tf.placeholder(tf.int32, shape=[None, ], name="city_ph")
                self.grade_ph = tf.placeholder(tf.int32, shape=[None, ], name="grade_ph")

                self.chinese_ph = tf.placeholder(tf.int32, shape=[None, ], name="chinese_ph")
                self.english_ph = tf.placeholder(tf.int32, shape=[None, ], name="english_ph")
                self.math_ph = tf.placeholder(tf.int32, shape=[None, ], name="math_ph")

                self.purchase_ph = tf.placeholder(tf.int32, shape=[None, ], name="purchase_ph")
                self.activity_ph = tf.placeholder(tf.int32, shape=[None, ], name="activity_ph")
                self.freshness_ph = tf.placeholder(tf.int32, shape=[None, ], name="freshness_ph")

                self.hour_ph = tf.placeholder(tf.int32, shape=[None, ], name="hour_ph")
                
                

            with tf.name_scope("Embedding_layer"):
                self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [USER_SUM, EMBEDDING_DIM])
                self.uid_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_ph)

                self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [AD_SUM, EMBEDDING_DIM])
                self.mid_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_ph)
                #3 区号
                self.mobile_embeddings_var = tf.get_variable("mobile_embedding_var", [3, 5])
                self.mobile_embedded = tf.nn.embedding_lookup(self.mobile_embeddings_var, self.mobile_ph)

                self.province_embeddings_var = tf.get_variable("province_embedding_var", [40, EMBEDDING_DIM])
                self.province_embedded = tf.nn.embedding_lookup(self.province_embeddings_var, self.province_ph)

                self.city_embeddings_var = tf.get_variable("city_embedding_var", [CITY_SUM, EMBEDDING_DIM])
                self.city_embedded = tf.nn.embedding_lookup(self.city_embeddings_var, self.city_ph)

                self.grade_embeddings_var = tf.get_variable("grade_embedding_var", [102, EMBEDDING_DIM])
                self.grade_embedded = tf.nn.embedding_lookup(self.grade_embeddings_var, self.grade_ph)

                self.chinese_embeddings_var = tf.get_variable("chinese_embedding_var", [6, ABILITY_DIM])
                self.chinese_embedded = tf.nn.embedding_lookup(self.chinese_embeddings_var, self.chinese_ph)

                self.math_embeddings_var = tf.get_variable("math_embedding_var", [6, ABILITY_DIM])
                self.math_embedded = tf.nn.embedding_lookup(self.math_embeddings_var, self.math_ph)

                self.english_embeddings_var = tf.get_variable("english_embedding_var", [6, ABILITY_DIM])
                self.english_embedded = tf.nn.embedding_lookup(self.english_embeddings_var, self.english_ph)

                self.purchase_embeddings_var = tf.get_variable("purchase_embedding_var", [6, ABILITY_DIM])
                self.purchase_embedded = tf.nn.embedding_lookup(self.purchase_embeddings_var, self.purchase_ph)

                self.activity_embeddings_var = tf.get_variable("activity_embedding_var", [6, ABILITY_DIM])
                self.activity_embedded = tf.nn.embedding_lookup(self.activity_embeddings_var, self.activity_ph)

                self.freshness_embeddings_var = tf.get_variable("freshness_embedding_var", [8, ABILITY_DIM])
                self.freshness_embedded = tf.nn.embedding_lookup(self.freshness_embeddings_var, self.freshness_ph)

                self.hour_embeddings_var = tf.get_variable("hour_embedding_var", [25, ABILITY_DIM])
                self.hour_embedded = tf.nn.embedding_lookup(self.hour_embeddings_var, self.hour_ph)
                
                
    def build_fcn_net(self, inp, use_dice=False):
        with self.graph.as_default():

            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')

            dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
            #d1 = tf.layers.dropout(d1, rate=0.5, training=self.is_training_mode)

            if use_dice:
                dnn1 = dice(dnn1, name='dice_1')
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
            #d1 = tf.layers.dropout(d1, rate=0.5, training=self.is_training_mode)
            #tf.nn.dropout(

            if use_dice:
                dnn2 = dice(dnn2, name='dice_2')
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
            self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric      
                # 四舍五入，0.5以上的都是1 ，，，，但是正样本太少了， 阈值应该设置高的  0.85 
                #  数据分布 预测分布  
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

    def train(self, sess, inps):
        pass

    def train_with_dict(self, sess, train_data):
        pass

    def calculate(self, sess, inps):
        pass

    def save(self, sess, path):
        pass
        

    def restore(self, sess, path):
        pass
        # lastesd = tf.train.latest_checkpoint(path)
        # saver = tf.train.Saver()
        # saver.restore(sess, save_path=lastesd)
        # print('model restored from %s' % lastesd)

    def build_tensor_info(self):
        """
        base tensor_info
        :return:
        """
        if len(self.tensor_info) > 0:
            print("will clear items in tensor_info")
            self.tensor_info.clear()

        base_ph = ["uid_ph", "mid_ph", "mobile_ph",
                   "province_ph", "city_ph", "grade_ph",
                   "math_ph", "english_ph", "chinese_ph",
                   "purchase_ph", "activity_ph", "freshness_ph",
                   "hour_ph"
                   
                   ]

        for i in base_ph:
            self.tensor_info[i] = tf.saved_model.build_tensor_info(getattr(self, i))

    def save_serving_model(self, sess, dir_path=None, version: int = 1):

        if dir_path is None:
            print("using the /current_path/model-serving for dir_path")
            dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-serving")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self.build_tensor_info()
        assert len(self.tensor_info) > 0, "when saving model for serving, tensor_info can't empty!"

        prediction_signature = (
            tf.saved_model.build_signature_def(
                inputs=self.tensor_info.copy(),

                outputs={"outputs": tf.saved_model.build_tensor_info(
                    self.y_hat)},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        )

        export_path = os.path.join(dir_path, str(version))
        
        #删除之前保存的，只保留最新的
        for i in range(version):
            pth = os.path.join(dir_path,str(i))
            if os.path.exists(pth):
                os.system("rm -rf {}".format(pth))


        try:
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "serving": prediction_signature,
                },
                strip_default_attrs=True
            )
            builder.save()
        except:
            pass

class UpdateModel(BaseModel):

    def __init__(self):
        self.graph = tf.Graph()
        self.tensor_info = {}

        self.build_inputs()

        # with self.graph.as_default():
        #     with tf.name_scope('Attention_layer'):
        #         attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask_ph)
        #         att_fea = tf.reduce_sum(attention_output, 1)

   

    def build_inputs(self):
        super(UpdateModel, self).build_inputs()
        with self.graph.as_default():
            with tf.name_scope('Inputs'):
                self.mid_his_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_ph')
                self.mask_ph = tf.placeholder(tf.float32, [None, None], name='mask_ph')
                self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            with tf.name_scope("Embedding_layer"):
                self.mid_his_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_ph)
            
            
            self.item_eb = self.mid_embedded
            self.item_his_eb = self.mid_his_embedded

            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)


    def train_with_dict(self, sess, train_data):
        assert isinstance(train_data, dict), "\"train_data\" must be dict!"
        loss, accuracy, _ = sess.run(
            [self.loss, self.accuracy, self.optimizer],
            feed_dict={
                # self.uid_ph: train_data["uid_ph"],
                self.mid_ph: train_data["mid_ph"],
                self.mobile_ph: train_data["mobile_ph"],
                self.province_ph: train_data["province_ph"],
                self.city_ph: train_data["city_ph"],
                self.grade_ph: train_data["grade_ph"],
                self.math_ph: train_data["math_ph"],
                self.english_ph: train_data["english_ph"],
                self.chinese_ph: train_data["chinese_ph"],
                self.purchase_ph: train_data["purchase_ph"],
                self.activity_ph: train_data["activity_ph"],
                self.freshness_ph: train_data["freshness_ph"],
                self.hour_ph: train_data["hour_ph"],
                self.mid_his_ph: train_data["mid_his_ph"],
                self.mask_ph: train_data["mask_ph"],
                self.seq_len_ph: train_data["seq_len_ph"],
                self.target_ph: train_data["target_ph"],
                self.lr: train_data["lr"], }
        )
        return loss, accuracy

    def calculate(self, sess, inps):
        probs, loss, accuracy, _ = sess.run(
            [self.y_hat, self.loss, self.accuracy, self.optimizer],
            feed_dict={
                # self.uid_ph: inps[0],
                self.mid_ph: inps[1],
                self.mobile_ph: inps[2],
                self.province_ph: inps[3],
                self.city_ph: inps[4],
                self.grade_ph: inps[5],
                self.math_ph: inps[6],
                self.english_ph: inps[7],
                self.chinese_ph: inps[8],
                self.purchase_ph: inps[9],
                self.activity_ph: inps[10],
                self.freshness_ph: inps[11],
                self.hour_ph: inps[12],
                self.mid_his_ph: inps[13],
                self.mask_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.target_ph: inps[16],
            }
        )
        return probs, loss, accuracy

    def build_tensor_info(self):
        super(UpdateModel, self).build_tensor_info()
        add_ph = ["mid_his_ph", "mask_ph", "seq_len_ph"]

        for i in add_ph:
            self.tensor_info[i] = tf.saved_model.build_tensor_info(getattr(self, i))



class UpdateModel2(UpdateModel):

    def __init__(self):
        self.graph = tf.Graph()
        self.tensor_info = {}

        self.build_inputs()

        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=1)

            #dsin
            #with tf.name_scope("Self_Attention_layer"):
               #out =  transformer(self.user_api_all_eb)
            
            hidden_units = 128 #嵌入向量长度  原为128  
            num_blocks = 1 
            num_heads = 2
            dropout_rate = 0.1
            sinusoid = False
            
            with tf.variable_scope("encoder"):
                # Embedding
                # attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
                # att_fea = tf.reduce_sum(attention_output, 1)

                self.enc_ad = embedding(self.mid_his_ph,
                                     vocab_size= AD_SUM,  #   len(de2idx), 200
                                     num_units = hidden_units,  #128
                                     zero_pad=True, # 让padding一直是0
                                     scale=True,
                                     scope="enc_embed_a")
                #self.enc = self.user_api_all_eb   # 128 * 30 * 512  批次  词数量 词嵌入长度
                if sinusoid:
                    self.enc_ad += tf.cast(positional_encoding(  #N=FLAGS.batch_size,
                                                     N=tf.shape(self.mid_his_ph)[0],
                                                     T= AD_SUM,
                                                    num_units = hidden_units,
                                                    zero_pad = False,
                                                    scale = False,
                                                    scope='enc_pe_a'),tf.float32)
                else:
                    self.enc_ad += tf.cast(embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.mid_his_ph)[1]),0),
                                                       [tf.shape(self.mid_his_ph)[0],1]),
                                          vocab_size = AD_SUM,
                                          num_units = hidden_units,
                                          zero_pad = False,
                                          scale = False,
                                          scope = "enc_pe_a"),tf.float32)
                ## Blocks
                for i in range(num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### MultiHead Attention  
                        #[128, 30, 512] 不变
                        self.enc = multihead_attention(queries = self.enc_ad,
                                                       keys = self.enc_ad,
                                                       num_units = hidden_units,
                                                       num_heads = num_heads,
                                                       dropout_rate = dropout_rate,
                                                       #is_training = is_training,
                                                       causality = False
                                                       )
                        self.enc = feedforward(self.enc,num_units = [4 * hidden_units,hidden_units])


                

            # Final linear projection
            #self.logits = tf.layers.dense(self.dec,USER_API_LEN*3))
            print(self.enc.get_shape().as_list())
            #self.final_state2 = tf.reduce_sum(self.enc,-2)
            self.item_eb_ex = tf.expand_dims(self.item_eb,axis=-2)

            # attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask_ph)
            # att_fea = tf.reduce_sum(attention_output, 1)
         
            self.mask_ph_2 = tf.ones_like(self.enc,dtype=tf.bool)
            self.final_state2 = AttentionSequencePoolingLayer(att_hidden_units=(64,16),att_activation="dice",
            weight_normalization=False, supports_masking=True)(
            inputs = [self.item_eb_ex , self.enc],mask=self.mask_ph_2)

            print(tf.shape(self.mask_ph)[1])
            print(tf.shape(self.mid_his_ph)[1])
            self.mask_ph_ = tf.squeeze(self.mask_ph,[-1])
            self.final_state2 = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                             supports_masking=False)(
             [self.item_eb_ex , self.enc,self.mask_ph])  #self.mask_ph

            #注意   -1 ，即 每个特征最终的嵌入特征空间大小128  都不一样也没关系啊
            #10个特征  B*( 128*10)    ， B*(12+800+200)
            inp = tf.concat(
                [self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 self.final_state2,

                 self.mobile_embedded,
                 self.province_embedded,
                 self.city_embedded,
                 self.grade_embedded,
                 self.chinese_embedded,
                 self.math_embedded,
                 self.english_embedded,
                 self.purchase_embedded,
                 self.activity_embedded,
                 self.freshness_embedded,
                 self.hour_embedded,

                 self.ad_img_eb_sum,
                 self.user_api_eb_sum
                 ], -1)
                  
       
        self.build_fcn_net(inp, use_dice=True,)

    def build_inputs(self):
        
        super(UpdateModel2, self).build_inputs()
         
       
        with self.graph.as_default():
            

            with tf.name_scope('Inputs'):
          
                #img AD 特征  N*F
                self.ad_label_ph = tf.placeholder(tf.int32,[None,None],name='ad_label_ph')
                #特征下的类别  N*F
                self.ad_value_ph = tf.placeholder(tf.int32,[None,None],name='ad_value_ph')

                #self.user_api_a_ph = tf.placeholder(tf.int32,[FLAGS.batch_size,USER_API_LEN],name= "user_api_a_ph")
                self.user_api_a_ph = tf.placeholder(tf.int32,[None,None],name= "user_api_a_ph")                
                self.user_api_b_ph = tf.placeholder(tf.int32,[None,None],name= "user_api_b_ph")
                self.user_api_c_ph = tf.placeholder(tf.int32,[None,None],name= "user_api_c_ph")

            with tf.name_scope("Embedding_layer"):
               
                self.ad_img_embeddings_var  = tf.get_variable("ad_img_embedding_var", [AD_IMG_LABEL_DIM,AD_IMG_VALUE_DIM,EMBEDDING_DIM])
                #索引 idx  
                self.ad_img_embedded = tf.nn.embedding_lookup(self.ad_img_embeddings_var, self.ad_label_ph)
                
                self.ad_value_ph_ohot = tf.one_hot(self.ad_value_ph,depth=AD_IMG_VALUE_DIM,axis=-1)

                self.ad_value_ph_ohot = tf.expand_dims(self.ad_value_ph_ohot,axis=-2)
                #n*7*8*128  就是对应相乘，
                self.ad_img_embedded = tf.matmul(self.ad_value_ph_ohot ,self.ad_img_embedded)        
                self.ad_img_eb = self.ad_img_embedded     # none*n*1*128     
                self.ad_img_eb = tf.squeeze(self.ad_img_eb,[-2])  #n*n*128

                self.ad_img_eb_sum = tf.reduce_sum(self.ad_img_eb,-2)
                
           # with tf.name_scope("Embedding_layer"):
                self.user_api_a_var = tf.get_variable("user_api_a_var", [USER_API_SUM_A, EMBEDDING_DIM])
                self.user_api_a_eb = tf.nn.embedding_lookup(self.user_api_a_var, self.user_api_a_ph)

                self.user_api_b_var = tf.get_variable("user_api_b_var", [USER_API_SUM_B, EMBEDDING_DIM])
                self.user_api_b_eb = tf.nn.embedding_lookup(self.user_api_b_var, self.user_api_b_ph)

                self.user_api_c_var = tf.get_variable("user_api_c_var", [USER_API_SUM_C, EMBEDDING_DIM])
                self.user_api_c_eb = tf.nn.embedding_lookup(self.user_api_c_var, self.user_api_c_ph)
                #B*30*128
                self.user_api_all_eb = tf.concat([self.user_api_a_eb,self.user_api_b_eb,self.user_api_c_eb],-1)

                self.user_api_eb_sum = tf.reduce_sum(self.user_api_all_eb,-2)


    def train(self, sess, inps):
        loss, accuracy, _ = sess.run(
            [self.loss, self.accuracy, self.optimizer],
            feed_dict={
                # self.uid_ph: inps[0],
                self.mid_ph: inps[1],
                self.mobile_ph: inps[2],
                self.province_ph: inps[3],
                self.city_ph: inps[4],
                self.grade_ph: inps[5],
                self.math_ph: inps[6],
                self.english_ph: inps[7],
                self.chinese_ph: inps[8],
                self.purchase_ph: inps[9],
                self.activity_ph: inps[10],
                self.freshness_ph: inps[11],
                self.hour_ph: inps[12],
                self.mid_his_ph: inps[13],
                self.mask_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.target_ph: inps[16],
                self.lr: inps[17],

                #单独喂入广告特征
                self.ad_label_ph: inps[18],
                self.ad_value_ph: inps[19],

                self.user_api_a_ph: inps[20],
                self.user_api_b_ph: inps[21],
                self.user_api_c_ph: inps[22],
                
            }
        )
        return loss, accuracy


    def test(self, sess, inps):

        prob, loss, acc = self.calculate(sess, inps)

        return prob, loss, acc 
        

    def train_with_dict(self, sess, train_data):
        assert isinstance(train_data, dict), "\"train_data\" must be dict!"
        loss, accuracy, _ = sess.run(
            [self.loss, self.accuracy, self.optimizer],
            feed_dict={
                # self.uid_ph: train_data["uid_ph"],
                self.mid_ph: train_data["mid_ph"],
                self.mobile_ph: train_data["mobile_ph"],
                self.province_ph: train_data["province_ph"],
                self.city_ph: train_data["city_ph"],
                self.grade_ph: train_data["grade_ph"],
                self.math_ph: train_data["math_ph"],
                self.english_ph: train_data["english_ph"],
                self.chinese_ph: train_data["chinese_ph"],
                self.purchase_ph: train_data["purchase_ph"],
                self.activity_ph: train_data["activity_ph"],
                self.freshness_ph: train_data["freshness_ph"],
                self.hour_ph: train_data["hour_ph"],
                self.mid_his_ph: train_data["mid_his_ph"],
                self.mask_ph: train_data["mask_ph"],
                self.seq_len_ph: train_data["seq_len_ph"],
                self.target_ph: train_data["target_ph"],
                self.lr: train_data["lr"], }
        )
        return loss, accuracy

    def calculate(self, sess, inps):
        probs, loss, accuracy= sess.run(
            [self.y_hat, self.loss, self.accuracy],
            feed_dict={
                # self.uid_ph: inps[0],
                self.mid_ph: inps[1],
                self.mobile_ph: inps[2],
                self.province_ph: inps[3],
                self.city_ph: inps[4],
                self.grade_ph: inps[5],
                self.math_ph: inps[6],
                self.english_ph: inps[7],
                self.chinese_ph: inps[8],
                self.purchase_ph: inps[9],
                self.activity_ph: inps[10],
                self.freshness_ph: inps[11],
                self.hour_ph: inps[12],
                self.mid_his_ph: inps[13],
                self.mask_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.target_ph: inps[16],
                #self.lr

                #单独喂入广告特征
                self.ad_label_ph: inps[17],
                self.ad_value_ph: inps[18],

                self.user_api_a_ph: inps[19],
                self.user_api_b_ph: inps[20],
                self.user_api_c_ph: inps[21]
            }
        )
        return probs, loss, accuracy

    def build_tensor_info(self):
        super(UpdateModel2, self).build_tensor_info()
        #ad img
        add_ph = ["ad_label_ph","ad_value_ph","user_api_a_ph","user_api_b_ph","user_api_c_ph"]

        for i in add_ph:
            self.tensor_info[i] = tf.saved_model.build_tensor_info(getattr(self, i))

    def save(self, sess, path):
        
        pos = path.rfind("_")
        pre = path[:pos+1]
        num = int(path[pos+1:])
        for i in range(num):
            mdoelpath = pre+ str(i)+ ".meta"
            pth = pre+ str(i)+ "*"
            if os.path.exists(mdoelpath):
                os.system("rm {}".format(pth))

            

        self.saver.save(sess, save_path=path)
    def restore(self, sess, path):
        self.saver.restore(sess, save_path=path)
        print('model restored from %s' % path)   


def parse_his(x):
    x = eval(x)
    if len(x) == 0:
        return []
    return [abs(i) if i < AD_BOUND else i - 90000 for i in x]

def pro_userapi_one():
    pass



def process_rbehavior_split(r_behavior):
    r_behavior = r_behavior.replace("[","[\"").replace(", ","\",\"").replace("]","\"]")
    # str -  list
    r_behavior = eval(r_behavior)
    #logging.info("r_behavior {}".format(r_behavior))
    
    try:
        user_api_a_update={}
        user_api_b_update={}
        user_api_c_update={}
        for api in r_behavior: 
            values_tmp = api.split("/")
            if os.path.exists(FILE_USER_API_A):
                #user_api_all={}          
                
                with open(FILE_USER_API_A,'r+') as jf:
                    fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
                    data = json.load(jf)
                    user_api_a = data["user_api_a"]
                    #
                        #logging.info("")
                    if values_tmp == ['']:
                        user_api_a[values_tmp[0]] = 1
                    else:
                        if '' in user_api_a:
                            if values_tmp[1] not in user_api_a:
                                user_api_a[values_tmp[1]] = len(user_api_a)+1                               
                        else:
                            if values_tmp[1] not in user_api_a:
                                user_api_a[values_tmp[1]] = len(user_api_a)+2
                    #jf.seek(0)
                    user_api_a_update["user_api_a"] = user_api_a
                    user_api_all_data = json.dumps(user_api_a_update)
                    #rst = jf.write(user_api_all_data)
                    jf.seek(0)

                    rst = jf.write(user_api_all_data)
                    jf.flush()
                    fcntl.flock(jf.fileno(),fcntl.LOCK_UN)
            if os.path.exists(FILE_USER_API_B):        
                with open(FILE_USER_API_B,'r+') as jf:
                    fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
                    data = json.load(jf)
                    user_api_a = data["user_api_b"]
                    
                    if values_tmp == ['']:
                        user_api_a[values_tmp[0]] = 1
                    else:
                        if '' in user_api_a:
                            if values_tmp[2] not in user_api_a:
                                user_api_a[values_tmp[2]] = len(user_api_a)+1                               
                        else:
                            if values_tmp[2] not in user_api_a:
                                user_api_a[values_tmp[2]] = len(user_api_a)+2
                    #jf.seek(0)
                    user_api_b_update["user_api_b"] = user_api_a
                    user_api_all_data = json.dumps(user_api_b_update)
                    #rst = jf.write(user_api_all_data)
                    jf.seek(0)

                    rst = jf.write(user_api_all_data)
                    jf.flush()
                    fcntl.flock(jf.fileno(),fcntl.LOCK_UN)
            if os.path.exists(FILE_USER_API_C):  
                with open(FILE_USER_API_C,'r+') as jf:
                    fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
                    data = json.load(jf)
                    user_api_a = data["user_api_c"]
                    #
                        #logging.info("")
                    if values_tmp == ['']:
                        user_api_a[values_tmp[0]] = 1
                    else:
                        if '' in user_api_a:
                            if values_tmp[3] not in user_api_a:
                                user_api_a[values_tmp[3]] = len(user_api_a)+1                               
                        else:
                            if values_tmp[3] not in user_api_a:
                                user_api_a[values_tmp[3]] = len(user_api_a)+2
                    #jf.seek(0)
                    user_api_c_update["user_api_c"] = user_api_a
                    user_api_all_data = json.dumps(user_api_c_update)
                    #rst = jf.write(user_api_all_data)
                    jf.seek(0)

                    rst = jf.write(user_api_all_data)
                    jf.flush()
                    fcntl.flock(jf.fileno(),fcntl.LOCK_UN)

           
    except Exception as e:
        logging.info("user_api_all error: {}".format(e)) 
        exce = 1.0/0.0

    rbehavior_int_a,rbehavior_int_b,rbehavior_int_c = [],[],[]

    for api in r_behavior:
        api_sp = api.split("/")
        if api_sp == ['']:

            rbehavior_int_a.append(user_api_a_update["user_api_a"][''])
            rbehavior_int_b.append(user_api_b_update["user_api_b"][''])
            rbehavior_int_c.append(user_api_c_update["user_api_c"][''])
        else:
            rbehavior_int_a.append(user_api_a_update["user_api_a"][api_sp[1]])  
            rbehavior_int_b.append(user_api_b_update["user_api_b"][api_sp[2]]) 
            rbehavior_int_c.append(user_api_c_update["user_api_c"][api_sp[3]]) 
    
    return  (rbehavior_int_a,rbehavior_int_b,rbehavior_int_c)



def process_rbehavior_split2(r_behavior):
    r_behavior = r_behavior.replace("[","[\"").replace(", ","\",\"").replace("]","\"]")
    # str -  list
    r_behavior = eval(r_behavior)
    #logging.info("r_behavior {}".format(r_behavior))
    
    
    try:
        user_api_all={}
        if os.path.exists(FILE_USER_API):
            #user_api_all={}
            
            for _ in range(20):
                with open(FILE_USER_API,'r+') as jf:

                    fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
                    data = json.load(jf)
                    user_api_a = data["user_api_a"]
                    user_api_b = data["user_api_b"]
                    user_api_c = data["user_api_c"]
                    
                
                    for api in r_behavior: 
                        values_tmp = api.split("/")
                        #logging.info("")
                        if values_tmp == ['']:
                            user_api_a[values_tmp[0]] = 1
                            user_api_b[values_tmp[0]] = 1
                            user_api_c[values_tmp[0]] = 1
                        else:
                            if '' in user_api_a:
                                if values_tmp[1] not in user_api_a:
                                    user_api_a[values_tmp[1]] = len(user_api_a)+1
                                if values_tmp[2] not in user_api_b:
                                    user_api_b[values_tmp[2]] = len(user_api_b)+1  
                                if values_tmp[3] not in user_api_c:
                                    user_api_c[values_tmp[3]] = len(user_api_c)+1        
                            else:
                                if values_tmp[1] not in user_api_a:
                                    user_api_a[values_tmp[1]] = len(user_api_a)+2
                                if values_tmp[2] not in user_api_b:
                                    user_api_b[values_tmp[2]] = len(user_api_b)+2
                                if values_tmp[3] not in user_api_c:
                                    user_api_c[values_tmp[3]] = len(user_api_c)+2

                    #jf.seek(0)
                    user_api_all["user_api_a"] = user_api_a
                    user_api_all["user_api_b"] = user_api_b
                    user_api_all["user_api_c"] = user_api_c
          
                    user_api_all_data = json.dumps(user_api_all)
                    #rst = jf.write(user_api_all_data)
                    write_ok = False
                    for _ in range(5):

                        jf.seek(0)

                        rst = jf.write(user_api_all_data)
                        if rst!=len(user_api_all_data):
                            logging.info("!!! user_api_all_data write bug")
                            logging.info("json data error:\n user_api_all_data: {},\n user_api_all:{}".format(user_api_all_data,user_api_all) )
                            continue
                        write_ok = True
                        break
                        
                    else:
                        fcntl.flock(jf.fileno(),fcntl.LOCK_UN)
                        time.sleep(10)
                        logging.info("!!! user_api_all_data write bug five")
                        
                    if write_ok:
                        jf.flush()
                        fcntl.flock(jf.fileno(),fcntl.LOCK_UN)
                        break

           
    except Exception as e:
        logging.info("user_api_all error: {}".format(e)) 
        exce = 1.0/0.0

    rbehavior_int_a,rbehavior_int_b,rbehavior_int_c = [],[],[]

    for api in r_behavior:
        api_sp = api.split("/")
        if api_sp == ['']:

            rbehavior_int_a.append(user_api_all["user_api_a"][''])
            rbehavior_int_b.append(user_api_all["user_api_b"][''])
            rbehavior_int_c.append(user_api_all["user_api_c"][''])
        else:
            rbehavior_int_a.append(user_api_all["user_api_a"][api_sp[1]])  
            rbehavior_int_b.append(user_api_all["user_api_b"][api_sp[2]]) 
            rbehavior_int_c.append(user_api_all["user_api_c"][api_sp[3]]) 
    
    return  (rbehavior_int_a,rbehavior_int_b,rbehavior_int_c)

    
def handle(data: pd.DataFrame) -> Tuple[List, List]:
    # data = data.drop(columns=["school_id", "county_id"], )
    
    to_int = ["mobile_os", "province_id",
              "grade_id", "city_id",
              "ad_id", "user_id", "log_hourtime",
              ]
    for i in to_int:
        data[i] = data[i].astype(int)

    for i in TO_MAP:
        data[i] = data[i].map(lambda x: MAP[i].get(x, 0))

    data["ad_id"] = data["ad_id"].map(lambda x: abs(x) if x < AD_BOUND else x - 90000)
    data["user_id"] = data["user_id"].map(lambda x: abs(x) % 6 if x < USER_BOUND else x - USER_BOUND)
    #这个是？？
    data["rclick_ad"] = data["rclick_ad"].map(lambda x: parse_his(x))

    #data['rencent_behavior'] = data['rencent_behavior'] .map(lambda x: process_rencent_behavior(x))
    #Pans  一列变三列    lambda 返回多个值
    data['rencent_behavior_all'] \
         = data['rencent_behavior'].map(lambda x: process_rbehavior_split(x))

    data['rencent_behavior_a'] = [ data['rencent_behavior_all'][i][0] for i in range(len(data['rencent_behavior_all']))]   
    data['rencent_behavior_b'] = [ data['rencent_behavior_all'][i][1]  for i in  range(len(data['rencent_behavior_all']))] 
    data['rencent_behavior_c'] = [ data['rencent_behavior_all'][i][2]  for i in range(len(data['rencent_behavior_all']))] 
  
    to_select = ["user_id", "ad_id", "mobile_os",
                 "province_id", "city_id", "grade_id",
                 "math_ability", "english_ability", "chinese_ability",
                 "purchase_power", "activity_degree", "app_freshness",
                 "log_hourtime", 
                 "rclick_ad",
                 "label_1","label_2","label_3","label_4","label_5","label_6","label_7",
                 "rencent_behavior_a","rencent_behavior_b","rencent_behavior_c"

                 ]
    #真的做成可自由扩展的，自适应扩展，那就检索 字符串匹配，"label_*  看有多少

    feature, target = [], []
    for row in data.itertuples(index=False):
        tmp = []
        
        for i in to_select:
            tmp.append(getattr(row, i))

        #其他不用转吧，因为喂入嵌入函数，就是索引值就可以4了，不用提前转one-hot，
        if getattr(row, "is_click") == "0":
            target.append([1, 0])
        else:
            target.append([0, 1])
        feature.append(tmp)
    
    return feature, target


def prepare_data(feature: List, target: List, choose_len: int = 0) -> Tuple:
    user_id = np.array([fea[0] for fea in feature])
    ad_id = np.array([fea[1] for fea in feature])
    mobile = np.array([fea[2] for fea in feature])
    province = np.array([fea[3] for fea in feature])
    city = np.array([fea[4] for fea in feature])
    grade = np.array([fea[5] for fea in feature])
    math = np.array([fea[6] for fea in feature])
    english = np.array([fea[7] for fea in feature])
    chinese = np.array([fea[8] for fea in feature])
    purchase = np.array([fea[9] for fea in feature])
    activity = np.array([fea[10] for fea in feature])
    freshness = np.array([fea[11] for fea in feature])
    hour = np.array([fea[12] for fea in feature])
 
    seqs_ad = [fea[13] for fea in feature]
    lengths_xx = [len(i) for i in seqs_ad]
    #ValueError: setting an array element with a sequence.  要转np.array  
    #  嵌套list不等长的时候 加 np.array 报错
    rencent_behavior_a = [fea[21] for fea in feature]
    rencent_behavior_b = [fea[22] for fea in feature]
    rencent_behavior_c = [fea[23] for fea in feature]

    if choose_len != 0:
        new_seqs_ad = []
        new_lengths_xx = []

        for l_xx, fea in zip(lengths_xx, seqs_ad):
            if l_xx > choose_len:
                new_seqs_ad.append(fea[l_xx - choose_len:])
                new_lengths_xx.append(l_xx)
            else:
                new_seqs_ad.append(fea)
                new_lengths_xx.append(l_xx)

        lengths_xx = new_lengths_xx
        seqs_ad = new_seqs_ad

    max_len = np.max(lengths_xx)
    cnt_samples = len(seqs_ad)

    ad_his = np.zeros(shape=(cnt_samples, max_len), ).astype("int64")
    ad_mask = np.zeros(shape=(cnt_samples, max_len)).astype("float32")

    for idx, x in enumerate(seqs_ad):
        ad_mask[idx, :lengths_xx[idx]] = 1.0
        ad_his[idx, :lengths_xx[idx]] = x
    
 

    label_list = []
    value_list = []
    #一个批次的  这是value
    label_all_tmp = []
    value_all_tmp = []
    for fea in feature:
        
        value_list = [fea[i] for i in range(14,21)]
        
        value_list = np.asarray(value_list,dtype=int)
        # -1 的就是没有value值的 ，去掉,   np.where 返回满足条件的索引
        value_list = value_list[np.where(value_list>-1)]
        label_list = np.where(value_list>-1)[0] 

        label_all_tmp.append(label_list)
        value_all_tmp.append(value_list.tolist())
    
    
    # 一个批次   label    
    # [1,2]        [1,2,0]  补零
    # [3,5,1]      [3,5,1]
    # ...   第二个维度不同，没法喂入 placeholder，，像rnn
    #mask   所以   这边是补 -1， 嵌入矩阵 0 表示第一行的数据，，-1才是全0
    label_len = [len(i) for i in label_all_tmp]
    label_len_max = np.max(label_len) #直接返回 第二维度的 最大维
    

 
    label_all = keras.preprocessing.sequence.pad_sequences(label_all_tmp,
    maxlen=label_len_max,padding='post',value=0)
    
    value_all = keras.preprocessing.sequence.pad_sequences(value_all_tmp,
    maxlen=label_len_max,padding='post',value=-1)
    #2 ,4     0r  2,8  
    #logging.info(" ad_lable_max {}, ad_value_max {}".format(np.max(label_all),np.max(value_all)))

    rencent_behavior_a = keras.preprocessing.sequence.pad_sequences(rencent_behavior_a,
    maxlen=USER_API_LEN,padding='post',value=0)
    rencent_behavior_b = keras.preprocessing.sequence.pad_sequences(rencent_behavior_b,
    maxlen=USER_API_LEN,padding='post',value=0)
    rencent_behavior_c= keras.preprocessing.sequence.pad_sequences(rencent_behavior_c,
    maxlen=USER_API_LEN,padding='post',value=0)

    return user_id, ad_id, mobile, province, city, grade, math, english, \
           chinese, purchase, activity, freshness, hour, \
           ad_his, ad_mask, np.array(lengths_xx), np.array(target), \
               label_all,value_all,\
               rencent_behavior_a,rencent_behavior_b,rencent_behavior_c
                   


Field = FIELD + []

# 最大队列，，文件量 400万，
MY_QUEUE = Queue(800000)


def produce(filter_str, request):
    try:
        with HbaseDataIterUpdate("10.9.75.202", b'midas_offline_v1', filter_str, request,) as d:
            for i in d.get_data(batch_size=128,model_num=0):
                MY_QUEUE.put(i)  #一直取，i 是一个批次，执行yeild下面的程序 data=[],队列的数据的单位是一个批次数据
    except:
        logging.info("*** HbaseDataIterUpdate except ***")
        logging.debug("execept e :{}".format(traceback.format_exc()))
        pass      
    finally:
        MY_QUEUE.put("done")   #一天的数据取完，done



def get_max_model_index(cnt: int = 1):
    model_path = "tmp-model-3/modeldsin_4/model/"
    if cnt != 1:
        model_path = "update-model-1/modeldsin_4/model/"
        
    #num = -1
    num =0
    #ckpt_0  拿0
    for i in os.listdir(os.path.join(PATH, model_path)):
        o = i.split(".")[0]
        try:
            a = int(o.split("_")[1])
            if a > num:
                num = a
        except:
            pass

    return num

def get_max_serving_index(cnt: int = 1):
    model_path = "tmp-model-3/modeldsin_4/serving/"
    if cnt != 1:
        model_path = "update-model-1/modeldsin_4/serving/"
        
    #num = -1
    num =0
    for i in os.listdir(os.path.join(PATH, model_path)):
        o = i.split(".")[0]
        try:
            a = int(o)
            if a > num:
                num = a
        except:
            pass

    return num


if __name__ == "__main__":
    from look_up_dir import  get_last_day_fmt,get_some_day_fmt,get_now_day_fmt
    
    #serving_iter = 5000

    save_iter = 2000  #48000  
    print_iter = 100  
    lr_iter = FLAGS.decay_step #10000  #1000
    lr =  FLAGS.learning_rate #0.001  #0.001
    
    lr_decay_rate  =FLAGS.decay_rate
    #test 
    test_max_step =30000
    
    

    #整个数据集的训练轮数
    restart_sum = 20
    #restart_cnt = 1

    #train
    train_break_sum = 8
    #break_cnt = 1
    import datetime
    import os, time

    PATH = os.path.dirname(os.path.abspath(__file__))

    filter_str = """RowFilter (=, 'substring:{}')"""
    #request = [get_last_day_fmt()]  #'2019-07-11'
    #提前建目录  去掉   "update-model-1/model/ckpt_"
    model_path = "update-model-1/modeldsin_4/model/ckpt_"
    best_model_path = "update-model-1/modeldsin_4/best-model/ckpt_"

    path1 ='update-model-1/modeldsin_4'
    pathm = os.path.join(PATH,path1,'model')
    pathb = os.path.join(PATH,path1,'best-model')
    paths = os.path.join(PATH,path1,'serving')

    if not os.path.exists(pathm):
        os.makedirs(pathm)
    if not os.path.exists(pathb):
        os.makedirs(pathb)
    if not os.path.exists(paths):
        os.makedirs(paths)

    model = UpdateModel2()
    
    version = get_max_serving_index(2) + 1
    #删除之前的 保留最大的
    

    
    MODE = {"test":False,"train":True,"serve":True}


    #6月2号没有数据 从第二天开始训练的吗
    Day_start = 'Jul 25, 2019'    # 缩写 01  1 都可以  jun jul 跨月底的
    # Day_nums = 25
    # import datetime
    # startday = datetime.datetime.strptime(Day_start, '%b %d, %Y')
    #  = startday.strftime("%Y-%m-%d")
    Day_start = 'Jul 25, 2019'
    start_day = "2019-07-25"

    metric_log_file = FLAGS.csvfile # 'test_metric_day.csv'
    headers =['log','date','all_auc','recall','precision','loss_average','acc_average','f1']

    with open(os.path.join(PATH,metric_log_file), "a") as fo:
        f_csv = csv.writer(fo)
        f_csv.writerow(headers)               


    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        
        #iiter = get_max_model_index(2)

        #day_nums  = iiter  //   (save_iter * train_break_sum)
        #print("day_nums ",day_nums)
        day_nums = (version-1) // train_break_sum 
        
        startday = datetime.datetime.strptime(Day_start, '%b %d, %Y')
        iterday =  startday+ datetime.timedelta(days=day_nums) 

        curentday = start_day

        nowday = get_now_day_fmt()
        if int(''.join(iterday.strftime("%Y-%m-%d").split('-') )) > int(''.join(start_day.split('-'))):    
            #if int(''.join(iterday.strftime("%Y-%m-%d").split('-') )) < int(''.join(nowday.split('-'))):
            model.restore(sess, os.path.join(PATH, model_path) + str(version-1))
            curentday = iterday.strftime("%Y-%m-%d")
        
        # break_cnt = 1
        # restart_cnt = 1
        while int(''.join(curentday.split('-'))) < int(''.join(nowday.split('-'))):
            request = [curentday,]

            # dates = get_some_day_fmt(Day_start,Day_nums)
        
            #for index,date in dates.items():
            #data = '2019-07-23'
            #request = [curentday,]
            pro = Thread(target=produce, args=(filter_str, request))
            pro.setDaemon(True)
            pro.start()

            iiter=0
            
            loss_sum = 0.0
            accuracy_sum = 0.0       

            #break_cnt_save = {"break_cnt":break_cnt,"restart_cnt":restart_cnt} 
                      
            break_dic = np.load(FLAGS.train_cnt_file,allow_pickle=True).item()
            if break_dic['break_cnt'] ==train_break_sum or \
                break_dic['restart_cnt'] == restart_sum:

                break_cnt = 1
                restart_cnt = 1
            else:
                break_cnt = break_dic['break_cnt']
                restart_cnt = break_dic['restart_cnt']


            


       
            #train 一个完整的数据集 1000轮 第一天的数据
            logging.info('########################### TRAIN ###########################')
            while True:
                
                item = MY_QUEUE.get(30)

                if item == "done":
                    
                    time.sleep(10)
                    #logging.info("restart")
                    logging.info("## the day {} train done ## ".format(request[0]))
                    #logging.info("## TRAIN restart ",extra={})
                    if restart_cnt >= restart_sum:
                        break   #整个数据集1000轮后，跳出while
                    restart_cnt += 1
                    #break_dic['break_cnt']=break_cnt 
                    break_dic['restart_cnt'] = restart_cnt           
                    np.save(FLAGS.train_cnt_file,break_dic)

                    #没数据了 上个线程死了，done ，，再开一个，再读一次完整的数据
                    pro = Thread(target=produce, args=(filter_str, request))
                    pro.setDaemon(True)
                    pro.start()
                    continue
                
                try:
                    data = pd.DataFrame.from_dict(item)

                    
                    feature, target = handle(data)
                    
                        

                    user_id, ad_id, mobile, province, city, grade, math, english, \
                        chinese, purchase, activity, freshness, hour, ad_his, mask, length, target, \
                            ad_label,ad_value, \
                            rencent_behavior_a,rencent_behavior_b,  rencent_behavior_c   \
                                 = prepare_data(feature,target)

                    #基类也有 继承类也有 怎么调用，python中，继承类 调用基类的函数
                    loss, acc, = model.train(sess, [user_id, ad_id, mobile, province, city, grade, math, english,
                                                    chinese, purchase, activity, freshness, hour, ad_his, mask, length,
                                                    target, lr,
                                                    ad_label,ad_value,
                                                    rencent_behavior_a,rencent_behavior_b,rencent_behavior_c 
                                                    ])

                    iiter += 1

                    loss_sum += loss
                    accuracy_sum += acc
                    

                    # logging.info("------iter: {},loss:{}, accuracy:{},loss:{},acc:{}".format(iiter,
                    #             loss_sum / iiter, accuracy_sum / iiter,loss,acc))
                except Exception as e:
                    print(e)
                    #logging.info("error in train ")
                    logging.debug("error in train {}".format(e))
                    logging.debug("train iiter{}\n,error {}".format(iiter,e))
                    logging.debug("execept e :{}".format(traceback.format_exc()))
                    continue
                
                if iiter % print_iter == 0:
                    logging.info("---train--- day:{}, iter: {},loss_average:{}, accuracy_average:{},loss:{},acc:{}".format(
                        request[0],iiter,
                                loss_sum / iiter, accuracy_sum / iiter,loss, acc))
                

                if iiter % save_iter == 0:

                    model.save(sess, os.path.join(PATH, model_path)+str(version))
                    # model.save(sess, os.path.join(PATH, best_model_path) + str(version))
                    model.save_serving_model(sess, os.path.join(PATH, paths), version=version)
                    
                    version += 1
                    
                    if break_cnt >= train_break_sum:
                        break
                    break_cnt += 1
                    #break_dic['restart_cnt'] = restart_cnt
                    break_dic['break_cnt']=break_cnt            
                    np.save(FLAGS.train_cnt_file,break_dic)

                if iiter % lr_iter == 0:
                    lr *= lr_decay_rate #0.8
            



            logging.info('########################### TEST ###########################')
            #test  第二天的数据，并保存日志， 训练到最后一天，不再测试
            tmp = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
            testday = tmp.strftime("%Y-%m-%d")
            if int(''.join(testday.split('-'))) == int(''.join(nowday.split('-'))):
                logging.info("****Day_nums-1 == index {}****".format(testday ))
                break
            request = [testday,]


            pro = Thread(target=produce, args=(filter_str, request))
            pro.setDaemon(True)
            pro.start()

            cnt=0 
            
            store_arr = []
            loss_test_sum = 0.0
            accuracy_test_sum = 0.0

            while True:
                
                

                item2 = MY_QUEUE.get(30)

                if item2 == "done" or cnt>=test_max_step:
                # 一天的数据集 读完
                    all_auc, r, p, f1 = calc_auc(store_arr)
                    logging.info("test done !!: date:{},all_auc:{},recall:{},precision:{},loss_average:{},acc_average:{},F1:{}".format(
                        request[0],all_auc, r, p, loss_test_sum / cnt, accuracy_test_sum / cnt,f1))

                    with open(metric_log_file, "a") as fo:
                        #headers =['date','all_auc','recall','precision','loss','acc','f1']
                        f_csv = csv.writer(fo) 
                        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f_csv.writerow([nowTime,request[0],all_auc, r, p, loss_test_sum / cnt, accuracy_test_sum / cnt,f1])
                    break

                try:
                    cnt += 1
                    data = pd.DataFrame.from_dict(item2)

                ## join()  或者直接return
                
                    feature, target = handle(data)

                                 
                    user_id, ad_id, mobile, province, city, grade, math, english, \
                    chinese, purchase, activity, freshness, hour, ad_his, mask, length, target, \
                        ad_label,ad_value,\
                            rencent_behavior_a,rencent_behavior_b, rencent_behavior_c= prepare_data(feature,target)
                                                                      
                    prob, loss2,acc2  = model.test(sess, [user_id, ad_id, mobile, province, city, grade, math, english,\
                        chinese, purchase, activity, freshness, hour, ad_his, mask, length,\
                            target, \
                            ad_label,ad_value,\
                            rencent_behavior_a,rencent_behavior_b,  rencent_behavior_c 
                                            ])

                    #target = inps[16]
                    loss_test_sum += loss2
                    accuracy_test_sum += acc2
     
                    prob_1 = prob[:, 1].tolist()
                    target_1 = target[:, 1].tolist()

                    for p, t in zip(prob_1, target_1):
                        store_arr.append([p, t])
      
                    logging.info(" ---test--- day:{}, iter:{},loss_average:{}, accuracy_average:{} ,loss:{},acc:{}".format( 
                        request[0],cnt,
                    loss_test_sum / cnt,accuracy_test_sum / cnt,loss2,acc2) )
                # 关闭打开的文件
                #fo.close()

                except Exception as e:
                    print(e)
                    logging.debug("test cnt {}\n,error {}".format(cnt,e))
                    logging.debug("execept e :{}".format(traceback.format_exc()))
                    continue
            

            curentday = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
            curentday = curentday.strftime("%Y-%m-%d")
            if int(''.join(nowday.split('-'))) != int(''.join(get_now_day_fmt().split('-'))):
                nowday = get_now_day_fmt()