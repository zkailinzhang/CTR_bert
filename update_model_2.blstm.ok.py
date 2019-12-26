#-*- coding:utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()


from keras import backend as K
import keras

import os
import pandas as pd
import numpy as np
from queue import Queue  
from typing import List, Tuple
from threading import Thread

from data_iter import DataIter, FIELD
from map2int import TO_MAP, MAP
from transport_model import trans_model
from utils import *
from Dice import dice

import csv

import logging

logging.basicConfig(filename='logblstm.out',filemode='w',
format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y %H:%M:%S",
level=logging.DEBUG)

#内存不足
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if tf.executing_eagerly():
    print("Eager执行方式")
else:
    print("Graphs执行方式")
AD_BOUND = 10000
USER_BOUND = 10000000
USER_SUM = 10000
AD_SUM = 100000

CITY_SUM = 5000


EMBEDDING_DIM = 128
ATTENTION_SIZE = 128
ABILITY_DIM = 5

#ad img fea
AD_IMG_VALUE_DIM = 40
AD_IMG_LABEL_DIM = 20

#rnn
HIDDEN_DIM = 256//2
NUM_LAYERS = 2
KEEP_PROB = 0.9


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
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1')
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
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
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

    def train(self, sess, inps):
        pass

    def train_with_dict(self, sess, train_data):
        pass

    def calculate(self, sess, inps):
        pass

    def save(self, sess, path,step):
        
        saver = tf.train.Saver(max_to_keep=3,keep_checkpoint_every_n_hours=1)
         
        saver.save(sess, save_path=path,global_step=step)
        #saver.save(sess, save_path=path)

    def restore(self, sess, path):
        lastesd = tf.train.latest_checkpoint(path)
        saver = tf.train.Saver()
        saver.restore(sess, save_path=lastesd)
        print('model restored from %s' % lastesd)

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

        with self.graph.as_default():
            with tf.name_scope('Attention_layer'):
                attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask_ph)
                att_fea = tf.reduce_sum(attention_output, 1)

        #     inp = tf.concat(
        #         [self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
        #          att_fea,
        #          self.mobile_embedded,
        #          self.province_embedded,
        #          self.city_embedded,
        #          self.grade_embedded,
        #          self.chinese_embedded,
        #          self.math_embedded,
        #          self.english_embedded,
        #          self.purchase_embedded,
        #          self.activity_embedded,
        #          self.freshness_embedded,
        #          self.hour_embedded
                 
        #          ], -1)

        # self.build_fcn_net(inp, use_dice=True)

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
            with tf.name_scope('Attention_layer'):
                attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask_ph)
                att_fea = tf.reduce_sum(attention_output, 1)
                

            inp = tf.concat(
                [self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 att_fea,
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

                 self.ad_img_eb_sum
                 ], -1)
          
       
        self.build_fcn_net(inp, use_dice=True)

    def build_inputs(self):
        
        super(UpdateModel2, self).build_inputs()
         
       
        with self.graph.as_default():
            #tf.enable_eager_execution()
            with tf.name_scope('Inputs'):
                
                #img AD 特征  N*F
                self.ad_label_ph = tf.placeholder(tf.int32,[None,None],name='ad_label_ph')
                #特征下的类别  N*F
                self.ad_value_ph = tf.placeholder(tf.int32,[None,None],name='ad_value_ph')

            with tf.name_scope("Embedding_layer"):
                #test
                #不是保证可以扩展吗，设置的初始长度，20 40  实际 可能7  9， 但是增加那么多，也学不到啥啊
                self.ad_img_embeddings_var2  = tf.get_variable("ad_img_embedding_var2", [AD_IMG_LABEL_DIM,AD_IMG_VALUE_DIM,EMBEDDING_DIM])
                self.ad_img_embedded2 = tf.nn.embedding_lookup(self.ad_img_embeddings_var2, self.ad_label_ph)
                

                #n*7 ->  n*7*8
                self.ad_value_ph_ohot = tf.one_hot(self.ad_value_ph,depth=AD_IMG_VALUE_DIM,axis=-1)
                
                    #n*7*8  n*7*1 *8 
                self.ad_value_ph_ohot = tf.expand_dims(self.ad_value_ph_ohot,axis=-2)
                #n*7*8*128  就是对应相乘，
                self.ad_img_embedded = tf.matmul(self.ad_value_ph_ohot ,self.ad_img_embedded2)

                print('self.ad_img_embedded {}'.format(self.ad_img_embedded.get_shape().as_list()))

                self.ad_img_eb = self.ad_img_embedded     # none*n*1*128      

                self.ad_img_eb = tf.squeeze(self.ad_img_eb,[-2])  #n*n*128

                #self.ad_img_eb_sum = tf.reduce_mean(self.adimg_eb,-2)  #


                # N*E    <- N*F*E F*1 相乘      n*40 
                 #self.adimg_embedded = tf.multiply(self.ad_value_ph_ohot,self.adimg_embedded2)
  

                #基类的成员变量，成员函数，成员函数内的变量
            # self.item_eb = self.mid_embedded
            # self.item_his_eb = self.mid_his_embedded          
              #  self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1) # N*F*E  -> N*E 
          
                ##RNN   self.adimg_eb  n*n*128
                                ##RNN   self.adimg_eb  n*n*128
                #两个  另一种阿里的 attentionRNN   还有一个双向的
                with tf.name_scope('cell'):

        #WARNING:tensorflow:At least two cells provided to MultiRNNCell are the same 
       # object and will share weights.
          
                    # cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
                    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=KEEP_PROB)
                    # cells = [cell for _ in range(NUM_LAYERS)]


                    def build_cell(n,m):

                        #cell = tf.nn.rnn_cell.GRUCell(n)
                        cell = tf.nn.rnn_cell.LSTMCell(n)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=m)
                        return cell
                    #128可以变得   time*128 -> time* 256 -> time*128
                    num_units=[HIDDEN_DIM*2,HIDDEN_DIM//2]
                    
                    cell_fw = [build_cell(n,KEEP_PROB) for n in num_units]
                    cell_bw = [build_cell(n,KEEP_PROB) for n in num_units]

                    #Cell_stacked = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
                    




                with tf.name_scope('rnn'):
                    #hidden一层 输入是[batch_size, seq_length, hidden_dim]
                    #hidden二层 输入是[batch_size, seq_length, 2*hidden_dim]
                    #2*hidden_dim = embendding_dim + hidden_dim
                    #rnnoutput, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.adimg_eb, sequence_length=tf.shape(self.adimg_eb)[-2], dtype=tf.float32)
                    
                    #rnnoutput, _ = tf.nn.dynamic_rnn(cell=Cell_stacked, inputs=self.ad_img_eb, dtype=tf.float32)
                    
                     #output:[batch_size, seq_length, hidden_dim]

                     #原先的rnn 输出不是结果还有个 要自定义个w b 才是最终的输出

                     #blstm 的输出 两个，怎么个拼接 送到下一层，
                    #batch*time*128  不能用MultiRNNCell
                    # (output_fw,output_bw),_ = tf.contrib.rnn.stack_bidirectional_rnn(
                    #     cell_fw,cell_bw,inputs= self.ad_img_eb,dtype=tf.float32
                    # )
                    #时间步长可不定 biout已经把双向的输出拼接再一起了，最后一维度拼接的，若128 则最后一维度变成256
                    #output_fw output_bw 是
                    biout,output_fw,output_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        cell_fw,cell_bw,inputs= self.ad_img_eb,dtype=tf.float32
                    )
                    #要自己写for循环，不能用MultiRNNCell
                    # (output_fw,output_bw),_ = tf.nn.bidirectional_dynamic_rnn(
                    #     cell_fw,cell_bw,inputs= self.ad_img_eb,dtype=tf.float32
                    # )
                    #biout = tf.transpose(biout,[1,0,2]) #time*b *max -> batch *

                    #bilstm_out = tf.concat([output_fw,output_bw],axis=-1) 
                    rnnoutput = tf.reduce_sum(biout, axis=-2)
   
                    #bW = tf.get_variable(name ="bW",shape=[None,2*HIDDEN_DIM,EMBEDDING_DIM],dtype=tf.float32)
                    #bB = tf.get_variable(name="bB",shape=[None,None,EMBEDDING_DIM],dtype=tf.float32)
                    #b*t*2H  b*2H*HId   -> b*t*128
                    #ad_img_out = tf.matmul(bilstm_out,bW)+bB
                    #rnnoutput = tf.reduce_sum(ad_img_out, axis=-2)

                self.ad_img_eb_sum = rnnoutput

            

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
                self.ad_value_ph: inps[19]

            }
        )
        return loss, accuracy


    def test(self, sess, inps):
        


        prob, loss, acc = self.calculate(sess, inps)

        return prob, loss, acc 
        
        # store_arr = []
        # target = inps[16]

        # prob_1 = prob[:, 1].tolist()
        # target_1 = target[:, 1].tolist()

        # for p, t in zip(prob_1, target_1):
        #     store_arr.append([p, t])
        
        
        # all_auc, r, p, f1 = calc_auc(store_arr)

        # return all_auc, r, p, f1, loss, acc



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
                self.ad_value_ph: inps[18]

            }
        )
        return probs, loss, accuracy

    def build_tensor_info(self):
        super(UpdateModel, self).build_tensor_info()
        add_ph = ["mid_his_ph", "mask_ph", "seq_len_ph"]

        for i in add_ph:
            self.tensor_info[i] = tf.saved_model.build_tensor_info(getattr(self, i))


def parse_his(x):
    x = eval(x)
    if len(x) == 0:
        return []
    return [abs(i) if i < AD_BOUND else i - 90000 for i in x]


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

    to_select = ["user_id", "ad_id", "mobile_os",
                 "province_id", "city_id", "grade_id",
                 "math_ability", "english_ability", "chinese_ability",
                 "purchase_power", "activity_degree", "app_freshness",
                 "log_hourtime", 
                 "rclick_ad",
                 "label_1","label_2","label_3","label_4","label_5","label_6","label_7"
                 ]
    #真的做成可自由扩展的，自适应扩展，那就检索 字符串匹配，"label_*  看有多少

    feature, target = [], []
    for row in data.itertuples(index=False):
        tmp = []
        #若索引字符  不在呢，就是HBASE元数据没有这个列，  没有label ，返回-1, 
        #getattr 是拿到的k v  还是kv一起拿
        
        #若 i 没有to_select里面，赋值为-1，，
        
        #没有这个label_1呢,
        for i in to_select:
            tmp.append(getattr(row, i, -1))

        #其他不用转吧，因为喂入嵌入函数，就是索引值就可以了，不用提前转one-hot，
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

    
    #or 直接 传两个
    

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
    
    #怎么传入
    #ad_img ->  label mat, value,mat
    
    #label_1 = np.array([fea[14] for fea in feature])
    #label_2 = 
    #一个样本的
    label_list = []
    value_list = []
    #一个批次的  这是value
    label_all_tmp = []
    value_all_tmp = []
    for fea in feature:
        # value_list = [fea[14],fea[15],fea[16],fea[17],fea[18],fea[19],fea[20]]
        value_list = [fea[i] for i in range(14,21)]
        
        value_list = np.asarray(value_list,dtype=int)
        # -1 的就是没有value值的 ，去掉
        value_list = value_list[np.where(value_list>-1)]
        label_list = np.where(value_list>-1)[0]
        
        #ufunc 'add' did not contain a loop with signature matching types dtype('<U3') dtype('<U3') dtype('<U3')
        #value_list = (np.array(value_list)+1).tolist()
        #.astype(int)

        label_all_tmp.append(label_list)
        value_all_tmp.append(value_list.tolist())
    
    #print(label_all)
    #print(np.shape(label_all))

    
    # 一个批次   label    
    # [1,2]        [1,2,0]
    # [3,5,1]      [3,5,1]
    # ...   第二个维度不同，没法喂入 placeholder，，像rnn
    #mask   所以   这边是补 -1， 嵌入矩阵 0 表示第一行的数据，，-1才是全0
    label_len = [len(i) for i in label_all_tmp]
    label_len_max = np.max(label_len) #直接返回最大数
    
    # tf.padding  在周围，图像，
    #矩阵对齐
    #  -1 要做one-hot 
    label_all = keras.preprocessing.sequence.pad_sequences(label_all_tmp,
    maxlen=label_len_max,padding='post',value=-1)
    
    value_all = keras.preprocessing.sequence.pad_sequences(value_all_tmp,
    maxlen=label_len_max,padding='post',value=-1)


    return user_id, ad_id, mobile, province, city, grade, math, english, \
           chinese, purchase, activity, freshness, hour, \
           ad_his, ad_mask, np.array(lengths_xx), np.array(target), \
               label_all,value_all


Field = FIELD + []

# 最大队列，，文件量 400万，
MY_QUEUE = Queue(800000)


def produce(filter_str, request,train_mode):
    #hbase(host), 就是ip
    # table, 注意 这是midas offline
    with DataIter("10.9.75.202", b'midas_offline', filter_str, request, train_mode,) as d:
        for i in d.get_data(batch_size=128,train_mode=train_mode):
            MY_QUEUE.put(i)  #一直取，i 是一个批次，执行yeild下面的程序 data=[],队列的数据的单位是一个批次数据
    MY_QUEUE.put("done")

'''
这是一个样本的，  128个  转 pandas

(b'10000022_101475_2019-06-19 18:42:27', 
{b'context:count_click': b'0', b'context:log_time': b'2019-06-19 18:42:27', 
b'context:log_day': b'2019/06/19', b'ad:exposure_duration': b'5', b'ad:test_timestamp': 
b'1560911463778', b'user:school_id': b'54085', b'user:activity_degree': b'E',
 b'context:week_accuracy': b'0.0', b'context:exposure_duration': b'0', b'context:exe_time': 
 b'2019-06-19 18:42:40', b'ad:label_6': b'-1', b'context:hexposure_alocation': b'1',
  b'ad:count_click': b'350', b'context:dexposure_alocation': b'1', b'ad:location_ad': b'6', 
  b'context:is_click': b'0', b'user:county_id': b'1558', b'context:hclick_similarad': b'0', 
  b'context:log_month': b'2019/06/01', b'ad:ad_id': b'101475', b'ad:label_3': b'4', 
  b'context:log_week': b'2019/06/17', b'context:window_otherad': b'0', b'user:grade_id': b'4', 
  b'user:english_ability': b'0', b'context:yeaterday_accuracy': b'0.0', b'ad:label_4': b'-1',
   b'ad:alldexposure_clocation': b'407573', b'user:mobile_os': b'1', b'context:location_ad': b'6', 
   b'context:hexposure_similarad': b'9', b'user:chinese_ability': b'0', b'ad:allhexposure_alocation': 
   b'63035', b'ad:label_5': b'-1', b'ad:label_2': b'3', b'context:hexposure_clocation': b'1',
    b'ad:alldexposure_alocation': b'4324', b'context:dclick_otherad': b'0', b'user:user_id': b'10000022', 
    b'user:purchase_power': b'B', b'user:app_freshness': b'G', b'user:math_ability': b'E', 
    b'context:log_hourtime': b'18', b'user:app_type': b'3', b'ad:label_1': b'1',
     b'context:log_weektime': b'4', b'context:rclick_ad': b'[]', b'user:province_id': b'13', 
     b'ad:window_otherad': b'0', b'user:mobile_type': b'OPPO_R11s;7.1.1', b'user:test_timestamp':
      b'1560940904729', b'context:duplicate_tag': b'0', b'ad:label_7': b'-1', b'context:rclick_category': 
      b'[]', b'context:month_accuracy': b'0.0', b'user:city_id': b'181', b'ad:allhexposure_clocation': 
      b'9418410', b'context:dexposure_clocation': b'1'})
'''
#tf.enable_eager_execution() 
#同样的一个批次，训练多少次
#所有的样本 训练多少轮

if __name__ == "__main__":
    from look_up_dir import get_max_serving_index, get_max_model_index, get_last_day_fmt,get_some_day_fmt

    save_iter = 2000  #48000  
    print_iter = 100  
    lr_iter = 1000
    lr = 0.001
    version = get_max_serving_index(2) + 1



    #整个数据集的训练轮数
    restart_sum = 1
    #restart_cnt = 1

    break_sum = 12
    #:break_cnt = 1:

    import os, time

    PATH = os.path.dirname(os.path.abspath(__file__))

    filter_str = """RowFilter (=, 'substring:{}')"""
    #request = [get_last_day_fmt()]  #'2019-07-11'
    #提前建目录  去掉   "update-model-1/model/ckpt_"
    model_path = "update-model-1/modelblstm/ckpt_"
    best_model_path = "update-model-1/best-model/ckpt_"
    

    model = UpdateModel2()


    MODE = {"test":False,"train":True,"serve":True}
    MODE_TREAIN = True

    #6月2号没有数据 从第二天开始训练的吗
    Day_start = 'Jun 20, 2019'    # 缩写 01  1 都可以  jun jul
    Day_nums = 10
    
    #csv  execel  方便可视，
    metric_log_file = 'test_metric_day.blstm.csv'
    headers =['date','all_auc','recall','precision','loss_average','acc_average','f1']

    with open(metric_log_file, "a") as fo:
        f_csv = csv.writer(fo)
        f_csv.writerow(headers)               

    

    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #iiter = get_max_model_index(2)

        #model.restore(sess, os.path.join(PATH, model_path) + str(iiter))
        
        
        
########## 10天的 每天在线训练 更新
        dates = get_some_day_fmt(Day_start,Day_nums)
        
        for index,date in dates.items():
            
            request = [date,]
            pro = Thread(target=produce, args=(filter_str, request,MODE_TREAIN))
            pro.setDaemon(True)
            pro.start()

            iiter=0
            loss_sum = 0.0
            accuracy_sum = 0.0
            break_cnt = 1
            restart_cnt = 1
##############################         
            #train 一个完整的数据集 1000轮 第一天的数据
            logging.info('########################### TRAIN ###########################')
            while True:
                try:
                    item = MY_QUEUE.get(30)

                    if item == "done":
                        time.sleep(10)
                        #logging.info("restart")
                        logging.info("## the day {} train done ## ".format(request[0]))
                        #logging.info("## TRAIN restart ",extra={})
                        if restart_cnt >= restart_sum:
                            break   #整个数据集1000轮后，跳出while
                        restart_cnt += 1
                        #没数据了 上个线程死了，done ，，再开一个，再读一次完整的数据
                        pro = Thread(target=produce, args=(filter_str, request,MODE_TREAIN))
                        pro.setDaemon(True)
                        pro.start()
                        continue
                except:
                    time.sleep(10)
                    continue
                
                data = pd.DataFrame.from_dict(item)

                try:
                    feature, target = handle(data)

                    user_id, ad_id, mobile, province, city, grade, math, english, \
                        chinese, purchase, activity, freshness, hour, ad_his, mask, length, target, \
                            ad_label,ad_value = prepare_data(feature,target)

                    #基类也有 继承类也有 怎么调用，python中，继承类 调用基类的函数
                    loss, acc, = model.train(sess, [user_id, ad_id, mobile, province, city, grade, math, english,
                                                    chinese, purchase, activity, freshness, hour, ad_his, mask, length,
                                                    target, lr,
                                                    ad_label,ad_value
                                                    
                                                    ])

                    iiter += 1
                    loss_sum += loss
                    accuracy_sum += acc
                    

                    # logging.info("------iter: {},loss:{}, accuracy:{},loss:{},acc:{}".format(iiter,
                    #             loss_sum / iiter, accuracy_sum / iiter,loss,acc))
                except Exception as e:
                    print(e)
                    continue
                
                if iiter % print_iter == 0:
                    logging.info("---train--- day:{}, iter: {},loss_average:{}, accuracy_average:{},loss{},acc{}".format(request[0],iiter,
                                loss_sum / iiter, accuracy_sum / iiter,loss, acc))

                if iiter % save_iter == 0:
                    # logging.info(" --------iter: %f ,loss: %f, accuracy: %f,", iiter,
                    #             loss_sum / iiter, accuracy_sum / iiter)
                                #"--aux_loss:", aux_loss_sum / print_iter)
                    

                    model.save(sess, os.path.join(PATH, model_path) ,iiter)
                    #model.save(sess, os.path.join(PATH, best_model_path) + str(version))
                    #model.save_serving_model(sess, os.path.join(PATH, "update-model-1", "serving"), version=version)

                    #print("\nstart transport the model! ")

                    #"""trans model !!!"""
                    #trans_model(version, port=[8502, 8503, 8504])

                    version += 1

                    # loss_sum = 0.0
                    # accuracy_sum = 0.0
                    
                    #8 上线8 什么意思
                    if break_cnt >= break_sum:
                        break
                    break_cnt += 1

                if iiter % lr_iter == 0:
                    lr *= 0.5
            


######################################################
            logging.info('########################### TEST ###########################')
            #test  第二天的数据，并保存日志， 训练到最后一天，不再测试
            if index == Day_nums-1:
                break

            MODE_TREAIN = True
            request = [dates[index+1],]

            pro = Thread(target=produce, args=(filter_str, request,MODE_TREAIN))
            pro.setDaemon(True)
            pro.start()

            cnt=0 
            
            store_arr = []
            loss_test_sum = 0.0
            accuracy_test_sum = 0.0

            while True:
                
                try:

                    item = MY_QUEUE.get(30)

                    if item == "done":
                    # 一天的数据集 读完
                        all_auc, r, p, f1 = calc_auc(store_arr)
                        logging.info("test done !!: date:{},all_auc:{},recall:{},precision:{},loss_average:{},acc_average:{},F1:{}".format(
                            request[0],all_auc, r, p, loss_test_sum / cnt, accuracy_test_sum / cnt,f1))

                        with open(metric_log_file, "a") as fo:
                            #headers =['date','all_auc','recall','precision','loss','acc','f1']
                            f_csv = csv.writer(fo) 
                            f_csv.writerow([request[0],all_auc, r, p, loss_test_sum / cnt, accuracy_test_sum / cnt,f1])
                        break


                    cnt += 1
                    data = pd.DataFrame.from_dict(item)

                ## join()  或者直接return
                
                    feature, target = handle(data)

                  
                
                    user_id, ad_id, mobile, province, city, grade, math, english, \
                    chinese, purchase, activity, freshness, hour, ad_his, mask, length, target, \
                        ad_label,ad_value = prepare_data(feature,target)
                
                    
                    prob, loss2,acc2  = model.test(sess, [user_id, ad_id, mobile, province, city, grade, math, english,\
                        chinese, purchase, activity, freshness, hour, ad_his, mask, length,\
                            target, 
                            ad_label,ad_value    
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
                    #continue