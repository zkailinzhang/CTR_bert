import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import os

EMBEDDING_DIM = 18
DEFAULT_EM = 5000


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        self.grath = tf.Graph()
        self.use_others = use_others
        self.use_negsampling = use_negsampling
        self.tensor_info = {}

        with self.grath.as_default():
            with tf.name_scope('Main_Inputs'):
                self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
                self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
                self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
                self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
                self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
                self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
                self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
                self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
                self.lr = tf.placeholder(tf.float64, [])
                if use_negsampling:
                    self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                             name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                    self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

            # Embedding layer
            with tf.name_scope('Main_Embedding_layer'):
                self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
                tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
                self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

                self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
                tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
                self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
                self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
                if self.use_negsampling:
                    self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                               self.noclk_mid_batch_ph)

                self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
                tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
                self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
                self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
                if self.use_negsampling:
                    self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                               self.noclk_cat_batch_ph)

            self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
            self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
            if self.use_negsampling:
                self.noclk_item_his_eb = tf.concat(
                    [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]],
                    -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
                self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                    [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                                     36])  # cat embedding 18 concate item embedding 18.

                self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded],
                                              -1)
                self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
                self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def other_inputs(self):
        """
        if use this method,must to rewrite train and test methods.
        :return: list of Var
        """

        with self.grath.as_default():
            with tf.name_scope("Portray_Inputs"):
                # To Choose: province,city,grade,
                self.province_ph = tf.placeholder(tf.int32, shape=[None, ], name="province_ph")
                self.city_ph = tf.placeholder(tf.int32, shape=[None, ], name="city_ph")
                self.grade_ph = tf.placeholder(tf.int32, shape=[None, ], name="grade_ph")
                self.chinese_ability_overall_ph = tf.placeholder(tf.int32, shape=[None, ],
                                                                 name="chinese_ability_overall_ph")
                self.english_ability_overall_ph = tf.placeholder(tf.int32, shape=[None, ],
                                                                 name="english_ability_overall_ph")
                self.math_ability_overall_ph = tf.placeholder(tf.int32, shape=[None, ],
                                                              name="math_ability_overall_ph")
                self.pay_test_ph = tf.placeholder(tf.int32, shape=[None, ],
                                                  name="pay_test_ph")
                self.seatwork_active_degree_ph = tf.placeholder(tf.int32, shape=[None, ],
                                                                name="seatwork_active_degree_ph")
                self.user_freshness_ph = tf.placeholder(tf.int32, shape=[None, ],
                                                        name="user_freshness_ph")

            with tf.name_scope("Portray_Embedding_layer"):
                self.province_embeddings_var = tf.get_variable("province_embedding_var", [40, EMBEDDING_DIM])
                tf.summary.histogram("province_embedding_var", self.province_embeddings_var)
                self.province_embedded = tf.nn.embedding_lookup(self.province_embeddings_var, self.province_ph)

                self.city_embeddings_var = tf.get_variable("city_embedding_var", [DEFAULT_EM, EMBEDDING_DIM])
                tf.summary.histogram("city_embedding_var", self.city_embeddings_var)
                self.city_embedded = tf.nn.embedding_lookup(self.city_embeddings_var, self.city_ph)

                self.grade_embeddings_var = tf.get_variable("grade_embedding_var", [102, EMBEDDING_DIM])
                tf.summary.histogram("grade_embedding_var", self.grade_embeddings_var)
                self.grade_embedded = tf.nn.embedding_lookup(self.grade_embeddings_var, self.grade_ph)

                self.chinese_ability_overall_embeddings_var = tf.get_variable("chinese_ability_overall_embedding_var",
                                                                              [10, EMBEDDING_DIM])
                tf.summary.histogram("chinese_ability_overall_embedding_var",
                                     self.chinese_ability_overall_embeddings_var)
                self.chinese_ability_overall_embedded = tf.nn.embedding_lookup(
                    self.chinese_ability_overall_embeddings_var, self.chinese_ability_overall_ph)

                self.english_ability_overall_embeddings_var = tf.get_variable("english_ability_overall_embedding_var",
                                                                              [10, EMBEDDING_DIM])
                tf.summary.histogram("english_ability_overall_embedding_var",
                                     self.english_ability_overall_embeddings_var)
                self.english_ability_overall_embedded = tf.nn.embedding_lookup(
                    self.english_ability_overall_embeddings_var, self.english_ability_overall_ph)

                self.math_ability_overall_embeddings_var = tf.get_variable("math_ability_overall_embedding_var",
                                                                           [10, EMBEDDING_DIM])
                tf.summary.histogram("math_ability_overall_embedding_var",
                                     self.math_ability_overall_embeddings_var)
                self.math_ability_overall_embedded = tf.nn.embedding_lookup(
                    self.math_ability_overall_embeddings_var, self.math_ability_overall_ph)

                self.pay_test_embeddings_var = tf.get_variable("pay_test_embedding_var",
                                                               [10, EMBEDDING_DIM])
                tf.summary.histogram("pay_test_embedding_var",
                                     self.pay_test_embeddings_var)
                self.pay_test_embedded = tf.nn.embedding_lookup(
                    self.pay_test_embeddings_var, self.pay_test_ph)

                self.seatwork_active_degree_embeddings_var = tf.get_variable("seatwork_active_degree_embedding_var",
                                                                             [10, EMBEDDING_DIM])
                tf.summary.histogram("seatwork_active_degree_embedding_var",
                                     self.seatwork_active_degree_embeddings_var)
                self.seatwork_active_degree_embedded = tf.nn.embedding_lookup(
                    self.seatwork_active_degree_embeddings_var, self.seatwork_active_degree_ph)

                self.user_freshness_embeddings_var = tf.get_variable("user_freshness_embedding_var",
                                                                     [10, EMBEDDING_DIM])
                tf.summary.histogram("user_freshness_embedding_var",
                                     self.user_freshness_embeddings_var)
                self.user_freshness_embedded = tf.nn.embedding_lookup(
                    self.user_freshness_embeddings_var, self.user_freshness_ph)

        return self.province_embedded, self.city_embedded, self.grade_embedded, \
               self.chinese_ability_overall_embedded, self.english_ability_overall_embedded, \
               self.math_ability_overall_embedded, self.pay_test_embedded, \
               self.seatwork_active_degree_embedded, self.user_freshness_embedded

    def build_fcn_net(self, inp, use_dice=False):
        with self.grath.as_default():
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
                if self.use_negsampling:
                    self.loss += self.aux_loss
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

            self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        with self.grath.as_default():
            mask = tf.cast(mask, tf.float32)
            click_input_ = tf.concat([h_states, click_seq], -1)
            noclick_input_ = tf.concat([h_states, noclick_seq], -1)
            click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
            noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
            click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
            noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
            loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
            return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        with self.grath.as_default():
            bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.nn.sigmoid(dnn1)
            dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
            dnn2 = tf.nn.sigmoid(dnn2)
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
            return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            if self.use_others:
                loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                       feed_dict={
                                                           self.uid_batch_ph: inps[0],
                                                           self.mid_batch_ph: inps[1],
                                                           self.cat_batch_ph: inps[2],
                                                           self.mid_his_batch_ph: inps[3],
                                                           self.cat_his_batch_ph: inps[4],
                                                           self.mask: inps[5],
                                                           self.target_ph: inps[6],
                                                           self.seq_len_ph: inps[7],
                                                           self.lr: inps[8],
                                                           self.noclk_mid_batch_ph: inps[9],
                                                           self.noclk_cat_batch_ph: inps[10],
                                                           self.province_ph: inps[11],
                                                           self.city_ph: inps[12],
                                                           self.grade_ph: inps[13],
                                                           self.chinese_ability_overall_ph: inps[14],
                                                           self.english_ability_overall_ph: inps[15],
                                                           self.math_ability_overall_ph: inps[16],
                                                           self.pay_test_ph: inps[17],
                                                           self.seatwork_active_degree_ph: inps[18],
                                                           self.user_freshness_ph: inps[19]

                                                       })
            else:
                loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                       feed_dict={
                                                           self.uid_batch_ph: inps[0],
                                                           self.mid_batch_ph: inps[1],
                                                           self.cat_batch_ph: inps[2],
                                                           self.mid_his_batch_ph: inps[3],
                                                           self.cat_his_batch_ph: inps[4],
                                                           self.mask: inps[5],
                                                           self.target_ph: inps[6],
                                                           self.seq_len_ph: inps[7],
                                                           self.lr: inps[8],
                                                           self.noclk_mid_batch_ph: inps[9],
                                                           self.noclk_cat_batch_ph: inps[10],
                                                       })
            return loss, accuracy, aux_loss
        else:
            if self.use_others:
                loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.lr: inps[8],
                    self.province_ph: inps[9],
                    self.city_ph: inps[10],
                    self.grade_ph: inps[11],
                    self.chinese_ability_overall_ph: inps[12],
                    self.english_ability_overall_ph: inps[13],
                    self.math_ability_overall_ph: inps[14],
                    self.pay_test_ph: inps[15],
                    self.seatwork_active_degree_ph: inps[16],
                    self.user_freshness_ph: inps[17]
                })
            else:
                loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.lr: inps[8],
                })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            if self.use_others:
                probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss],
                                                           feed_dict={
                                                               self.uid_batch_ph: inps[0],
                                                               self.mid_batch_ph: inps[1],
                                                               self.cat_batch_ph: inps[2],
                                                               self.mid_his_batch_ph: inps[3],
                                                               self.cat_his_batch_ph: inps[4],
                                                               self.mask: inps[5],
                                                               self.target_ph: inps[6],
                                                               self.seq_len_ph: inps[7],
                                                               self.noclk_mid_batch_ph: inps[8],
                                                               self.noclk_cat_batch_ph: inps[9],
                                                               self.province_ph: inps[10],
                                                               self.city_ph: inps[11],
                                                               self.grade_ph: inps[12],
                                                               self.chinese_ability_overall_ph: inps[13],
                                                               self.english_ability_overall_ph: inps[14],
                                                               self.math_ability_overall_ph: inps[15],
                                                               self.pay_test_ph: inps[16],
                                                               self.seatwork_active_degree_ph: inps[17],
                                                               self.user_freshness_ph: inps[18]
                                                           })
            else:
                probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss],
                                                           feed_dict={
                                                               self.uid_batch_ph: inps[0],
                                                               self.mid_batch_ph: inps[1],
                                                               self.cat_batch_ph: inps[2],
                                                               self.mid_his_batch_ph: inps[3],
                                                               self.cat_his_batch_ph: inps[4],
                                                               self.mask: inps[5],
                                                               self.target_ph: inps[6],
                                                               self.seq_len_ph: inps[7],
                                                               self.noclk_mid_batch_ph: inps[8],
                                                               self.noclk_cat_batch_ph: inps[9],
                                                           })
            return probs, loss, accuracy, aux_loss
        else:
            if self.use_others:
                probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.province_ph: inps[8],
                    self.city_ph: inps[9],
                    self.grade_ph: inps[10],
                    self.chinese_ability_overall_ph: inps[11],
                    self.english_ability_overall_ph: inps[12],
                    self.math_ability_overall_ph: inps[13],
                    self.pay_test_ph: inps[14],
                    self.seatwork_active_degree_ph: inps[15],
                    self.user_freshness_ph: inps[16]
                })
            else:
                probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7]
                })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

    def build_tensor_info(self):
        """
        Only build the base tensor_info,better to add other tensor_info!
        :return:
        """
        if len(self.tensor_info) > 0:
            print("will clear items in tensor_info")
            self.tensor_info.clear()

        base_ph = ["uid_batch_ph", "mid_batch_ph", "cat_batch_ph",
                   "mid_his_batch_ph", "cat_his_batch_ph",
                   "mask", "seq_len_ph",
                   ]
        if self.use_negsampling:
            base_ph += ["noclk_mid_batch_ph", "noclk_cat_batch_ph"]

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
                outputs={'outputs': tf.saved_model.build_tensor_info(
                    self.y_hat)},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        )

        export_path = os.path.join(dir_path, str(version))

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "serving": prediction_signature,
            },
            strip_default_attrs=True
        )
        builder.save()


class Model_DIN_V2_Gru_att_Gru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_DIN_V2_Gru_att_Gru, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling, use_others)
        with self.grath.as_default():
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
                rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                             sequence_length=self.seq_len_ph, dtype=tf.float32,
                                             scope="gru1")
                tf.summary.histogram('GRU_outputs', rnn_outputs)

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
                tf.summary.histogram('alpha_outputs', alphas)

            with tf.name_scope('rnn_2'):
                rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=att_outputs,
                                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                         scope="gru2")
                tf.summary.histogram('GRU2_Final_State', final_state2)

            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 final_state2], 1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_DIN_V2_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling, use_others)
        with self.grath.as_default():
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
                rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                             sequence_length=self.seq_len_ph, dtype=tf.float32,
                                             scope="gru1")
                tf.summary.histogram('GRU_outputs', rnn_outputs)

            with tf.name_scope('rnn_2'):
                rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                              sequence_length=self.seq_len_ph, dtype=tf.float32,
                                              scope="gru2")
                tf.summary.histogram('GRU2_outputs', rnn_outputs2)

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs2, ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
                att_fea = tf.reduce_sum(att_outputs, 1)
                tf.summary.histogram('att_fea', att_fea)

            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 att_fea],
                1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                             ATTENTION_SIZE,
                                             use_negsampling, use_others)
        with self.grath.as_default():
            other_inputs = []
            if self.use_others:
                other_inputs = list(self.other_inputs())

            inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
            if self.use_others:
                inp = tf.concat([inp] + other_inputs, 1)
            # Fully connected layer
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
            dnn1 = prelu(dnn1, 'p1')
            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
            dnn2 = prelu(dnn2, 'p2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
            d_layer_wide = tf.concat([tf.concat([self.item_eb, self.item_his_eb_sum], axis=-1),
                                      self.item_eb * self.item_his_eb_sum], axis=-1)
            if self.use_others:
                d_layer_wide = tf.concat([d_layer_wide] + other_inputs, 1)
            d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
            self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()


class Model_DIN_V2_Gru_QA_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_DIN_V2_Gru_QA_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                         EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                         use_negsampling, use_others)
        with self.grath.as_default():
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
                rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                             sequence_length=self.seq_len_ph, dtype=tf.float32,
                                             scope="gru1")
                tf.summary.histogram('GRU_outputs', rnn_outputs)

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
                tf.summary.histogram('alpha_outputs', alphas)

            with tf.name_scope('rnn_2'):
                rnn_outputs2, final_state2 = dynamic_rnn(QAAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                         att_scores=tf.expand_dims(alphas, -1),
                                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                         scope="gru2")
                tf.summary.histogram('GRU2_Final_State', final_state2)

            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 final_state2], 1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling, use_others)
        with self.grath.as_default():
            inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        self.build_fcn_net(inp, use_dice=False)


class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling, use_others)
        with self.grath.as_default():
            inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                             self.item_eb * self.item_his_eb_sum], 1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling, use_others)

        with self.grath.as_default():
            # Attention layer
            with tf.name_scope('Attention_layer'):
                # print(self.item_eb.shape, self.item_his_eb.shape, self.mask.shape)
                attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
                att_fea = tf.reduce_sum(attention_output, 1)
                tf.summary.histogram('att_fea', att_fea)
            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 att_fea],
                -1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True,
                 use_others=False):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(n_uid, n_mid, n_cat,
                                                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                              use_negsampling, use_others)
        with self.grath.as_default():
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
                rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                             sequence_length=self.seq_len_ph, dtype=tf.float32,
                                             scope="gru1")
                tf.summary.histogram('GRU_outputs', rnn_outputs)

            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
                tf.summary.histogram('alpha_outputs', alphas)

            with tf.name_scope('rnn_2'):
                rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                         att_scores=tf.expand_dims(alphas, -1),
                                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                         scope="gru2")
                tf.summary.histogram('GRU2_Final_State', final_state2)

            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 final_state2], 1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_others=False):
        super(Model_DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling, use_others)
        with self.grath.as_default():
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
                rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                             sequence_length=self.seq_len_ph, dtype=tf.float32,
                                             scope="gru1")
                tf.summary.histogram('GRU_outputs', rnn_outputs)

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
                tf.summary.histogram('alpha_outputs', alphas)

            with tf.name_scope('rnn_2'):
                rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                         att_scores=tf.expand_dims(alphas, -1),
                                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                         scope="gru2")
                tf.summary.histogram('GRU2_Final_State', final_state2)

            # inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
            inp = tf.concat(
                [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
                 final_state2], 1)
            if self.use_others:
                inp = tf.concat([inp] + list(self.other_inputs()), 1)
        self.build_fcn_net(inp, use_dice=True)
