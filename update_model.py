import os
import pandas as pd
import numpy as np
from queue import Queue
from typing import List, Tuple
from threading import Thread

from data_iter import HbaseDataIterUpdate, FIELD
from map2int import TO_MAP, MAP
from transport_model import trans_model
from utils import *
from Dice import dice

AD_BOUND = 10000
USER_BOUND = 10000000
USER_SUM = 10000
AD_SUM = 100000
CITY_SUM = 5000

EMBEDDING_DIM = 128
ATTENTION_SIZE = 128
ABILITY_DIM = 5


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
                self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
                self.lr = tf.placeholder(tf.float64, [])

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

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

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
                 ], -1)

        self.build_fcn_net(inp, use_dice=True)

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
            }
        )
        return loss, accuracy

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
    data["rclick_ad"] = data["rclick_ad"].map(lambda x: parse_his(x))

    to_select = ["user_id", "ad_id", "mobile_os",
                 "province_id", "city_id", "grade_id",
                 "math_ability", "english_ability", "chinese_ability",
                 "purchase_power", "activity_degree", "app_freshness",
                 "log_hourtime",
                 "rclick_ad"]
    feature, target = [], []
    for row in data.itertuples(index=False):
        tmp = []
        for i in to_select:
            tmp.append(getattr(row, i))

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

    return user_id, ad_id, mobile, province, city, grade, math, english, \
           chinese, purchase, activity, freshness, hour, \
           ad_his, ad_mask, np.array(lengths_xx), np.array(target)


Field = FIELD + []
MY_QUEUE = Queue(800000)


def produce(filter_str, request):
    try:
        with HbaseDataIterUpdate("10.9.75.202", b'midas_offline_v1', filter_str, request, ) as d:
            for i in d.get_data(batch_size=128, model_num=0):
                MY_QUEUE.put(i)
    except:
        pass
    finally:
        MY_QUEUE.put("done")


if __name__ == "__main__":
    from look_up_dir import get_max_serving_index, get_max_model_index, get_last_day_fmt

    save_iter = 500
    print_iter = 500
    lr_iter = 1000
    lr = 0.001
    version = get_max_serving_index(2) + 1

    loss_sum = 0.0
    accuracy_sum = 0.0

    restart_sum = 0
    restart_cnt = 1

    break_sum = 15
    break_cnt = 1

    import os, time

    PATH = os.path.dirname(os.path.abspath(__file__))

    filter_str = """RowFilter (=, 'substring:{}')"""
    request = [get_last_day_fmt()]

    model_path = "update-model-1/model/ckpt_"
    best_model_path = "update-model-1/best-model/ckpt_"

    model = UpdateModel()

    pro = Thread(target=produce, args=(filter_str, request))
    pro.setDaemon(True)

    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        iiter = get_max_model_index(2)

        model.restore(sess, os.path.join(PATH, model_path) + str(iiter))

        pro.start()
        while True:
            item = MY_QUEUE.get()
            if item == "done":
                time.sleep(5)
                print("restart")
                if restart_cnt > restart_sum:
                    break
                restart_cnt += 1

                pro = Thread(target=produce, args=(filter_str, request))
                pro.setDaemon(True)
                pro.start()

            try:
                data = pd.DataFrame.from_dict(item)
                feature, target = handle(data)

                user_id, ad_id, mobile, province, city, grade, math, english, \
                chinese, purchase, activity, freshness, hour, ad_his, mask, length, target = prepare_data(feature,
                                                                                                          target)

                loss, acc, = model.train(sess, [user_id, ad_id, mobile, province, city, grade, math, english,
                                                chinese, purchase, activity, freshness, hour, ad_his, mask, length,
                                                target, lr])

                iiter += 1
                loss_sum += loss
                accuracy_sum += acc
            except Exception as e:
                print("model train error: ", e.__class__.__name__)
                continue

            if iiter % print_iter == 0:
                print(iiter, loss_sum, accuracy_sum)
                # model.save(sess, model_path + str(iiter))
            if iiter % save_iter == 0:

                model.save(sess, os.path.join(PATH, model_path) + str(iiter))
                model.save(sess, os.path.join(PATH, best_model_path) + str(version))
                model.save_serving_model(sess, os.path.join(PATH, "update-model-1", "serving"), version=version)

                print("start transport the model! ")

                """trans model !!!"""
                trans_model(version, port=[8502, 8503, 8504])

                version += 1

                loss_sum = 0.0
                accuracy_sum = 0.0

                if break_cnt >= break_sum:
                    break
                break_cnt += 1

            if iiter % lr_iter == 0:
                lr *= 0.5
