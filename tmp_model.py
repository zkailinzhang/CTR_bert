import pandas as pd
import numpy as np
from typing import Tuple, List
from queue import Queue
from threading import Thread
import tensorflow as tf
import os
import time

from data_iter import DataIter, FIELD
from utils import *
from Dice import dice

user_purchase_power = {"A1": 1, "A3": 2, "A2": 3, "A4": 4, "B": 5}
user_math_ability = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, }
user_english_ability = user_math_ability.copy()
user_chinese_ability = user_math_ability.copy()
user_activity_degree = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, }
user_app_freshness = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

AD_BOUND = 10000
USER_BOUND = 10000000

MY_QUEUE = Queue(1000000)


def handle(data: pd.DataFrame) -> Tuple[List, List]:
    data = data.drop(columns=["user_school_id", "user_county_id"], )
    to_int = ["user_mobile_os", "user_province_id",
              "user_grade_id", "user_city_id",
              "ad_ad_id", "user_user_id", "context_log_hourtime",

              "user_purchase_power", "user_math_ability",
              "user_english_ability", "user_chinese_ability",
              "user_activity_degree", "user_app_freshness"]
    for i in to_int:
        data[i] = data[i].astype(int)
    # data["user_purchase_power"] = data["user_purchase_power"].map(lambda x: user_purchase_power.get(x, 0))
    # data["user_math_ability"] = data["user_math_ability"].map(lambda x: user_math_ability.get(x, 0))
    # data["user_english_ability"] = data["user_english_ability"].map(lambda x: user_english_ability.get(x, 0))
    # data["user_chinese_ability"] = data["user_chinese_ability"].map(lambda x: user_chinese_ability.get(x, 0))
    # data["user_activity_degree"] = data["user_activity_degree"].map(lambda x: user_activity_degree.get(x, 0))
    # data["user_app_freshness"] = data["user_app_freshness"].map(lambda x: user_app_freshness.get(x, 0))

    data["ad_ad_id"] = data["ad_ad_id"].map(lambda x: abs(x) if x < AD_BOUND else x - 90000)
    data["user_user_id"] = data["user_user_id"].map(lambda x: abs(x) % 6 if x < USER_BOUND else x - USER_BOUND)

    to_select = ["user_user_id", "ad_ad_id", "user_mobile_os",
                 "user_province_id", "user_city_id", "user_grade_id",
                 "user_math_ability", "user_english_ability", "user_chinese_ability",
                 "user_purchase_power", "user_activity_degree", "user_app_freshness",
                 "context_log_hourtime"]
    feature, target = [], []
    for row in data.itertuples(index=False):
        tmp = []
        for i in to_select:
            tmp.append(getattr(row, i))

        if getattr(row, "context_is_click") == "0":
            target.append([1, 0])
        else:
            target.append([0, 1])
        feature.append(tmp)

    return feature, target


def prepare_data(feature: List, target: List) -> Tuple:
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

    return user_id, ad_id, mobile, province, city, grade, math, english, \
           chinese, purchase, activity, freshness, hour, np.array(target)


USER_SUM = 10000
AD_SUM = 100000
CITY_SUM = 5000
EMBEDDING_DIM = 128
ABILITY_DIM = 5


# DNN
class TmpModel(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.tensor_info = {}

        self.build_inputs()

        with self.graph.as_default():
            inp = tf.concat([
                # self.uid_embedded,
                self.mid_embedded,
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
                self.hour_embedded], 1)
        self.build_fcn_net(inp, use_dice=False)

    def build_inputs(self):
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
                self.target_ph: inps[13],
                self.lr: inps[14],
            }
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
                self.target_ph: inps[13],
            }
        )
        return probs, loss, accuracy

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

    def build_tensor_info(self):
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


def produce():
    with DataIter('10.9.135.235', b'midas_ctr_pro', filter_str, request, ) as d:
        for i in d.get_data(batch_size=128):
            MY_QUEUE.put(i)
    MY_QUEUE.put("done")


if __name__ == "__main__":
    from look_up_dir import get_max_serving_index, get_max_model_index, get_last_day_fmt

    save_iter = 500
    print_iter = 500
    lr_iter = 1000
    best_auc = -1.0
    version = get_max_serving_index() + 1
    PATH = os.path.dirname(os.path.abspath(__file__))
    filter_str = """RowFilter (=, 'substring:{}')"""
    data = None
    request = [get_last_day_fmt()]
    # request = ["2019-04-%02d" % (i) for i in range(18, 19)]

    model_path = "tmp-model-3/model/ckpt_"
    best_model_path = "tmp-model-3/best-model/ckpt_"

    model = TmpModel()

    pro = Thread(target=produce, )
    pro.setDaemon(True)

    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        iiter = get_max_model_index()

        model.restore(sess, os.path.join(PATH, model_path) + str(iiter))

        lr = 0.001
        loss_sum = 0.0
        accuracy_sum = 0.0

        break_sum = 1
        break_cnt = 1

        pro.start()
        while True:
            try:
                item = MY_QUEUE.get(30)
                if item == "done":
                    time.sleep(10)
                    print("restart")

                    pro = Thread(target=produce, )
                    pro.setDaemon(True)
                    pro.start()
                    continue
            except:
                time.sleep(10)
                continue
            data = pd.DataFrame.from_dict(item)

            data = data[FIELD]
            try:
                feature, target = handle(data)

                user_id, ad_id, mobile, province, city, grade, math, english, \
                chinese, purchase, activity, freshness, hour, target = prepare_data(feature, target)

                loss, acc, = model.train(sess, [user_id, ad_id, mobile, province, city, grade, math, english,
                                                chinese, purchase, activity, freshness, hour, target, lr])
                iiter += 1
                loss_sum += loss
                accuracy_sum += acc
                print(iiter, loss_sum, accuracy_sum)
            except:
                continue

            if iiter % save_iter == 0:
                pass
                # model.save(sess, model_path + str(iiter))
            if iiter % print_iter == 0:
                model.save(sess, os.path.join(PATH, model_path) + str(iiter))
                model.save(sess, os.path.join(PATH, best_model_path) + str(version))
                model.save_serving_model(sess, os.path.join(PATH, "tmp-model-3", "serving"), version=version)

                address = ["ubuntu@10.19.90.95:/data/midas-model",
                           "ubuntu@10.19.160.33:/data/midas-model"]
                print("start transport the model! ")
                for addre in address:
                    os.system(
                        """sshpass -p {pwd} scp -r /data/lishuang/ad_kafka/tmp-model-3/serving/{version} {address}""".format(
                            pwd="Knowbox.cn", version=version, address=addre
                        ))

                version += 1

                loss_sum = 0.0
                accuracy_sum = 0.0
                if break_cnt >= break_sum:
                    break
                break_cnt += 1

            if iiter % lr_iter == 0:
                lr *= 0.5

        # model.save(sess, best_model_path + str(version))
        # model.save_serving_model(sess, os.path.join(PATH, "tmp-model", "serving"), version=version)
