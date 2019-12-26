import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("mnist", one_hot=True)
import os

export_path = os.path.join("model", "13")


def add_s():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.int32, shape=[None, ])
    xx = tf.placeholder(tf.int32, shape=[None, ])

    sess.run(tf.global_variables_initializer())

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.build_tensor_info(x)
    tensor_info_xx = tf.saved_model.build_tensor_info(xx)
    tensor_info_y = tf.saved_model.build_tensor_info(tf.identity(xx))

    prediction_signature = (
        tf.saved_model.build_signature_def(
            inputs={'x': tensor_info_x, "xx": tensor_info_xx},
            outputs={'outputs': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving": prediction_signature,
        },
        strip_default_attrs=True
    )
    builder.save()


# add_s()


def request():
    import requests, json

    base_url = "http://localhost:8502/v1/models/mnist"
    status_url = base_url + ""
    print(requests.get(status_url).json())

    metadata_url = base_url + "/metadata"
    print(requests.get(metadata_url).json())

    predict_url = base_url + ":predict"

    from data_iter import TrainDataIter
    test_data = TrainDataIter(file_path="test_filter.csv", batch_size=10)
    from train import prepare_data
    tmp = None
    tt = None
    for data, target in test_data:
        tmp = data
        tt = target
        break

    user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, lengths_xx, target = prepare_data(tmp,
                                                                                             tt,
                                                                                             choose_len=0)

    base_ph = ["uid_batch_ph", "mid_batch_ph", "cat_batch_ph",
               "mid_his_batch_ph", "cat_his_batch_ph",
               "mask", "seq_len_ph",
               ]
    data = {}
    data["uid_batch_ph"] = user_ids.tolist()
    # data["uid_batch_ph"] = [[3953]]
    [3953]
    data["mid_batch_ph"] = ad_ids.tolist()
    # data["mid_batch_ph"] = [[267]]
    [267]
    data["cat_batch_ph"] = code_ids.tolist()
    # data["cat_batch_ph"] = [[6]]
    [6]
    data["mid_his_batch_ph"] = ad_his.tolist()
    # data["mid_his_batch_ph"] = [[246]]
    [[246]]
    data["cat_his_batch_ph"] = code_his.tolist()
    # data["cat_his_batch_ph"] = [[7]]
    [[7]]
    data["seq_len_ph"] = lengths_xx.tolist()
    # data["seq_len_ph"] = [[1]]
    [1]
    data["mask"] = ad_mask.tolist()
    # data["mask"] = [[1.0]]
    [[1.0]]

    import pickle
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f, 2)

    dd = {"signature_name": "serving",
          "instances": [{"x": [1, 1, 1], "xx": [2, 2, 2]}
                        ]
          # "inputs": data.copy()
          }

    import time
    begin = time.time()
    resp = requests.post(predict_url,
                         data=json.dumps(dd),
                         ).json()
    print(resp)

    # data = resp["outputs"]
    #
    # print(len(data), )
    # try:
    #     print(len(data[0][0]))
    # except:
    #     pass
    print(time.time() - begin)


# request()


def get_all_feature(request_data) -> dict:
    import random
    ad_list = request_data["ad_list"]
    request_length = len(ad_list)
    user_info = request_data["user_info"]
    tmp = {}
    tmp["uid_batch_ph"] = [user_info["user_id"]] * request_length
    tmp["mid_batch_ph"] = [i["ad_id"] for i in ad_list]

    def get_cat(ad_id):
        """
        map ad_id to cat_id
        :param ad_id:
        :return: cat_id
        """
        return random.randint(1, 10)

    tmp["cat_batch_ph"] = [get_cat(i["ad_id"]) for i in ad_list]

    seq_length = 0

    def get_mid_his(user_id) -> list:
        """
        look up user_id's mid_his
        :param user_id:
        :return: [[mid1,mid2,mid3,...],]
        """
        nonlocal seq_length
        seq_length = random.randint(2, 8)
        mid_his = [random.randint(1, 10) for _ in range(seq_length)]
        # use [mid_his] to expand_dim
        mid_his = [mid_his]
        return mid_his

    tmp["mid_his_batch_ph"] = get_mid_his(user_info["user_id"]) * request_length

    def get_cat_his(user_id):
        """
        look up user_id's cat_his, always related to mid_his
        :param user_id:
        :return: [[cat1,cat2,cat3,...],]
        """
        cat_his = [random.randint(1, 10) for _ in range(seq_length)]
        # use [mid_his] to expand_dim
        cat_his = [cat_his]
        return cat_his

    tmp["cat_his_batch_ph"] = get_cat_his(user_info["user_id"]) * request_length

    tmp["seq_len_ph"] = [seq_length] * request_length

    tmp["mask"] = [[1.0] * seq_length] * request_length

    return tmp


def handler(request):
    # to get the request's post data, it's a dict
    # request_data = request.POST
    request_data = {"user_info": {"user_id": 1,
                                  "province_id": 1, "city_id": 1, "county_id": 1, "school_id": 1, "grade": 1,
                                  "chinese_ability_overall": 1, "english_ability_overall": 1, "math_ability_overall": 1,
                                  "pay_test": 1,
                                  "seatwork_active_degree": 1, "user_freshness": 1},
                    "ad_list": [{"ad_id": 1}, {"ad_id": 3}, {"ad_id": 2}, {"ad_id": 4}]
                    }

    features = get_all_feature(request_data)

    # post_data for serving.api,signature_name's value can change.
    dd = {"signature_name": "serving",
          "inputs": features
          }
    import requests, json

    # serving.api
    predict_url = "http://10.9.24.174:8501/v1/models/midas" + ":predict"
    resp = requests.post(predict_url,
                         data=json.dumps(dd),
                         ).json()

    def handle_resp(resp, ad_id_list) -> list:
        tmp = []
        for i, j in zip(resp["outputs"], ad_id_list):
            tmp.append([j, i[1]])
        return tmp

    return handle_resp(resp, features["mid_batch_ph"])


if __name__ == "__main__":
    # request()
    import requests, json

    """test for load"""
    # base_url = "http://10.19.66.30:8501/v1/models/midas"

    """8501,8502"""
    base_url = "http://10.19.90.95:8502/v1/models/midas"
    base_url = "http://10.19.160.33:8502/v1/models/midas"

    """8503,8504"""
    base_url = "http://10.19.117.187:8504/v1/models/midas"
    base_url = "http://10.19.128.25:8503/v1/models/midas"

    status_url = base_url + ""
    print(requests.get(status_url).json())

    metadata_url = base_url + "/metadata"
    print(requests.get(metadata_url).json())

    predict_url = base_url + ":predict"

    base_ph = ["uid_ph", "mid_ph", "mobile_ph",
               "province_ph", "city_ph", "grade_ph",
               "math_ph", "english_ph", "chinese_ph",
               "purchase_ph", "activity_ph", "freshness_ph",
               "hour_ph",
               ]
    data = {}
    for i in base_ph:
        data[i] = [1] * 300
    ["mid_his_ph", "mask_ph", "seq_len_ph"]
    data["mid_his_ph"] = [[1]] * 300
    data["mask_ph"] = [[1]] * 300
    data["seq_len_ph"] = [1] * 300

    dd = {"signature_name": "serving",
          "inputs": data.copy()
          }

    # print(json.dumps(dd))

    dd = {"inputs": {"activity_ph": [5, 5, 5, 5, 5], "mid_ph": [10215, 10221, 10219, 10220, 10232],
                     "hour_ph": [16, 16, 16, 16, 16], "uid_ph": [3976484, 3976484, 3976484, 3976484, 3976484],
                     "math_ph": [0, 0, 0, 0, 0], "seq_len_ph": [0, 0, 0, 0, 0], "city_ph": [258, 258, 258, 258, 258],
                     "chinese_ph": [0, 0, 0, 0, 0], "mobile_ph": [1, 1, 1, 1, 1], "freshness_ph": [7, 7, 7, 7, 7],
                     "grade_ph": [2, 2, 2, 2, 2], "purchase_ph": [0, 0, 0, 0, 0], "english_ph": [0, 0, 0, 0, 0],
                     "mask_ph": [[0], [0], [0], [0], [0]], "mid_his_ph": [[0], [0], [0], [0], [0]],
                     "province_ph": [19, 19, 19, 19, 19]}, "signature_name": "serving"}


    dd["inputs"]["mid_ph"]=[10,11,12,13,14]
    resp = requests.post(predict_url,
                         data=json.dumps(dd),
                         ).json()
    print(resp)
    pass

    """
    siege -c 1000 -t 20s  -b "http://10.19.90.95:8501/v1/models/midas:predict POST </tmp/ad_kafka/test.json"
    """

    import threading
    import subprocess


    def test_load(port: int):
        status, output = subprocess.getstatusoutput(
            """siege -c 1000 -t 5m  -b "http://10.19.90.95:{}/v1/models/midas:predict POST </tmp/ad_kafka/test.json" """.format(
                port))

        with open("{}.load".format(port), "w") as f:
            f.write(output)


    import requests
    import threading

    ports = [8501, 8502, ]
    # p = []
    # for i in ports:
    #     p.append(threading.Thread(target=test_load, args=(i,)))
    # for i in p:
    #     i.start()
    # for i in p:
    #     i.join()
