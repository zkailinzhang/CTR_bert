import pandas as pd
import numpy as np
from typing import List, Tuple

from model_base import Model_Base, UpdateModel
from map2int import TO_MAP, MAP
from data_iter import HbaseDataIter
from config import *


class Model_4(Model_Base):

    def __init__(self, model, data_iter, handle,
                 prepare_data, train_data_cate, *,
                 base_dir_path=None,
                 save_iter=500, print_iter=100,
                 lr_iter=1000, lr=0.001,
                 restart_sum=1000, break_sum=8):
        super(Model_4, self).__init__(model, data_iter, handle, prepare_data,
                                      train_data_cate,
                                      base_dir_path=base_dir_path,
                                      save_iter=save_iter,
                                      print_iter=print_iter,
                                      lr_iter=lr_iter,
                                      lr=lr,
                                      restart_sum=restart_sum,
                                      break_sum=break_sum)


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


if __name__ == '__main__':
    from model_base import parse_argv
    import sys

    argv = sys.argv.copy()
    if len(argv) == 1:
        argv = ["aa.py", "", ""]
    filter_str, days = parse_argv(argv)
    inner_model = UpdateModel()
    data_iter = HbaseDataIter(HBASE_HOST, HBASE_TABLE, filter_str, days,
                              HBASE_FIELD)

    train_data_cate = ["uid_ph", "mid_ph", "mobile_ph", "province_ph",
                       "city_ph", "grade_ph", "math_ph", "english_ph",
                       "chinese_ph", "purchase_ph", "activity_ph",
                       "freshness_ph", "hour_ph", "mid_his_ph", "mask_ph",
                       "seq_len_ph", "target_ph", ]

    model = Model_4(inner_model, data_iter,
                    handle, prepare_data,
                    train_data_cate,
                    )
    model.run({"batch_size": 128}, "ubuntu@10.19.90.95:/data/midas-ll")
