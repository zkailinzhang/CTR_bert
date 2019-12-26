from train import USER_MAP, CODE_MAP, AD_MAP, select_model_cls

import numpy as np
import typing, os

from model import *
from data_iter import TrainDataIter
from utils import calc_auc

MODEL_PATH = "save-portray/%s/%s/ckpt_"
BEST_MODEL_PATH = "save-portray/%s/%s/ckpt_best_"

best_auc = -1.0


def prepare_data(feature: list, target: list, choose_len: int = 0) -> typing.Tuple:
    user_ids = np.array([fea[0] for fea in feature])
    ad_ids = np.array([fea[1] for fea in feature])
    code_ids = np.array([fea[2] for fea in feature])

    province_ids = np.array([fea[6] for fea in feature])
    city_ids = np.array([fea[7] for fea in feature])
    grade_ids = np.array([fea[8] for fea in feature])
    chinese_ids = np.array([fea[9] for fea in feature])
    english_ids = np.array([fea[10] for fea in feature])
    math_ids = np.array([fea[11] for fea in feature])
    pay_ids = np.array([fea[12] for fea in feature])
    seatwork_ids = np.array([fea[13] for fea in feature])
    fresh_ids = np.array([fea[14] for fea in feature])

    seqs_ad = [fea[3] for fea in feature]
    seqs_code = [fea[4] for fea in feature]
    lengths_xx = [fea[5] for fea in feature]

    if choose_len != 0:
        new_seqs_ad = []
        new_seqs_code = []
        new_lengths_xx = []

        for l_xx, fea in zip(lengths_xx, feature):
            if l_xx > choose_len:
                new_seqs_ad.append(fea[3][l_xx - choose_len:])
                new_seqs_code.append(fea[4][l_xx - choose_len:])
                new_lengths_xx.append(l_xx)
            else:
                new_seqs_ad.append(fea[3])
                new_seqs_code.append(fea[4])
                new_lengths_xx.append(l_xx)
        lengths_xx = new_lengths_xx
        seqs_ad = new_seqs_ad
        seqs_code = new_seqs_code

    max_len = np.max(lengths_xx)
    cnt_samples = len(seqs_ad)

    ad_his = np.zeros(shape=(cnt_samples, max_len), ).astype("int64")
    code_his = np.zeros(shape=(cnt_samples, max_len)).astype("int64")
    ad_mask = np.zeros(shape=(cnt_samples, max_len)).astype("float32")

    for idx, [x, y, ] in enumerate(zip(seqs_ad, seqs_code)):
        ad_mask[idx, :lengths_xx[idx]] = 1.0
        ad_his[idx, :lengths_xx[idx]] = x
        code_his[idx, :lengths_xx[idx]] = y

    return user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, np.array(lengths_xx), np.array(target), \
           province_ids, city_ids, grade_ids, chinese_ids, english_ids, math_ids, \
           pay_ids, seatwork_ids, fresh_ids


def evalute(sess: tf.Session, test_data: TrainDataIter, model: Model) -> typing.Tuple:
    loss_sum = 0.0

    accuracy_sum = 0.0
    aux_loss_sum = 0.0

    cnt = 0

    store_arr = []

    for feature, target in test_data:
        cnt += 1

        user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, \
        lengths_xx, target, province_ids, city_ids, grade_ids, \
        chinese_ids, english_ids, math_ids, \
        pay_ids, seatwork_ids, fresh_ids = prepare_data(feature, target, choose_len=0)

        prob, loss, acc, aux_loss = model.calculate(sess, [user_ids, ad_ids, code_ids, ad_his, code_his,
                                                           ad_mask, target, lengths_xx,
                                                           province_ids, city_ids, grade_ids,
                                                           chinese_ids, english_ids, math_ids,
                                                           pay_ids, seatwork_ids, fresh_ids
                                                           ])

        loss_sum += loss
        accuracy_sum += acc
        aux_loss_sum += aux_loss

        prob_1 = prob[:, 1].tolist()
        target_1 = target[:, 1].tolist()

        for p, t in zip(prob_1, target_1):
            store_arr.append([p, t])
    all_auc, r, p, f1 = calc_auc(store_arr)

    return all_auc, r, p, f1, loss_sum / cnt, accuracy_sum / cnt, aux_loss_sum / cnt


def train(cnt_user: int = len(USER_MAP) + 100,
          cnt_ad: int = len(AD_MAP) + 100,
          cnt_code: int = len(CODE_MAP) + 100,
          batch_size: int = 128,
          model_type: str = "DIN",
          print_iter: int = 100,
          save_iter: int = 1000,
          seed: int = 66,
          ) -> None:
    global best_auc

    model_path = MODEL_PATH % (model_type, seed) + model_type + "_" + str(seed)
    best_model_path = BEST_MODEL_PATH % (model_type, seed) + model_type + "_" + str(seed)

    path_prefix = model_path.split("/")[:3]
    best_path_prefix = best_model_path.split("/")[:3]
    if not os.path.exists("/".join(path_prefix)):
        os.makedirs("/".join(path_prefix))
    if not os.path.exists("/".join(best_path_prefix)):
        os.makedirs("/".join(best_path_prefix))

    # to use gpu for future
    gpu_options = tf.GPUOptions(allow_growth=True)

    model_cls = select_model_cls(model_type)
    model = model_cls(cnt_user, cnt_ad, cnt_code, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                      use_negsampling=False, use_others=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=model.grath) as sess:
        train_data = TrainDataIter(file_path="train_portray.csv", batch_size=batch_size, use_others=True)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        lr = 0.001
        iiter = 0

        for epoch in range(4):
            loss_sum = 0.0
            accuracy_sum = 0.0
            aux_loss_sum = 0.0

            for feature, target in train_data:
                user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, \
                lengths_xx, target, province_ids, city_ids, grade_ids, \
                chinese_ids, english_ids, math_ids, \
                pay_ids, seatwork_ids, fresh_ids = prepare_data(feature, target, choose_len=0)

                loss, acc, aux_loss = model.train(sess, [user_ids, ad_ids, code_ids, ad_his, code_his,
                                                         ad_mask, target, lengths_xx, lr,
                                                         province_ids, city_ids, grade_ids,
                                                         chinese_ids, english_ids, math_ids,
                                                         pay_ids, seatwork_ids, fresh_ids
                                                         ])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iiter += 1

                if iiter % print_iter == 0:
                    # print("--------iter:", iiter,
                    #       "--loss:", loss_sum / print_iter,
                    #       "--accuracy:", accuracy_sum / print_iter,
                    #       "--aux_loss:", aux_loss_sum / print_iter)
                    if accuracy_sum / print_iter > best_auc:
                        best_auc = accuracy_sum / print_iter
                        model.save(sess, best_model_path)
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if iiter % save_iter == 0:
                    model.save(sess, model_path + "--" + str(iiter))


def test(cnt_user: int = len(USER_MAP) + 100,
         cnt_ad: int = len(AD_MAP) + 100,
         cnt_code: int = len(CODE_MAP) + 100,
         batch_size: int = 128,
         model_type: str = "DIN",
         seed: int = 66,
         ) -> None:
    best_model_path = BEST_MODEL_PATH % (model_type, seed) + model_type + "_" + str(seed)

    gpu_options = tf.GPUOptions(allow_growth=True)

    model_cls = select_model_cls(model_type)
    model = model_cls(cnt_user, cnt_ad, cnt_code, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                      use_negsampling=False, use_others=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=model.grath) as sess:
        test_data = TrainDataIter(file_path="test_portray.csv", batch_size=batch_size, use_others=True)

        model.restore(sess, best_model_path)

        print(evalute(sess, test_data, model, ))


if __name__ == "__main__":
    model_type = [
        "DIN",
        "WideDeep",
        "PNN",
        "DNN",
        "DIN_V2_Gru_att_Gru",
        "DIN_V2_Gru_Gru_att",
        "DIN_V2_Gru_QA_attGru",
        "DIN_V2_Gru_Vec_attGru",
    ]

    seed = 66
    EMBEDDING_DIM = 18
    HIDDEN_SIZE = 18 * 2
    ATTENTION_SIZE = 18 * 2

    seed = 88
    EMBEDDING_DIM = 30
    HIDDEN_SIZE = 30 * 2
    ATTENTION_SIZE = 30 * 2

    seed = 11
    EMBEDDING_DIM = 64
    HIDDEN_SIZE = 64 * 2
    ATTENTION_SIZE = 64 * 2

    seed = 22
    EMBEDDING_DIM = 64
    HIDDEN_SIZE = 64 * 2
    ATTENTION_SIZE = 18 * 2

    for i in model_type:
        print("using --------", i)
        train(model_type=i, seed=seed)
        test(model_type=i, seed=seed)
        best_auc = -1.0

"DIN"
a66 = (0.7043090505493997, 0.24872097599370327, 0.09988936304725779, 0.14253495714930087, 0.12468207560491301,
       0.9069991286185239, 0.0)
a88 = (0.7181499692429094, 0.09169618260527351, 0.1495507060333761, 0.11368626494266894, 0.07899692451875767,
       0.9555713384177577, 0.0)
a11 = (0.7208449964099735, 0.1345926800472255, 0.1607898448519041, 0.14652956298200512, 0.08462010171550075,
       0.951279965178321, 0.0)
a22 = (0.6788075965721165, 0.13970877607241244, 0.11334610472541506, 0.125154239379517, 0.10126975464673948,
       0.9392994646921591, 0.0)

"WideDeep"
a66 = (0.7256153517926587, 0.13852813852813853, 0.3003412969283277, 0.18960409372475087, 0.07318255426114387,
       0.9632004463989783, 0.0)
a88 = (0.7331859336178363, 0.12632821723730814, 0.3651877133105802, 0.18771929824561404, 0.06970029673794159,
       0.9660246834881802, 0.0)
a11 = (0.714959967909809, 0.12711530893349074, 0.24903623747108714, 0.16831683168316833, 0.08027318661015172,
       0.9609575063596496, 0.0)
a22 = (0.7245625690773461, 0.13262495080676898, 0.2897678417884781, 0.1819654427645788, 0.07282283457348791,
       0.9629270255285809, 0.0)

"PNN"
a66 = (0.700263548678228, 0.14049586776859505, 0.27503852080123264, 0.18598593383693668, 0.08405539832123102,
       0.9617822147870847, 0.0)
a88 = (0.7199505094397496, 0.05785123966942149, 0.24058919803600654, 0.0932741116751269, 0.07273573981064586,
       0.9650465927213571, 0.0)
a11 = (0.7138335993596235, 0.0881542699724518, 0.17418351477449456, 0.1170629736085707, 0.08725951339740522,
       0.9586712191922004, 0.0)
a22 = (0.7199441549543327, 0.0747737111373475, 0.2074235807860262, 0.10992189759907434, 0.08435651007019485,
       0.9623690692471786, 0.0)

"DNN"
a66 = (0.71767361138379, 0.05194805194805195, 0.1875, 0.08135593220338982, 0.07857975334290514,
       0.9635372207571457, 0.0)
a88 = (0.7299052256082234, 0.11530893349075168, 0.2644404332129964, 0.1605919429980817, 0.07702007992061931,
       0.9625402351313727, 0.0)
a11 = (0.6876672694683946, 0.12750885478158205, 0.19103773584905662, 0.15293839981118718, 0.0870501779847675,
       0.9561092883395104, 0.0)
a22 = (0.7213780577413892, 0.09799291617473435, 0.18906605922551253, 0.1290824261275272, 0.08013069362077914,
       0.9589090731595417, 0.0)

"DIN_V2_Gru_att_Gru"
a66 = (0.7439657988382626, 0.19323101141282958, 0.2176418439716312, 0.20471127788200957, 0.08219529842850001,
       0.9533406245130142, 0.0)
a88 = (0.7248556305889612, 0.1778827233372688, 0.17546583850931677, 0.1766660152433066, 0.08980402091287765,
       0.9484801803582897, 0.0)
a11 = (0.7125316154898473, 0.14246359700905156, 0.14353687549563837, 0.14299822239778784, 0.09143845350168214,
       0.9469396874005432, 0.0)
a22 = (0.7206468497446359, 0.14325068870523416, 0.14113997673516868, 0.14218750000000002, 0.09000950312921092,
       0.9462917022675229, 0.0)

"DIN_V2_Gru_Gru_att"
a66 = (0.737730180612341, 0.19008264462809918, 0.14587737843551796, 0.1650717703349282, 0.09428787938900676,
       0.9402464343721691, 0.0)
a88 = (0.7357114664392022, 0.10940574576938213, 0.26127819548872183, 0.15423023578363385, 0.07349431998328144,
       0.962705843605346, 0.0)
a11 = (0.7154171629462157, 0.1782762691853601, 0.1193047142480906, 0.14294730198800884, 0.10104840176532713,
       0.9335598502547156, 0.0)
a22 = (
0.707020448341402, 0.05588351042896497, 0.1459403905447071, 0.08081957882754695, 0.0768834928594273, 0.9604929132454085,
0.0)

"DIN_V2_Gru_QA_attGru"
a66 = (0.7386238527775736, 0.33569460842188115, 0.1390609716335181, 0.196657060518732, 0.11832175700574041,
       0.9147793960310111, 0.0)
a88 = (0.716453391208748, 0.1633215269578906, 0.1319974554707379, 0.14599824098504835, 0.0947984808236827,
       0.9406310019545339, 0.0)
a11 = (0.7144237069780758, 0.048406139315230225, 0.1554993678887484, 0.07382953181272509, 0.07645301030531922,
       0.9622590340359111, 0.0)
a22 = (0.7238876670871166, 0.14010232192050373, 0.16079494128274616, 0.14973711882229232, 0.085721667529505,
       0.9505419511936827, 0.0)

"DIN_V2_Gru_Vec_attGru"
a66 = (0.7470121957578971, 0.12829594647776466, 0.2056782334384858, 0.1580222976248182, 0.07580672012254339,
       0.9575097364065979, 0.0)
a88 = (0.7203504197097559, 0.2388823297914207, 0.14715151515151514, 0.1821182118211821, 0.10483008286010631,
       0.9333264421968953, 0.0)
a11 = (0.704316970056658, 0.2066115702479339, 0.13092269326683292, 0.16028087314913755, 0.10742968788694048,
       0.932727361602216, 0.0)
a22 = (0.7119629610910723, 0.08225108225108226, 0.16163959783449341, 0.1090245174752217, 0.08098100893146976,
       0.9582244096227654, 0.0)
