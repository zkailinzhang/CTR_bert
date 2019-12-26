import pickle
import numpy as np
import typing, os

from model import *
from data_iter import TrainDataIter
from utils import calc_auc

MODEL_PATH = "save/%s/%s/ckpt_"
BEST_MODEL_PATH = "save/%s/%s/ckpt_best_"

best_auc = -1.0

with open("user.pkl", "rb") as f:
    USER_MAP = pickle.load(f)

#?
with open("code.pkl", "rb") as f:
    CODE_MAP = pickle.load(f)

with open("ad.pkl", "rb") as f:
    AD_MAP = pickle.load(f)


def prepare_data(feature: list, target: list, choose_len: int = 0) -> typing.Tuple:
    user_ids = np.array([fea[0] for fea in feature])
    ad_ids = np.array([fea[1] for fea in feature])
    code_ids = np.array([fea[2] for fea in feature])

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

    return user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, np.array(lengths_xx), np.array(target)


def evalute(sess: tf.Session, test_data: TrainDataIter, model: Model) -> typing.Tuple:
    loss_sum = 0.0
    accuracy_sum = 0.0
    aux_loss_sum = 0.0

    cnt = 0

    store_arr = []

    for feature, target in test_data:
        cnt += 1

        user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, lengths_xx, target = prepare_data(feature,
                                                                                                 target,
                                                                                                 choose_len=0)

        prob, loss, acc, aux_loss = model.calculate(sess, [user_ids, ad_ids, code_ids, ad_his, code_his,
                                                           ad_mask, target, lengths_xx])

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
#



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
                      use_negsampling=False)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=model.grath) as sess:
        train_data = TrainDataIter(file_path="train_filter.csv", batch_size=batch_size)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        lr = 0.001
        iiter = 0

        tag = False

        for epoch in range(4):
            loss_sum = 0.0
            accuracy_sum = 0.0
            aux_loss_sum = 0.0

            for feature, target in train_data:
                user_ids, ad_ids, code_ids, ad_his, code_his, ad_mask, lengths_xx, target = prepare_data(feature,
                                                                                                         target,
                                                                                                         choose_len=0)
                loss, acc, aux_loss = model.train(sess, [user_ids, ad_ids, code_ids, ad_his, code_his,
                                                         ad_mask, target, lengths_xx, lr])
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
                        tag = True
                        break
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if iiter % save_iter == 0:
                    model.save(sess, model_path + "--" + str(iiter))
            if tag == True:
                break


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
                      use_negsampling=False)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=model.grath) as sess:
        
        test_data = TrainDataIter(file_path="test_filter.csv", batch_size=batch_size)

        model.restore(sess, best_model_path)

        print(evalute(sess, test_data, model, ))


def select_model_cls(model_type: str):
    if model_type == "DIN":
        return Model_DIN
    elif model_type == "WideDeep":
        return Model_WideDeep
    elif model_type == "PNN":
        return Model_PNN
    elif model_type == "DNN":
        return Model_DNN
    elif model_type == "DIN_V2_Gru_att_Gru":
        return Model_DIN_V2_Gru_att_Gru
    elif model_type == "DIN_V2_Gru_Gru_att":
        return Model_DIN_V2_Gru_Gru_att
    elif model_type == "DIN_V2_Gru_QA_attGru":
        return Model_DIN_V2_Gru_QA_attGru
    elif model_type == "DIN_V2_Gru_Vec_attGru":
        return Model_DIN_V2_Gru_Vec_attGru
    else:
        raise NotImplementedError()


def test_serving(cnt_user: int = len(USER_MAP) + 100,
                 cnt_ad: int = len(AD_MAP) + 100,
                 cnt_code: int = len(CODE_MAP) + 100,
                 batch_size: int = 128,
                 model_type: str = "DIN",
                 seed: int = 66, ):
    best_model_path = BEST_MODEL_PATH % (model_type, seed) + model_type + "_" + str(seed)

    gpu_options = tf.GPUOptions(allow_growth=True)

    model_cls = select_model_cls(model_type)
    model = model_cls(cnt_user, cnt_ad, cnt_code, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                      use_negsampling=False)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=model.grath) as sess:
        model.restore(sess, best_model_path)
        # model.save_serving_model(sess, version=26)


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

    seed = 33

    for i in model_type:
        print("using --------", i)
        train(model_type=i, seed=seed)
        test(model_type=i, seed=seed)
        best_auc = -1.0
        break

    for i in model_type:
        test_serving(model_type=i, seed=seed)
        # test(model_type=i, seed=seed,)
        break

"DIN"
a66 = (0.7646180111665538, 0.12622549019607843, 0.34105960264900664, 0.18425760286225404, 0.06606934770076935,
       0.9668921183072178, 0.0)
a88 = (0.7265593943983149, 0.20772058823529413, 0.13335955940204564, 0.16243411595591759, 0.09839040367095,
       0.9365494934523978, 0.0)
a11 = (0.7298071658402769, 0.19454656862745098, 0.17502756339581035, 0.18427161926871735, 0.08617978286253382,
       0.9489736965447492, 0.0)
a22 = (0.7422871013891461, 0.09037990196078431, 0.3118393234672304, 0.14014251781472684, 0.07335780001602121,
       0.9671393469512531, 0.0)

"WideDeep"
a66 = (0.7480035667612411, 0.10753676470588236, 0.29745762711864404, 0.15796579657965795, 0.06997463092127865,
       0.9660391856707485, 0.0)
a88 = (0.7689119504629801, 0.14675245098039216, 0.305874840357599, 0.19834368530020702, 0.06902159755815872,
       0.9648595979820145, 0.0)
a11 = (0.7872328219498919, 0.1485906862745098, 0.40182270091135047, 0.21695370163274433, 0.06050825708504641,
       0.9682191233740174, 0.0)
a22 = (0.7450252188039694, 0.14736519607843138, 0.369431643625192, 0.21068769163381518, 0.06816302663749398,
       0.967284526974482, 0.0)

"PNN"
a66 = (0.7598530408078982, 0.11029411764705882, 0.2477632484514797, 0.1526393894424422, 0.07190062482619569,
       0.9637253790505395, 0.0)
a88 = (0.766780650703393, 0.18566176470588236, 0.4488888888888889, 0.26267880364109236, 0.06271074761546776,
       0.9691174247677456, 0.0)
a11 = (0.7527646361288965, 0.12867647058823528, 0.3056768558951965, 0.18111254851228978, 0.06733967827981323,
       0.9655242191928327, 0.0)
a22 = (0.7684279610018236, 0.16881127450980393, 0.31960556844547566, 0.22093023255813957, 0.06788519302460946,
       0.9647257290650744, 0.0)

"DNN"
a66 = (0.7609353265086488, 0.1724877450980392, 0.15085744908896034, 0.16094911377930243, 0.08347992827135822,
       0.9467234061847026, 0.0)
a88 = (0.7300118030633588, 0.16544117647058823, 0.28738690792974986, 0.2099941668286992, 0.08365662819430077,
       0.9631196750581057, 0.0)
a11 = (0.7709617966107942, 0.18167892156862744, 0.4386094674556213, 0.2569324090121317, 0.06305899197380647,
       0.9688633597270953, 0.0)
a22 = (0.7666606687137243, 0.1246936274509804, 0.30950570342205325, 0.17776807163136057, 0.06571430443384783,
       0.9658304893873572, 0.0)

"DIN_V2_Gru_att_Gru"
a66 = (0.7540411000397508, 0.22457107843137256, 0.17324509572205152, 0.19559706470980656, 0.08817746530370257,
       0.9452875161004815, 0.0)
a88 = (0.7684927816588075, 0.15349264705882354, 0.26244106862231537, 0.19369804755461048, 0.06853879215282309,
       0.9621397099013113, 0.0)
a11 = (0.7552199646500122, 0.19791666666666666, 0.1946955997588909, 0.1962929200850805, 0.08013926683324762,
       0.9519861820267468, 0.0)
a22 = (0.736063091563427, 0.19944852941176472, 0.1724503311258278, 0.18496945588862054, 0.08572377776186242,
       0.9479302151277921, 0.0)

"DIN_V2_Gru_Gru_att"
a66 = (0.7610141288222338, 0.23958333333333334, 0.224777234837597, 0.23194423846952397, 0.08074945802221661,
       0.9529933684378966, 0.0)
a88 = (0.7462652581777532, 0.22916666666666666, 0.14498933901918976, 0.17760892793541494, 0.09643653720080783,
       0.9371187781069198, 0.0)
a11 = (0.7292855594745805, 0.3449754901960784, 0.09371618809821057, 0.14739184501603508, 0.146406160689713,
       0.8817688942509384, 0.0)
a22 = (0.7364543314495043, 0.34221813725490197, 0.10148087580630508, 0.15654123747459883, 0.13762842266821454,
       0.8907587445848356, 0.0)

"DIN_V2_Gru_QA_attGru"
a66 = (0.7650814752807735, 0.11488970588235294, 0.2648305084745763, 0.16025641025641027, 0.06716131389359581,
       0.9643264840011951, 0.0)
a88 = (0.7348895496134497, 0.38480392156862747, 0.0971835345094398, 0.1551766740795651, 0.1522576103752758,
       0.8758937023519501, 0.0)
a11 = (0.7376893061273041, 0.23774509803921567, 0.13530950305143855, 0.17246360706745192, 0.10244056613332701,
       0.9324141001452136, 0.0)
a22 = (0.7613334282411983, 0.17647058823529413, 0.21044939715016442, 0.1919680053324446, 0.0761914280666153,
       0.9559877064169907, 0.0)

"DIN_V2_Gru_Vec_attGru"
a66 = (0.7709764174453222, 0.20680147058823528, 0.21293375394321767, 0.2098228162884675, 0.0784098001101793,
       0.9538485384292027, 0.0)
a88 = (0.7673729421111798, 0.18229166666666666, 0.21810850439882698, 0.1985981308411215, 0.07305984191583528,
       0.9564141727352253, 0.0)
a11 = (0.7447832398380445, 0.203125, 0.15785714285714286, 0.17765273311897106, 0.090055034147185,
       0.9442848043990052, 0.0)
a22 = (0.7430606584684206, 0.2104779411764706, 0.15490417136414883, 0.17846473567995844, 0.0921104427575243,
       0.9425948492741335, 0.0)
