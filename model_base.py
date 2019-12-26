import logging
import os, sys
import inspect
import time
from queue import Queue
from threading import Lock, Thread
import tensorflow as tf
import pandas as pd
import pickle
from functools import lru_cache
from typing import List, AnyStr
from abc import ABCMeta

from update_model import UpdateModel
from data_iter import DataGenerator


def _init():
    def init_logger():
        global _logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s : %(message)s')
        _logger = logging.getLogger('model')

    init_logger()

    def init_model_nums_select():
        global _Model_NUMS_SELECT
        _Model_NUMS_SELECT = [0, 1, 2, 3, 4]

    init_model_nums_select()

    def init_base_dir_path():
        global _BASE_DIR_PATH
        _BASE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

    init_base_dir_path()

    def init_lr():
        global _MIN_LR, _INIT_LR
        _MIN_LR = 1 / 10 ** 12
        _INIT_LR = 1 / 10 ** 5

    init_lr()

    def init_trans():
        global _NUM_PORT, _IP_PORT, _PORT_IP, _IP_PWD, \
            _ADDRESS, _STATE
        _NUM_PORT = {1: 8501, 2: 8502, 3: 8503, 4: 8504}
        _IP_PORT = {
            "10.19.90.95": [8501, 8502],
            "10.19.160.33": [8501, 8502],
            "10.19.117.187": [8503, 8504],
            "10.19.128.25": [8503, 8504]
        }
        _PORT_IP = {}
        for key, value in _IP_PORT.items():
            for port in value:
                target = _PORT_IP.get(port, set())
                target.add(key)
                _PORT_IP[port] = target.copy()

        # better to move to config to void leak
        _IP_PWD = {
            "10.19.90.95": "Knowbox.cn",
            "10.19.160.33": "Knowbox.cn",
            "10.19.117.187": "root!@#.com",
            "10.19.128.25": "root!@#.com",
        }

        _TARGET = [
            "ubuntu@10.19.90.95:/data/midas-model",
            "ubuntu@10.19.160.33:/data/midas-model",
            "ubuntu@10.19.117.187:/data/midas-model",
            "ubuntu@10.19.128.25:/data/midas-model",
        ]
        _STATE = """sshpass -p {pwd} scp -r {source} {target}"""

    init_trans()


_init()


def _num2port(model_num: int):
    return _NUM_PORT[model_num]


def _ip2port(ip: str, default=None):
    return _IP_PORT.get(ip, default)


def _port2ip(port: int, default=None):
    return _PORT_IP.get(port, default)


def _ip2pwd(ip: str, default=None):
    return _IP_PWD.get(ip, default)


def _trans_model(model_num: int, source: str,
                 version: int, target: List[AnyStr] = None):
    def parse2list(item):
        if not isinstance(item, (list, tuple)):
            item = [item]
        return item

    @lru_cache(maxsize=20)
    def parse_ip(address):
        ip = address.split(":")[0].split("@")[1]
        return ip

    @lru_cache(maxsize=20)
    def check_dir(ip, pwd, dir):
        import paramiko

        logging.getLogger("paramiko.transport").setLevel(logging.ERROR)
        # logging.getLogger("paramiko").setLevel(logging.DEBUG)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, 22, 'ubuntu', pwd)
        cmd = "mkdir -p %s" % (dir.split(":")[1])
        ssh.exec_command(cmd)
        ssh.close()

    port = _num2port(model_num)
    port_list = parse2list(port)
    target = target or _TARGET.copy()
    target_list = parse2list(target)

    for port in port_list:
        ips = _port2ip(port, set())
        if len(ips) == 0:
            continue
        for target in target_list:
            if parse_ip(target) not in ips:
                continue

            real_source = os.path.join(source, str(version))
            real_target = os.path.join(target, str(port))

            # TODO: use thread???
            pwd = _ip2pwd(parse_ip(target), None)
            if pwd is None:
                continue
            check_dir(parse_ip(target), pwd, real_target)
            os.system(_STATE.format(pwd=pwd,
                                    source=real_source,
                                    target=real_target))


class Model_Exception(Exception):
    pass


class Model_Meta(ABCMeta):
    def __new__(cls, name, bases, attrs):
        if not hasattr(attrs, "MODEL_NUM"):
            if not name.startswith("Model_"):
                raise Model_Exception("the name of this cls must start with Model_, "
                                      "but \"%s\" given!" % (name))
            model_num = name.split("Model_")[1]
            if model_num in ["", "Base"]:
                model_num = 0
        else:
            model_num = getattr(attrs, "MODEL_NUM")

        try:
            model_num = int(model_num)
        except:
            raise Model_Exception("MODEL_NUM must be set and trans to int, "
                                  "but \"%s\" given!" % (name))
        if model_num not in _Model_NUMS_SELECT:
            raise Model_Exception("MODEL_NUM must in %s, but %s given!" % (
                _Model_NUMS_SELECT, model_num))
        if model_num == 0 and "Model_Base" != name:
            _logger.warning("MODEL_NUM set to be 0, nothing will exec!")

        attrs["MODEL_NUM"] = model_num
        attrs["CLASS_NAME"] = name

        return super(Model_Meta, cls).__new__(cls, name, bases, attrs)


class Model_Base(object, metaclass=Model_Meta):
    _QUEUE = None
    _LOCK = Lock()
    _INSTANCE_LOCK = Lock()

    def __new__(cls, *args, **kwargs):

        # just to make sure for Model_Base can't be instantiate
        if cls.CLASS_NAME == "Model_Base":
            raise Model_Exception("Can't instantiate abstract class Model_Base!")

        # Single instance
        if not hasattr(cls, "_INSTANCE"):
            with cls._INSTANCE_LOCK:
                if not hasattr(cls, "_INSTANCE"):
                    cls._INSTANCE = super(Model_Base, cls).__new__(cls, )

        return cls._INSTANCE

    def __init__(self, model, data_iter, handle,
                 prepare_data, train_data_cate, *, base_dir_path=None,
                 save_iter=500, print_iter=100,
                 lr_iter=1000, lr=0.001,
                 restart_sum=1000, break_sum=8):
        assert isinstance(model, UpdateModel), "the model must instance of %s" % (UpdateModel)
        assert isinstance(data_iter, DataGenerator), "the data_iter must instance of %s" % (DataGenerator)
        assert callable(handle), "handle must callable!"
        assert callable(prepare_data), "prepare_data must callable!"

        self.model = model
        self.data_iter = data_iter
        self.handle, self.prepare_data = handle, prepare_data
        self.train_data_cate = train_data_cate
        self.base_dir_path = base_dir_path or _BASE_DIR_PATH

        self.path = os.path.join(self.base_dir_path, self.CLASS_NAME.lower())
        self.model_path = os.path.join(self.path, "model")
        self.model_serving_path = os.path.join(self.path, "serving")
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.model_serving_path, exist_ok=True)

        self.save_iter, self.print_iter = save_iter, print_iter
        self.lr_iter, self.lr = lr_iter, lr
        self.restart_sum, self.break_sum = restart_sum, break_sum

    def produce(self, kwargs):
        def inner():
            assert isinstance(kwargs, dict)
            try:
                with self._LOCK:
                    self._QUEUE = Queue(800000)

                with self.data_iter as d:
                    default = inspect.signature(d.get_data).parameters.get("batch_size").default
                    default = kwargs.get("batch_size", None) or default
                    for i in d.get_data(batch_size=default, model_num=self.MODEL_NUM):
                        self._QUEUE.put(i)
                self._QUEUE.put("done")
            except:
                pass

        p = Thread(target=inner, args=(), )
        p.setDaemon(True)
        return p

    def get_max_model_index(self):
        num = -1

        for i in os.listdir(self.model_path):
            o = i.split(".")[0]
            try:
                a = int(o.split("_")[1])
                if a > num:
                    num = a
            except:
                pass

        return num

    def get_max_serving_index(self):
        num = -1

        for i in os.listdir(self.model_serving_path):
            o = i.split(".")[0]
            try:
                a = int(o)
                if a > num:
                    num = a
            except:
                pass

        return num

    def get_lr(self):
        lr_path = os.path.join(self.path, "lr.index")
        if os.path.exists(lr_path):
            with open(lr_path, "rb") as f:
                return pickle.load(f)

        return None

    def update_lr(self, lr):
        lr_path = os.path.join(self.path, "lr.index")
        if os.path.exists(lr_path):
            os.remove(lr_path)
        with open(lr_path, "wb") as f:
            pickle.dump(lr, f)

    def trans_model(self, version, target=None):
        _trans_model(self.MODEL_NUM, self.model_serving_path, version,
                     target=target)

    def run(self, produce_kwargs=None, trans_target=None):
        if self.MODEL_NUM == 0:
            return

        produce_kwargs = produce_kwargs or {}

        restart_cnt, break_cnt = 1, 1
        loss_sum, accuracy_sum = 0.0, 0.0

        lr = self.get_lr() or self.lr
        if lr < _MIN_LR:
            lr = _INIT_LR
        with tf.Session(graph=self.model.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            version = self.get_max_serving_index()
            if version == -1:
                version = 0
            version += 1
            iiter = self.get_max_model_index()
            if iiter != -1:
                self.model.restore(sess, os.path.join(self.model_path, "ckpt_") + str(iiter))
            if iiter == -1:
                iiter = 0

            self.produce(kwargs=produce_kwargs).start()
            time.sleep(0.5)
            while True:
                try:
                    item = self._QUEUE.get(30)
                    if item == "done":
                        time.sleep(10)
                        print("restart")
                        if restart_cnt > self.restart_sum:
                            break
                        restart_cnt += 1

                        self.produce(kwargs=produce_kwargs).start()
                        time.sleep(0.5)
                        continue
                except:
                    time.sleep(10)
                    continue

                data = pd.DataFrame.from_dict(item)

                try:
                    feature, target = self.handle(data)
                    prepared_data = self.prepare_data(feature, target)
                    if len(self.train_data_cate) != len(prepared_data):
                        _logger.error("train_data_cate'length must equal to "
                                      "the length of prepare_data's returns! ")
                        sys.exit(1)
                    train_data = {k: v for k, v in zip(self.train_data_cate, prepared_data)}
                    train_data["lr"] = lr
                    loss, acc, = self.model.train_with_dict(sess, train_data)
                    iiter += 1
                    loss_sum += loss
                    accuracy_sum += acc
                except Exception as e:
                    _logger.error(e)
                    continue

                if iiter % self.print_iter == 0:
                    print(iiter, loss_sum, accuracy_sum)

                if iiter % self.save_iter == 0:

                    self.model.save(sess, os.path.join(self.model_path, "ckpt_") + str(iiter))
                    self.model.save_serving_model(sess, self.model_serving_path,
                                                  version=version)

                    print("start transport the model! ")
                    self.trans_model(version, target=trans_target)

                    version += 1

                    loss_sum = 0.0
                    accuracy_sum = 0.0

                    if break_cnt >= self.break_sum:
                        break
                    break_cnt += 1

                if iiter % self.lr_iter == 0:
                    lr *= 0.5
                    self.update_lr(lr)


def parse_argv(argv):
    filter_str = """RowFilter (=, 'substring:{}')"""
    import datetime

    if len(argv) == 1:
        sys.exit(1)

    # command:
    # 1、hbase_fliter_str
    # 2、hbase_fliter_str day
    # 3、hbase_fliter_str day,day,day
    # 4、hbase_fliter_str day~day
    cmd = argv[1:]
    assert 1 <= len(cmd) <= 2
    if len(cmd) == 1 or len(cmd[1]) == 0:
        day = datetime.datetime.today()
        yes = day + datetime.timedelta(days=-1)
        if len(cmd) == 1:
            cmd.append(yes.strftime("%Y-%m-%d"))
        else:
            cmd[1] = yes.strftime("%Y-%m-%d")

    if "~" in cmd[1]:
        days = cmd[1].split("~").sort()
        assert len(days) == 2
        begin, end = days[0], days[1]
        days = []
        begin = datetime.datetime.strptime(begin, "%Y-%m-%d")
        end = datetime.datetime.strptime(end, "%Y-%m-%d")
        for i in range((end - begin).days + 1):
            days.append((begin + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
    elif "," in cmd[1]:
        days = cmd[1].split(",").sort()
    else:
        days = [cmd[1]]
    if len(cmd[0]) == 0:
        cmd[0] = filter_str
    return cmd[0], days


if __name__ == "__main__":
    pass
