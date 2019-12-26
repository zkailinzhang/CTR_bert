import pandas as pd
import happybase
import os, pickle
import logging
from abc import ABCMeta, abstractmethod
from deprecated import deprecated
from queue import Queue

PATH = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(PATH, "index.pkl")


class DataGenerator(object, metaclass=ABCMeta):

    @abstractmethod
    def get_data(self, batch_size=128):
        pass

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TrainDataIter(object):

    def __init__(self, file_path: str, batch_size: int = 128, use_others=False):
        """

        :param file_path:
        :param batch_size:
        :param use_others: if True, remember to rewrite __next__ method!
        """
        self.data = pd.read_csv(file_path)
        self.batch_size = batch_size
        self.use_others = use_others

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.data):
            self._index += self.batch_size
            feature, target = [], []
            for row in self.data[self._index - self.batch_size:self._index].itertuples(index=False):
                tmp = []
                tmp.append(getattr(row, "user_id"))
                tmp.append(getattr(row, "ad_id"))
                tmp.append(getattr(row, "action_code"))
                tmp.append([int(i) for i in getattr(row, "ad_his").split(",")])
                tmp.append([int(i) for i in getattr(row, "code_his").split(",")])
                tmp.append(getattr(row, "seq_length"))
                if self.use_others:
                    tmp.append(getattr(row, "province"))
                    tmp.append(getattr(row, "city"))
                    tmp.append(getattr(row, "grade"))
                    tmp.append(getattr(row, "chinese_ability_overall"))
                    tmp.append(getattr(row, "english_ability_overall"))
                    tmp.append(getattr(row, "math_ability_overall"))
                    tmp.append(getattr(row, "pay_test"))
                    tmp.append(getattr(row, "seatwork_active_degree"))
                    tmp.append(getattr(row, "user_freshness"))

                feature.append(tmp)
                if getattr(row, "event") == 0:
                    target.append([1, 0])
                else:
                    target.append([0, 1])
            return feature, target
        else:
            raise StopIteration()


FIELD = ["mobile_os",
         "province_id", "grade_id", "school_id", "city_id", "county_id",
         "purchase_power",
         "math_ability", "english_ability", "chinese_ability",
         "activity_degree", "app_freshness", "ad_id", "user_id",
         "log_hourtime",
         ##########CLICK##########
         "is_click",
         "label_1","label_2","label_3","label_4","label_5","label_6","label_7",
         "rencent_behavior"
         ]


@deprecated(version="1.0.0", reason="This cls is deprecated, please use "
                                    "HbaseDataIter to instead!")
class DataIter(DataGenerator):

    def __init__(self, hbase: str, table: str, filter: str, request: list = None, batch_size: int = 128,
                 field=FIELD):
        self.connection = happybase.Connection(hbase, autoconnect=False, timeout=10 * 1000,
                                               # transport="framed",
                                               # protocol="compact"
                                               )
        self.table = happybase.Table(table, self.connection)
        self.filter = filter
        self.request = request or []

        assert isinstance(self.request, list), "request must be list!"

        self.field = field.copy()

    def get_data(self, batch_size=128):

        index = 0
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, "rb") as f:
                index = pickle.load(f)

        for req in self.request[index:]:
            data = []
            self.connection.open()

            print("consuming %s's data!" % (req))

            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)

            with open(INDEX_PATH, "wb") as f:
                pickle.dump(self.request.index(req), f)

            try:
                for key, item in self.table.scan(filter=self.filter.format(req), batch_size=1, ):

                    tmp = {}
                    tag = False
                    try:
                        for k, v in item.items():
                            k = k.decode("utf8").split(":")[1]
                            v = v.decode("utf8")
                            if k in self.field:
                                if len(v) == 0:
                                    tag = True
                                    break
                            tmp[k] = v
                        if tag:
                            continue
                    except:
                        continue
                    ad_id = tmp["ad_id"]
                    try:
                        ad_id = int(ad_id)
                    except:
                        print("parse error")
                        continue
                    if ad_id < 10000:
                        continue
                    data.append(tmp)
                    if len(data) == batch_size:
                        yield data
                        data = []

            except Exception as e:
                logging.info(e)
            finally:
                if len(data) > 0:
                    yield data

        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)

        return

    def __enter__(self):
        self.connection.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        return True


@deprecated(version="1.1.0", reason="This cls is deprecated, please use "
                                    "HbaseDataIterUpdate to instead!")
class HbaseDataIter(DataGenerator):

    def __init__(self, host: str, table: str, filter: str, request: list = None,
                 field=FIELD):
        self.connection = happybase.Connection(host, autoconnect=False, timeout=30 * 1000)
        self.table = happybase.Table(table, self.connection)
        self.filter = filter
        self.request = request or []

        assert isinstance(self.request, list), "request must be list!"

        self.field = field.copy()

    def get_data(self, batch_size=128, model_num=0):
        try:
            for req in self.request[:]:
                data = []

                self.connection.open()

                print("consuming %s's data!" % (req))

                try:
                    for key, item in self.table.scan(filter=self.filter.format(req), batch_size=1, ):

                        tmp = {}
                        tag = False
                        try:
                            for k, v in item.items():
                                k = k.decode("utf8").split(":")[1]
                                v = v.decode("utf8")
                                if k in self.field:
                                    if len(v) == 0:
                                        tag = True
                                        break
                                tmp[k] = v
                            if tag:
                                continue
                            if 0 != model_num:
                                if int(tmp["user_id"]) % 4 != model_num - 1:
                                    continue
                        except Exception as e:
                            print("parse data error: ", e)
                            continue
                        ad_id = tmp["ad_id"]
                        try:
                            ad_id = int(ad_id)
                        except:
                            print("parse int error: ", "can't int")
                            continue
                        if ad_id < 10000:
                            continue
                        data.append(tmp)
                        if len(data) == batch_size:
                            yield data
                            data = []

                except Exception as e:
                    print("hbase conn error: ", e)
                finally:
                    if len(data) > 0:
                        yield data
        except:
            ## this only omit the GeneratorExit Error
            ## we can do something here,like release resources ... ,but no need!!!
            pass
        finally:
            return

    def __enter__(self):
        self.connection.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        return True


class HbaseDataIterUpdate(DataGenerator):

    def __init__(self, host: str, table: str, filter: str, request: list = None,
                 field=FIELD):
        self.connection = happybase.Connection(host, autoconnect=False, timeout=30 * 1000)
        self.table = happybase.Table(table, self.connection)
        self.filter = filter
        self.request = request or []

        assert isinstance(self.request, list), "request must be list!"

        self.field = field.copy()

    def get_data(self, batch_size=128, model_num=0):
        def __inner(scanner):
            try:
                yield from scanner
            except Exception as e:
                #inner hbase error:  TTransportException(message='TSocket read 0 bytes', type=4)
                print("inner hbase error: ", e)
                return

        try:
            for req in self.request[:]:
                data = []
                self.connection.open()

                print("consuming %s's data!" % (req))
                for key, item in __inner(self.table.scan(filter=self.filter.format(req), batch_size=1, )):

                    tmp = {}
                    tag = False
                    try:
                        for k, v in item.items():
                            k = k.decode("utf8").split(":")[1]
                            v = v.decode("utf8")
                            if k in self.field:
                                if len(v) == 0:
                                    tag = True
                                    break
                            tmp[k] = v
                        if tag:
                            continue
                        if 0 != model_num:
                            if int(tmp["user_id"]) % 4 != model_num - 1:
                                continue
                    except Exception as e:
                        print("parse data error: ", e)
                        continue
                    ad_id = tmp["ad_id"]
                    try:
                        ad_id = int(ad_id)
                    except:
                        print("parse int error: ", "can't int")
                        continue
                    if ad_id < 10000:
                        continue
                    data.append(tmp)
                    if len(data) == batch_size:
                        yield data
                        data = []
                if len(data) > 0:
                    yield data
        except:
            ## this only omit the GeneratorExit Error
            ## we can do something here,like release resources ... ,but no need!!!
            pass
        finally:
            return

    def __enter__(self):
        self.connection.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        return True


if __name__ == "__main__":
    """To Test"""

    # t = TrainDataIter("train_filter.csv", batch_size=2000)
    # for i in t:
    #     pass
    #
    # for i in t:
    #     print(i)

    # filter_str = """RowFilter (=, 'substring:{}')"""
    # request = ["2019-04-%02d" % (i) for i in range(12, 16)]
    # with DataIter('10.9.135.235', b'midas_ctr_pro', filter_str, request, ) as d:
    #     for i in d.get_data(batch_size=128):
    #         pass

    mime = "10.9.75.202"
    other = "10.9.135.235"
    conn = happybase.Connection(
        host=mime,
        # host="localhost",
        # timeout=100,
    )

    conn.open()

    table = happybase.Table(b"midas_offline", conn)

    filter_str = """RowFilter (=, 'substring:{}')"""

    # scan = table.scan(filter=filter_str.format("2019-06-12"),
    #                   batch_size=1)

    cnt = 1

    # for key, value in scan:
    #     print(cnt, key, value, )
    #     cnt += 1
    #     if cnt >= 10:
    #         break

    conn.close()

    with HbaseDataIter("10.9.75.202", b'midas_offline_v1', filter_str, ["2019-07-17"], ) as d:
        for i in d.get_data():
            print(len(i), type(i))
