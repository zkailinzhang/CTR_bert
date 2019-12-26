from pykafka import KafkaClient
from pykafka.common import OffsetType
import pandas as pd

"""recom  kafka"""
# def desc(value, partition_key):
#     return (eval(value.decode("utf-8")), partition_key)
#
#
# client = KafkaClient(hosts="10.10.53.28:9092,10.10.32.192:9092,10.10.59.134:9092",
#                      zookeeper_hosts="10.10.53.28:2181")
#
# topic = client.topics["modelKafka".encode()]
#
# consumer = topic.get_simple_consumer(consumer_group="test",
#                                      auto_offset_reset=OffsetType.EARLIEST,
#                                      reset_offset_on_start=True,
#                                      fetch_message_max_bytes=1024 * 1024 * 1024,
#                                      auto_commit_enable=False,
#                                      auto_commit_interval_ms=1,
#                                      queued_max_messages=10000,
#                                      deserializer=desc,
#                                      )
#
# msg = consumer.consume()
#
# a = msg.value
# print(a["requestParams"])
# print(a["hisHomeWork"])
# print(pd.read_json(a["modelParams"]))


client = KafkaClient(hosts="10.19.127.87:9092,10.19.103.105:9092,10.19.176.30:9092", )

topic_log = client.topics["KFK-SUBTPC-app_dotting_log-ad_test".encode()]
topic_api = client.topics["susuan_api_log".encode()]


def parse_log(value, partition_key):
    return (value.decode("utf-8").split("|"), partition_key)


def parse_api(value, partition_key):
    value_split = value.decode("utf-8").split(" ")
    t = value_split[0] + " " + value_split[1]

    detail = {}
    for nn in value_split[3].split("\t"):
        key, value = nn.split("=")
        detail[key] = value
    return ([t, detail], partition_key)


consumer = topic_log.get_simple_consumer(consumer_group="ad_test",
                                         auto_offset_reset=OffsetType.LATEST,
                                         # auto_offset_reset=OffsetType.EARLIEST,
                                         reset_offset_on_start=False,
                                         fetch_message_max_bytes=1024 * 1024,
                                         auto_commit_enable=False,
                                         auto_commit_interval_ms=1000 * 30,
                                         queued_max_messages=10000,
                                         deserializer=parse_log,
                                         )

# consumer = topic_api.get_simple_consumer(consumer_group="ad_test",
#                                          auto_offset_reset=OffsetType.EARLIEST,
#                                          reset_offset_on_start=True,
#                                          fetch_message_max_bytes=1024 * 1024,
#                                          auto_commit_enable=False,
#                                          # auto_commit_interval_ms=100,
#                                          queued_max_messages=10000,
#                                          deserializer=parse_api,
#                                          )

i = 0
cc = []
import MySQLdb

TABLE = "ad_log"
COLS = ["action_time", "user_id", "product_id",
        "app_version", "app_source", "app_channel",
        "page_code", "page_from", "device_type", "device_version",
        "device_id", "ad_id", "event", "action_code"]
insert = "insert into {}({}) values({})".format(TABLE, ",".join(COLS),
                                                ",".join(["%s"] * len(COLS)))

conn = MySQLdb.connect(host='10.10.169.75',
                       user='root',
                       passwd='Knowbox512+_*',
                       db='richard',
                       port=3306, )
cursor = conn.cursor()

try:
    for msg in consumer:
        i += 1
        tmp = msg.value
        try:
            int(tmp[1])
        except:
            continue
        cc = tmp.pop(-2)
        try:
            cc = eval(cc)
            if len(cc) != 2:
                continue
            for key in cc:
                cc[key] = int(cc[key])
        except:
            continue
        if cc["event"] not in (1, "1"):
            continue
        tmp.insert(-1, cc["adId"])
        tmp.insert(-1, cc["event"])
        try:
            cursor.execute(insert, tmp)
            conn.commit()
        except Exception as e:
            print(e)
            continue
finally:
    cursor.close()
    conn.close()

# pd.DataFrame(cc).to_csv("kafka_data.csv",index=False)


#
# a = a.groupby(by=["1"], axis=0)
#
# cnt = 0
# for k, v in a:
#     if k == "(null)":
#         continue
#     if len(v) > 1:
#         v = v.sort_values(by=["0"])
#         for oo in v["11"]:
#             o = eval(oo)
#             if o["event"] in (1, "1"):
#                 print(v[["1", "11", ]])
#                 cnt += 1
#                 if cnt >= 10:
#                     break
