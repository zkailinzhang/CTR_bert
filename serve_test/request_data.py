"""
Request Data Form
"""
import json, time

request_data = {"user_info": {"user_id": 1,
                              "user_type": 2,  # 1,2,3 1 老师 2 学生 3 parent
                              "device_os": 1,
                              "province_id": 1, "city_id": 1, "county_id": 1,
                              "school_id": 1, "grade": 1,
                              "chinese_ability_overall": 1,
                              "english_ability_overall": 1,
                              "math_ability_overall": 1,
                              "pay_test": 1, "seatwork_active_degree": 1,
                              "user_freshness": 1,
                              },
                "ad_list": [
                    {"ab_id": 1,
                     "tags": {"key": 1, }
                     },
                    {"ab_id": 3,
                     "tags": {"key": 1, }
                     },
                    {"ab_id": 2,
                     "tags": {"key": 1, }
                     },
                    {"ab_id": 4,
                     "tags": {"key": 1, }
                     }
                ],
                "relation": [
                    {"ad_id": 1009, "ab_id": [1, 2], "gap": 0.6},
                    {"ad_id": 1008, "ab_id": [3, 4], "gap": -0.1}, ],
                }

response_data = {
    "code": 0,  # !=0
    "message": "success",  # "error"
    "response": [
        {"ad_id": 1009, "ab_id": 1, "score": 0.8},
        {"ad_id": 1008, "ab_id": 4, "score": 0.6},
    ]  # "error"
}

request_data = {
    "user_info": {"user_id": 284, "user_type": 3, "device_os": 2, "user_freshness": 7, "seatwork_active_degree": 5,
                  "pay_test": 0, "math_ability_overall": 0, "english_ability_overall": 0, "chinese_ability_overall": 0,
                  "grade": 2, "province_id": 19, "city_id": 258, "county_id": 2167, "school_id": 931772},
    "ad_list": [{"ab_id": 100215, "tags": {}}, {"ab_id": 100221, "tags": {}}, {"ab_id": 100219, "tags": {}},
                {"ab_id": 100220, "tags": {}}, {"ab_id": 100232, "tags": {}}],
    "relation": [{"ad_id": 100124, "ab_id": [100215], "gap": -0.05686274509803918},
                 {"ad_id": 100130, "ab_id": [100221], "gap": -0.05686274509803918},
                 {"ad_id": 100128, "ab_id": [100219], "gap": -0.03137254901960784},
                 {"ad_id": 100129, "ab_id": [100220], "gap": -0.0549019607843137},
                 {"ad_id": 100141, "ab_id": [100232], "gap": 0.2}]}

request_data = {
    "user_info": {"user_id": 284, "user_type": 3, "device_os": 2, "user_freshness": 0, "seatwork_active_degree": 0,
                  "pay_test": 0, "math_ability_overall": 0, "english_ability_overall": 0, "chinese_ability_overall": 0,
                  "grade": 0, "province_id": 0, "city_id": 0, "county_id": 0, "school_id": 0},
    "ad_list": [{"ab_id": 100232, "tags": {}}, {"ab_id": 100215, "tags": {}}, {"ab_id": 100221, "tags": {}},
                {"ab_id": 100219, "tags": {}}, {"ab_id": 100220, "tags": {}}],
    "relation": [{"ad_id": 100141, "ab_id": [100232], "gap": 0.2}, {"ad_id": 100124, "ab_id": [100215], "gap": 0.2},
                 {"ad_id": 100130, "ab_id": [100221], "gap": 0.2}, {"ad_id": 100128, "ab_id": [100219], "gap": 0.2},
                 {"ad_id": 100129, "ab_id": [100220], "gap": 0.2}]}
# request_data["ad_list"]=[]
# request_data["relation"]=[]
# request_data["ad_list"]=[{"ab_id":10,"tags":{}}]
# request_data["relation"]=[{"ad_id":20,"ab_id":[10],"gap":0.2}]

jsondata = json.dumps(request_data)
print(jsondata)

# print(jsondata)

def get_json_list(a: int = 10000):
    ll = []
    aim = request_data.copy()
    for i in range(a):
        aim["user_info"].update({"user_id": i})
        ll.append(json.dumps(aim))
    return ll

jsondatalist =get_json_list()


# with open("/data/lishuang/post3.json","w") as f:
#     f.write(jsondata)


def req(cnt: int):
    begin = time.time()
    # "10.9.34.134"
    "10.19.90.95"
    "10.19.83.137"

    d = requests.post("http://10.19.83.137:8000/advertisement/predict",
                      data=jsondata).json()
    # d = requests.post("http://172.16.0.82:8000/advertisement/predict/new",
    #                   data=jsondatalist[cnt]).json()
    print(type(d),d)
    print(cnt, time.time() - begin, )


base_ph = ["uid_ph", "mid_ph", "mobile_ph",
           "province_ph", "city_ph", "grade_ph",
           "math_ph", "english_ph", "chinese_ph",
           "purchase_ph", "activity_ph", "freshness_ph",
           "hour_ph"
           ]
data = {}
for i in base_ph:
    data[i] = [1] * 7
# data["uid_ph"]=[284]*2
# data["mid_ph"]=[1,1]
# data["mobile_ph"]=[2]*2
# data["province_ph"]=[19]*2
# data["city_ph"]=[258]*2
# data["math_ph"]=[0]*2
# data["english_ph"]=[0]*2
# data["chinese_ph"]=[0]*2
# data["purchase_ph"]=[0]*2
# data["activity_ph"]=[5]*2
# data["freshness_ph"]=[7]*2
# data["hour_ph"]=[10]*2

dd = {"signature_name": "serving",
      "inputs": data.copy()
      }
serve_data = json.dumps(dd)


# with open("/data/lishuang/serve2.json","w") as f:
#     f.write(serve_data)


def serve(cnt: int):
    begin = time.time()
    d = requests.post("http://10.19.83.137:8501/v1/models/midas:predict",
                      data=serve_data).json()
    print(d)
    print(cnt, time.time() - begin, )


if __name__ == "__main__":
    import requests
    import threading

    cnt = 10
    p = []
    for i in range(cnt):
        p.append(threading.Thread(target=req, args=(i,)))
    for i in p:
        i.start()
    for i in p:
        i.join()

    pass

