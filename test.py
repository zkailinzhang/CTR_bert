import json

import requests

headers = {
    "Content-Type": "application/json"
}

request_data = {"user_info": {"user_id": 1,
                              "user_type": 2,
                              "province_id": 1, "city_id": 1, "county_id": 1,
                              "school_id": 1, "grade": 1,
                              "chinese_ability_overall": 1,
                              "english_ability_overall": 1,
                              "math_ability_overall": 1,
                              "pay_test": 1, "seatwork_active_degree": 1,
                              "user_freshness": 1,
                              },
                "ad_list": [{"ab_id": 1,
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
                    {"ad_id": 1009, "ab_id": [1, 3], "gap": 0},
                    {"ad_id": 1008, "ab_id": [2, 4], "gap": -0.1}, ],
                }

# print(requests.post("http://10.9.34.134:8000/advertisement/predict",
#                     data=json.dumps(request_data),
#                     headers=headers
#                     ).json()
#       )

if __name__ == "__main__":
    pass

    import json

    request_data = {
        "user_info": {"user_id": 284, "user_type": 3, "device_os": 2, "user_freshness": 0, "seatwork_active_degree": 0,
                      "pay_test": 0, "math_ability_overall": 0, "english_ability_overall": 0,
                      "chinese_ability_overall": 0,
                      "grade": 0, "province_id": 0, "city_id": 0, "county_id": 0, "school_id": 0},
        "ad_list": [{"ab_id": 100232, "tags": {}}, {"ab_id": 100215, "tags": {}}, {"ab_id": 100221, "tags": {}},
                    {"ab_id": 100219, "tags": {}}, {"ab_id": 100220, "tags": {}}],
        "relation": [{"ad_id": 100141, "ab_id": [100232], "gap": 0.2}, {"ad_id": 100124, "ab_id": [100215], "gap": 0.2},
                     {"ad_id": 100130, "ab_id": [100221], "gap": 0.2}, {"ad_id": 100128, "ab_id": [100219], "gap": 0.2},
                     {"ad_id": 100129, "ab_id": [100220], "gap": 0.2}]}

    # with open("test_serve.json", "w") as f:
    #     f.write(json.dumps(request_data))

    """
    siege -c 1000 -t 20s  -b "http://10.19.117.187:8002/advertisement/predict POST </tmp/ad_kafka/test_serve.json"
    """

