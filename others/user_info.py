import urllib
import json
from urllib.request import urlopen

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


class RequestHandler():
    @staticmethod
    def post_recommend(json_text):
        urlpath = 'http://10.9.84.238:8130/midas/user/getUserLabel'
        head = {"Content-Type": "application/json"}
        request = urllib.Request(urlpath, json.dumps(json_text), head)
        response = urllib.urlopen(request)
        temp = response.read()
        #    params = urllib.urlencode(json_text)
        #    f = urllib.urlopen(urlpath, params)
        #    temp = f.read()
        out_text = json.loads(temp)
        if out_text["data"] == []:
            return ('0', '0', '0', '0', '0', '0')
        info = out_text["data"][0]
        # print info.keys()
        ability = info["math_ability_overall"]
        if ability == '未知':
            ability = '0'
        purchase = info["pay_test"]
        grade = info["grade"]
        province = info["province_id"]
        city = info["city_id"]
        county = info["county_id"]
        return (ability, purchase, grade, province, city, county)


import requests

answerinfo = {}
answerinfo["type"] = 3
answerinfo["columns"] = ["province_id", "city_id", "county_id",
                         "school_id", "grade", "chinese_ability_overall",
                         "english_ability_overall", "math_ability_overall",
                         "pay_test",
                         "seatwork_active_degree", "user_freshness"]
answerinfo["ids"] = [10000048]

url = "http://10.9.84.238:8130/midas/user/getUserLabel"
header = {"Content-Type": "application/json"}
o = requests.post(url=url, data=json.dumps(answerinfo), headers=header)
print(o.json())

# type 1 老师 2 学生 3 parent
