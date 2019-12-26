import os 
import re
import json
import numpy as np 
import copy

PATH = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(PATH,'userapi.json')





jf = open(path,'r+')

userapi = json.load(jf)

values = list(userapi.values())
keys = list(userapi.keys())

max_num = np.max(values)

values_cp = copy.deepcopy(values)

values_cp.sort()

one_dict = {}
two_dict = {}
three_dict ={}

for i in range(max_num):
    values_tmp = keys[i].split("/")
    if values_tmp == ['']:
        one_dict[values_tmp[0]] = 1
        two_dict[values_tmp[0]] = 1
        three_dict[values_tmp[0]] = 1
        continue
    if values_tmp[1] not in one_dict:
        one_dict[values_tmp[1]] = len(one_dict)+1
    if values_tmp[2] not in two_dict:
        two_dict[values_tmp[2]] = len(two_dict)+1
    if values_tmp[3] not in three_dict:
        three_dict[values_tmp[3]] = len(three_dict)+1


print(len(list(one_dict.keys())))  # 35         200     35 
print(len(list(two_dict.keys())))   # 75        200     76 
print(len(list(three_dict.keys())))  # 358     800      359

with open(os.path.join(PATH,'usersplit.json'),'w') as jf:
    # jf.write(json.dump(one_dict))
    # jf.write(json.dumps(two_dict))
    # jf.write(json.dumps(three_dict))
    # json.dump(one_dict,jf)
    # json.dump(two_dict,jf)
    # json.dump(three_dict,jf)
    jf.write(json.dumps(one_dict)+'\n'+json.dumps(two_dict)+'\n'+json.dumps(three_dict))



#read 
'''one_ ={}
two_ ={}
three ={}
with open('data.json', 'r') as f:
    print(json.loads(f.readline()))
    print(json.loads(f.readline()))

    one_ = json.loads(f.readline())
    two_ = json.loads(f.readline())
    three_ = json.loads(f.readline())




if os.path.exists(path):
    #a+
    with open(path,'r+') as jf:
        
        user_api_a = json.loads(jf.readline())
        user_api_b = json.loads(jf.readline())
        user_api_c = json.loads(jf.readline())
        
        
        user_api_a["values_tmp"] = 1
        user_api_b["values_tmp"] = 1
        user_api_c["values_tmp"] = 1
        jf.seek(0)
        jf.truncate()
        jf.write(json.dumps(user_api_a)+'\n'+json.dumps(user_api_b)+'\n'+json.dumps(user_api_c))
            '''