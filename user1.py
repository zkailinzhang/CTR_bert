import json 
import os 
import fcntl 
import time 

PATH = os.path.dirname(os.path.abspath(__file__))

FILE_USER_API = os.path.join('/Users/zhangkailin/zklcode/Midas_Engine','apisplit.error.json')


for i in range(10):
    tmp = {}
    tmp[str(i)] =i

print(tmp)

dataall ={}
aaa = dataall["user_api_a"]
with open(FILE_USER_API,'r+') as jf:
                #fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
                data = json.load(jf)
                user_api_a = data["user_api_a"]
                user_api_b = data["user_api_b"]
                user_api_c = data["user_api_c"]
                aa = data["user_api_c"]["reward-share-circle"]
                bb = data["user_api_c"][""]
aaa = dataall["user_api_a"]

with open(FILE_USER_API,'r+') as jf:
            user_api_all = json.load(jf)
            for i in r_behavior:
                if i == "":
                        user_api_all[i] = 1
                else:
                    if "" in user_api_all:
                        user_api_all[i] = len(user_api_all)+1 
                    else: 
                        user_api_all[i] = len(user_api_all)+2
            
            user_api_data = json.dumps(user_api_all)
            jf.seek(0)
            #jf.truncate()
            jf.write(user_api_data)
            jf.flush()


# df = {'b':[1,2,3,4],'a':'hello world!'}
# df1 = {'b':[1,2,3,4],'a':'hello world!'}
# df2 = {'b':[1,2,3,4],'a':'hello world!'}
# alldf = {}

# with open(FILE_USER_API,'w') as jf:

#     alldf["df"] = df
#     alldf["df1"] = df1
#     alldf["df2"] = df2
#     json.dump(alldf,jf)
 


if os.path.exists(FILE_USER_API):
    #a+

    with open(FILE_USER_API,'r+') as jf:
        fcntl.flock(jf.fileno(),fcntl.LOCK_EX)

        aa = json.load(jf)
        user_api_a = aa["df"]
        user_api_b = aa["df1"]
        user_api_c = aa["df2"]
        #time.sleep(5)
        i=1
        while i<10:
            user_api_a[str(i+10)] = i
            user_api_b[str(i+10)] = i
            user_api_c[str(i+10)] = i
            
            print("user2 : {}".format(i))
            i =i+1 
                
            
        jf.seek(0)
        # jf.truncate()

        aa["df"] = user_api_a
        aa["df1"] = user_api_b
        aa["df2"] = user_api_c
        json.dump(aa,jf)

        jf.flush()
        #jf.write(json.dumps(user_api_a)+'\n'+json.dumps(user_api_b)+'\n'+json.dumps(user_api_c))
        fcntl.flock(jf.fileno(),fcntl.LOCK_UN)




  File "/Users/zhangkailin/anaconda3/envs/tf12/lib/python3.6/json/decoder.py", line 355, in raw_decode
    obj, end = self.scan_once(s, idx)
json.decoder.JSONDecodeError: Expecting ':' delimiter: line 1 column 8194 (char 8193)



  File "/Users/zhangkailin/zklcode/Midas_Engine/user1.py", line 18, in <module>
    aaa = dataall["user_api_a"]
KeyError: 'user_api_a'