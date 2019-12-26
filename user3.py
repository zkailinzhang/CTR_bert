import json 
import os 
import fcntl 
import time

PATH = os.path.dirname(os.path.abspath(__file__))

FILE_USER_API = os.path.join('/Users/zhangkailin/zklcode/Midas_Engine','usersplit2.json')

if os.path.exists(FILE_USER_API):
    with open(FILE_USER_API,'r+') as jf:
        fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
        aa = json.load(jf)
        user_api_a = aa["df"]
        user_api_b = aa["df1"]
        user_api_c = aa["df2"]
        time.sleep(5)
        i=1
        while i<10000:
            user_api_a[str(i)] = i
            user_api_b[str(i)] = i
            user_api_c[str(i)] = i
            
            print("user3 : {}".format(i))
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