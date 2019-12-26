import csv
import logging
import json
import traceback
import fcntl 
import time
import sys
import os
import datetime



PATH = os.path.dirname(os.path.abspath(__file__))

FILE_USER_API_A = os.path.join(PATH,"userapi_a.json")
FILE_USER_API_B = os.path.join(PATH,"userapi_b.json")
FILE_USER_API_C = os.path.join(PATH,"userapi_c.json")

logging.basicConfig(filename=os.path.join(PATH,"1.log"),filemode='w', # w
format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y %H:%M:%S",
level=logging.DEBUG)


def process_rbehavior_split(index):

    logging.info("start now time:{} ".format( 
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))
    user_api_a_update={}
    user_api_b_update={}
    user_api_c_update={}
    if os.path.exists(FILE_USER_API_A):
        #user_api_all={}          
        
        with open(FILE_USER_API_A,'r+') as jf:
            fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
            data = json.load(jf)
            user_api_a = data["user_api_a"]
            #
                #logging.info("")
        
            #jf.seek(0)
            user_api_a_update["user_api_a"] = user_api_a
            user_api_all_data = json.dumps(user_api_a_update)
            #rst = jf.write(user_api_all_data)
            jf.seek(0)

            rst = jf.write(user_api_all_data)
            jf.flush()

            logging.info("file A now time:{} index:{}".format( 
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') ,index ))

            fcntl.flock(jf.fileno(),fcntl.LOCK_UN)
    if os.path.exists(FILE_USER_API_B):        
        with open(FILE_USER_API_B,'r+') as jf:
            fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
            data = json.load(jf)
            user_api_a = data["user_api_b"]
            

            #jf.seek(0)
            user_api_b_update["user_api_b"] = user_api_a
            user_api_all_data = json.dumps(user_api_b_update)
            #rst = jf.write(user_api_all_data)
            jf.seek(0)

            rst = jf.write(user_api_all_data)
            jf.flush()

            logging.info("file B now time:{} index:{}".format( 
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),index  ))

            fcntl.flock(jf.fileno(),fcntl.LOCK_UN)
    if os.path.exists(FILE_USER_API_C):  
        with open(FILE_USER_API_C,'r+') as jf:
            fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
            data = json.load(jf)
            user_api_a = data["user_api_c"]
            #
                #logging.info("")

            #jf.seek(0)
            user_api_c_update["user_api_c"] = user_api_a
            user_api_all_data = json.dumps(user_api_c_update)
            #rst = jf.write(user_api_all_data)
            jf.seek(0)

            rst = jf.write(user_api_all_data)
            jf.flush()
            
            logging.info("file C now time:{} index:{}".format( 
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),index  ))

            fcntl.flock(jf.fileno(),fcntl.LOCK_UN)


    logging.info("end now time:{} ".format( 
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') ))





def testtime():
    user_api_c_update={}
    if os.path.exists(FILE_USER_API_C):
        #user_api_all={}          
        
        with open(FILE_USER_API_C,'r+') as jf:
            fcntl.flock(jf.fileno(),fcntl.LOCK_EX)
            data = json.load(jf)
            user_api_a = data["user_api_c"]

            for key,value in user_api_a.items():
                pass
                #print('{key}:{value}'.format(key = key, value = value))

        
            #jf.seek(0)
            user_api_c_update["user_api_c"] = user_api_a
            user_api_all_data = json.dumps(user_api_c_update)
            #rst = jf.write(user_api_all_data)
            jf.seek(0)

            rst = jf.write(user_api_all_data)
            jf.flush()

            fcntl.flock(jf.fileno(),fcntl.LOCK_UN)

if __name__ == "__main__":
    
    # starttime = datetime.datetime.now()
    start = time.clock()
    testtime()
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)  #0.003s
    # endtime = datetime.datetime.now()

    # print ((endtime - starttime).seconds)


    for i in range(100):
        process_rbehavior_split(i)