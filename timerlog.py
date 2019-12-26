import os
import time
import re 
import json
import logging
import traceback 
import time

PATH = os.path.dirname(os.path.abspath(__file__))

path_user = os.path.join(PATH,'apisplit.error.json')
path_log = os.path.join(PATH,'logs.dsin_ns.out')
path_timelog = os.path.join(PATH,'logs.timer.out')

logging.basicConfig(filename=path_timelog,filemode='a',
format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y %H:%M:%S",
level=logging.DEBUG)

def search_user():
   
    pat = "\"80"
    with open(path_user,'r') as jf:
        userapi = json.load(jf)
        userapi_str = json.dumps(userapi)
        for match in re.finditer(pat,userapi_str):
            s  = match.start()
            e  = match.end()

            print(s,e,userapi_str[s:e])
            logging.info("!!!! match json ok {},{}, {}".format(s,e,userapi_str[s:e]))

def search_log1():
   
    pat = 'another exception occurred'

    pat1 = "JSONDecodeError: Expecting ':' delimiter:"
    pat2 = "KeyError: 'user_api_a'"
    with open(path_log,'r') as jf:
        user_str = jf.read()
        for match in re.finditer(pat1,user_str):
            s  = match.start()
            e  = match.end()

            print(s,e,user_str[s:e])
            logging.info("!!!! match logout ok {},{}, {}".format(s,e,user_str[s:e]))
        for match in re.finditer(pat2,user_str):
            s  = match.start()
            e  = match.end()

            print(s,e,user_str[s:e])
            logging.info("!!!! match logout ok {},{}, {}".format(s,e,user_str[s:e]))

def search_log():
    pat4 = '*** HbaseDataIterUpdate|another exception occurred'
    pat0 = 'another exception occurred'
    pat1 = "JSONDecodeError: Expecting ':' delimiter:"
    pat2 = "KeyError: 'user_api_a'"
    
    with open(path_log,'r') as jf:
        user_str = jf.read()
        m = re.match(pat4,user_str)
        if m is not None:
            logging.info("!!!! match logout ok {}".format(m.group()))
            print(m.group())
            
        else:
            print("match failed")
                      


def print_ts(message):
    #print ("[%s] %s"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
    logging.info("[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
    
def run(interval, command):
    #print_ts("-"*100)
    
    print_ts("Starting every %s seconds."%interval)
    #print_ts("-"*100)
    while True:
        try:
            # sleep for the remaining seconds of interval
            time_remaining = interval-time.time()%interval
            #print_ts("Sleeping until %s (%s seconds)..."%((time.ctime(time.time()+time_remaining)), time_remaining))
            time.sleep(time_remaining)
            
            # execute the command
            search_user()
            search_log()
            #print_ts("-"*100)

        except Exception as e:
            print(e)
            logging.info("execept e time error {}".format(e))

if __name__=="__main__":
    interval = 5
    command = r"ls"
    run(interval, command)
