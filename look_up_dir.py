import os
import collections

PATH = os.path.dirname(os.path.abspath(__file__))

# In [2]: os.listdir('./update-model-1/model0.8/model/')                                                                                                     
# Out[2]: ['ckpt_0.data-00000-of-00001', 'ckpt_0.meta', 'checkpoint', 'ckpt_0.index']
def get_max_model_index(cnt: int = 1):
    model_path = "tmp-model-3/model/"
    if cnt != 1:
        model_path = "update-model-1/model/"

    num = -1
    #ckpt_0  0
    for i in os.listdir(os.path.join(PATH, model_path)):
        o = i.split(".")[0]
        try:
            a = int(o.split("_")[1])
            if a > num:
                num = a
        except:
            pass

    return num


def get_max_serving_index(cnt: int = 1):
    model_path = "tmp-model-3/serving/"
    if cnt != 1:
        model_path = "update-model-1/serving/"

    num = -1

    for i in os.listdir(os.path.join(PATH, model_path)):
        o = i.split(".")[0]
        try:
            a = int(o)
            if a > num:
                num = a
        except:
            pass

    return num


def get_last_day_fmt():
    import datetime
    day = datetime.datetime.today()
    yes = day + datetime.timedelta(days=-1)
    #'2019-07-23'
    return yes.strftime("%Y-%m-%d")

def get_now_day_fmt():
    import datetime
    day = datetime.datetime.today()
    
    return day.strftime("%Y-%m-%d")

def get_some_day_fmt(start,num):
    '''
    'Feb 28, 2018'
    number 包括当天
    '''

    import datetime
  
    start = datetime.datetime.strptime(start, '%b %d, %Y')
    
    dates = []

    for i in range(num):
        tmp = start + datetime.timedelta(days=i)
        dates.append(tmp.strftime("%Y-%m-%d"))

    index = [i for i in range(num)]
    rst = collections.OrderedDict(zip(index,dates))


    return rst


if __name__ == "__main__":
    print(get_max_model_index())
    print(get_max_serving_index())

    print(get_last_day_fmt())
    # 2019-04-%02
