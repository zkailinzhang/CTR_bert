import codecs

user = {}

qqqq = -10
uuuu = ""
llll = ""

with codecs.open("/data/lishuang/midas_time_pro_8001_2019-06-14.log", "r") as f:
    while True:
        line = f.readline().strip()
        if not line:
            break
        items = line.split(">")
        ll = user.get(items[2], [0, 0, 0, 0])
        index = -1
        if items[3] == "SUM":
            index = 3
        elif items[3] == "Hbase":
            index = 0
        elif items[3] == "Feature":
            index = 1
        elif items[3] == "Tf":
            index = 2
        else:
            print("event error!")
            print(line)
            continue
        if qqqq < float(items[4][:-2]):
            qqqq = float(items[4][:-2])
            uuuu = items[2]
            llll = line
        ll[index] = float(items[4][:-2])
        user[items[2]] = ll

print(qqqq)
print(uuuu)
print(llll)

import pandas as pd

a = pd.DataFrame.from_dict(user, orient="index", columns=["a", "b", "c", "d"])
print(len(a))
# tf
print(max(a["c"]), len(a[a["c"] > 300]), len(a[a["c"] > 100]), len(a[a["c"] > 50]))
# hbase
print(max(a["a"]), len(a[a["a"] > 500]), len(a[a["a"] > 200]))
# sum
print(max(a["d"]), len(a[a["d"] > 500]), len(a[a["d"] > 200]))


# a["e"] = a["d"] - a["a"]
# b = a[a["d"] > 1000]
# print(len(b))
# print(len(b[b["a"] > 1000]))
# print(b["c"])





