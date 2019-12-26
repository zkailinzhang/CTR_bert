# Log


uuid = sha256 同一个request是一个uuid，各自不同

time = 2019-04-15 14:29:53,783

每个Level一天一个 * N台机器


## info

+ event :
    + Raw : 原始request (just for bugs)
    + Request : json化的request
    + TFRequest : json化的tf-request (may null when ad_list=[])
    + TFResponse : json化的tf-response (may null when ad_list=[])
    + Response : json化的response

+ LEVEL : INFO

+ format : 
	+ time - LEVEL - uuid - event - `content`

+ instance : 
	+ 2019-04-15 14:29:53,783 - LEVEL - n12adgyadbadbaduczbubna - event - This is a info log.

## error

+ event
	+ Error : error 栈信息

+ LEVEL : ERROR

+ format : 
	+ time - LEVEL - uuid - event - `content`

+ instance : 
	+ 2019-04-15 14:29:53,783 - LEVEL - n12adgyadbadbaduczbubna - event - This is a error log.

## analysis

+ event :
	+ Analysis: [abid+tf-response的map]+[request里面的"relation"]+["relation"里面和tf-response的结合结果--基本和resopnse一致]+[resopnse]-->json化
		+ tmp["tf"] = {10: [0.1, 0.9], 11: [0.2, 0.8], 20: [0.3, 0.7], 21:[0.4, 0.6]}
		+ tmp["relation"] = [{"ad_id": 1, "ab_id": [10, 11], "gap": 0.2},
                   {"ad_id": 2, "ab_id": [20, 21], "gap": 0.2}]
		+ tmp["before_gap"] = [{"ad_id": 1, "ab_id": 10, "score": 0.9},
                     {"ad_id": 2, "ab_id": 20, "score": 0.7}]
		+ tmp["after_gap"] = [{"ad_id": 1, "ab_id": 10, "score": 0.008},
                    {"ad_id": 2, "ab_id": 20, "score": 0.006}]

+ LEVEL : ANALYSIS

+ format : 
	+ time - LEVEL - uuid - event - `content`

+ instance : 
	+ 2019-04-15 14:29:53,783 - LEVEL - n12adgyadbadbaduczbubna - event - This is a analysis log.


## time

+ event
	+ Hbase : 过程消耗毫秒
	+ Feature : 过程消耗毫秒
	+ Tf : 过程消耗毫秒

+ LEVEL : TIME

+ format : 
	+ time - LEVEL - uuid - event - `content`

+ instance : 
	+ 2019-04-15 14:29:53,783 - LEVEL - n12adgyadbadbaduczbubna - event - 100ms.
