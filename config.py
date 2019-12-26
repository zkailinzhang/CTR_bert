"""
no import, just some configs!

"""


AD_BOUND = 10000
USER_BOUND = 10000000
USER_SUM = 10000
AD_SUM = 100000
CITY_SUM = 5000


EMBEDDING_DIM = 128
ATTENTION_SIZE = 128
ABILITY_DIM = 5

HBASE_HOST = "10.9.75.202"
HBASE_TABLE = b"midas_offline"

HBASE_FIELD = ["mobile_os",
               "province_id", "grade_id", "school_id", "city_id", "county_id",
               "purchase_power",
               "math_ability", "english_ability", "chinese_ability",
               "activity_degree", "app_freshness", "ad_id", "user_id",
               "log_hourtime",
               ##########CLICK##########
               "is_click"
               ]
