import happybase
import contextlib


@contextlib.contextmanager
def hbase(**kwargs):
    conn = happybase.Connection(**kwargs)
    conn.open()
    yield conn
    conn.close


mine = "10.9.75.202"
filter_str = """RowFilter (=, 'substring:{}')"""
"MIDAS_RECENT_CLICK_PRO"
"midas_ctr_pro"

"midas_online_user"
"midas_online_context"
"midas_online_ad"

# happybase.Table().regions()

if __name__ == "__main__":
    cnt = 0
    with hbase(host=mine) as conn:
        table = conn.table("midas_online_context")
        for i in table.scan():
            print(i)
            cnt += 1
            if cnt >= 100:
                break

(b'10002337_103465', {b'context:duplicate_tag': b'0', 
b'context:count_click': b'0', b'context:hexposure_alocation': b'7', 
b'context:window_otherad': b'0', b'context:hexposure_similarad': b'53', 
b'context:month_accuracy': b'0.0', b'context:dexposure_clocation': b'3', 
b'context:week_accuracy': b'0.0', b'context:dclick_otherad': b'0', 
b'context:log_week': b'2019/07/01', b'context:log_weektime': b'7', 
b'context:exposure_duration': b'0', b'context:is_click': b'0', 
b'context:hclick_similarad': b'1', b'context:exe_time': b'2019-07-06 21:09:28', 
b'context:log_month': b'2019/07/01', b'context:log_hourtime': b'21', 
b'context:rclick_ad': b'[101590, 102169]', b'context:location_ad': 
b'7', b'context:hexposure_clocation': b'6', b'context:dexposure_alocation': 
b'3', b'context:log_time': b'2019-07-06 21:09:17', b'context:rclick_category':
 b'[7, 5]', b'context:log_day': b'2019/07/06', b'context:yeaterday_accuracy': b'0.0'})

(b'10002337_103529', {b'context:duplicate_tag': b'0', b'context:count_click': b'0', 
b'context:hexposure_alocation': b'2', b'context:window_otherad': b'0', 
b'context:hexposure_similarad': b'78', b'context:month_accuracy': b'0.0', 
b'context:dexposure_clocation': b'2', b'context:week_accuracy': b'0.0', 
b'context:dclick_otherad': b'0', b'context:log_week': b'2019/07/01', 
b'context:log_weektime': b'6', b'context:exposure_duration': b'0', 
b'context:is_click': b'0', b'context:hclick_similarad': b'0', b'context:exe_time':
 b'2019-07-05 06:46:28', b'context:log_month': b'2019/07/01', b'context:log_hourtime': 
 b'6', b'context:rclick_ad': b'[101590, 102169]', b'context:location_ad': b'6', 
 b'context:hexposure_clocation': b'2', b'context:dexposure_alocation': b'2', 
 b'context:log_time': b'2019-07-05 06:46:13', b'context:rclick_category': b'[7, 5]', 
 b'context:log_day': b'2019/07/05', b'context:yeaterday_accuracy': b'0.0'})


userlog
(b'10000022', {b'user:app_freshness': b'G', b'user:county_id': b'1558', 
 b'user:city_id': b'181', b'user:app_type': b'3', b'user:chinese_ability': b'0', 
 b'user:english_ability': b'0', b'user:test_timestamp': b'1560940904729', 
 b'user:province_id': b'13', b'user:school_id': b'54085', b'user:purchase_power': b'B', 
 b'user:mobile_type': b'OPPO_R11s;7.1.1', b'user:mobile_os': b'1', b'user:activity_degree': b'E',
  b'user:grade_id': b'4', b'user:math_ability': b'E', b'user:user_id': b'10000022'})
(b'10000048', {b'user:app_freshness': b'G', b'user:county_id': b'1856', b'user:city_id': b'221', b'user:app_type': b'3', b'user:chinese_ability': b'0', b'user:english_ability': b'0', b'user:test_timestamp': b'1562471120571', b'user:province_id': b'16', b'user:school_id': b'861332', b'user:purchase_power': b'B', b'user:mobile_type': b'PAAT00;8.1.0', b'user:mobile_os': b'1', b'user:activity_degree': b'C', b'user:grade_id': b'4', b'user:math_ability': b'B', b'user:user_id': b'10000048'})
(b'10000237', {b'user:app_freshness': b'G', b'user:county_id': b'1086', b'user:city_id': b'138', b'user:app_type': b'3', b'user:chinese_ability': b'0', b'user:english_ability': b'0', b'user:test_timestamp': b'1562408881967', b'user:province_id': b'10', b'user:school_id': b'132854', b'user:purchase_power': b'B', b'user:mobile_type': b'OPPO_A77;7.1.1', b'user:mobile_os': b'1', b'user:activity_degree': b'D', b'user:grade_id': b'5', b'user:math_ability': b'B', b'user:user_id': b'10000237'})
(b'10000369', {b'user:app_freshness': b'G', b'user:county_id': b'1266', b'user:city_id': b'149', b'user:app_type': b'3', b'user:chinese_ability': b'0', b'user:english_ability': b'0', b'user:test_timestamp': b'1562114308748', b'user:province_id': b'11', b'user:school_id': b'929064', b'user:purchase_power': b'B', b'user:mobile_type': b'MI_5X;7.1.2', b'user:mobile_os': b'1', b'user:activity_degree': b'E', b'user:grade_id': b'2', b'user:math_ability': b'D', b'user:user_id': b'10000369'})

contextlog
(b'10000022_101809', {b'context:dexposure_alocation': b'5', b'context:exe_time': 
b'2019-06-19 18:50:03', b'context:rclick_ad': b'[]', b'context:dexposure_clocation': b'1', 
b'context:hclick_similarad': b'0', b'context:dclick_otherad': b'0', b'context:count_click': b'0', 
b'context:hexposure_alocation': b'5', b'context:window_otherad': b'0', 
b'context:hexposure_clocation': b'1', b'context:exposure_duration': b'0', b'context:is_click': b'0', 
b'context:yeaterday_accuracy': b'0.0', b'context:log_week': b'2019/06/17',
 b'context:log_hourtime': b'18', b'context:log_day': b'2019/06/19', 
 b'context:location_ad': b'10', b'context:week_accuracy': b'0.0', 
 b'context:rclick_category': b'[]', b'context:hexposure_similarad': b'1', 
 b'context:duplicate_tag': b'0', b'context:log_month': b'2019/06/01', 
 b'context:log_time': b'2019-06-19 18:49:52', b'context:log_weektime': b'4', 
 'context:month_accuracy': b'0.0'})
(b'10000022_101925', {b'context:dexposure_alocation': b'17', b'context:exe_time': b'2019-06-19 19:06:58', b'context:rclick_ad': b'[]', b'context:dexposure_clocation': b'17', b'context:hclick_similarad': b'0', b'context:dclick_otherad': b'0', b'context:count_click': b'0', b'context:hexposure_alocation': b'17', b'context:window_otherad': b'0', b'context:hexposure_clocation': b'17', b'context:exposure_duration': b'0', b'context:is_click': b'0', b'context:yeaterday_accuracy': b'0.0', b'context:log_week': b'2019/06/17', b'context:log_hourtime': b'19', b'context:log_day': b'2019/06/19', b'context:location_ad': b'6', b'context:week_accuracy': b'0.0', b'context:rclick_category': b'[]', b'context:hexposure_similarad': b'40', b'context:duplicate_tag': b'1', b'context:log_month': b'2019/06/01', b'context:log_time': b'2019-06-19 19:06:48', b'context:log_weektime': b'4', b'context:month_accuracy': b'0.0'})

adlog

(b'100198', {b'ad:alldexposure_alocation': b'1', b'ad:alldexposure_clocation': b'1',
 b'ad:location_ad': b'3', b'ad:label_3': b'4', b'ad:allhexposure_alocation': b'2', 
 b'ad:exposure_duration': b'97', b'ad:label_2': b'4', b'ad:window_otherad': b'0', 
 b'ad:label_1': b'1', b'ad:count_click': b'0', b'ad:label_7': b'-1', 
 b'ad:test_timestamp': b'1561433406087', b'ad:ad_id': b'100198', 
 b'ad:allhexposure_clocation': b'17', b'ad:label_5': b'-1', b'ad:label_4': 
 b'-1', b'ad:label_6': b'-1'})
(b'100199', {b'ad:alldexposure_alocation': b'0', b'ad:alldexposure_clocation': b'0', b'ad:location_ad': b'3', b'ad:label_3': b'1', b'ad:allhexposure_alocation': b'6', b'ad:exposure_duration': b'109', b'ad:label_2': b'3', b'ad:window_otherad': b'0', b'ad:label_1': b'3', b'ad:count_click': b'1', b'ad:label_7': b'-1', b'ad:test_timestamp': b'1562428833586', b'ad:ad_id': b'100199', b'ad:allhexposure_clocation': b'6', b'ad:label_5': b'-1', b'ad:label_4': b'-1', b'ad:label_6': b'-1'})



'''
(b'10000022_101475_2019-06-19 18:42:27', 
{b'context:count_click': b'0', b'context:log_time': b'2019-06-19 18:42:27', 
b'context:log_day': b'2019/06/19', b'ad:exposure_duration': b'5', b'ad:test_timestamp': 
b'1560911463778', b'user:school_id': b'54085', b'user:activity_degree': b'E',
 b'context:week_accuracy': b'0.0', b'context:exposure_duration': b'0', b'context:exe_time': 
 b'2019-06-19 18:42:40', b'ad:label_6': b'-1', b'context:hexposure_alocation': b'1',
  b'ad:count_click': b'350', b'context:dexposure_alocation': b'1', b'ad:location_ad': b'6', 
  b'context:is_click': b'0', b'user:county_id': b'1558', b'context:hclick_similarad': b'0', 
  b'context:log_month': b'2019/06/01', b'ad:ad_id': b'101475', b'ad:label_3': b'4', 
  b'context:log_week': b'2019/06/17', b'context:window_otherad': b'0', b'user:grade_id': b'4', 
  b'user:english_ability': b'0', b'context:yeaterday_accuracy': b'0.0', b'ad:label_4': b'-1',
   b'ad:alldexposure_clocation': b'407573', b'user:mobile_os': b'1', b'context:location_ad': b'6', 
   b'context:hexposure_similarad': b'9', b'user:chinese_ability': b'0', b'ad:allhexposure_alocation': 
   b'63035', b'ad:label_5': b'-1', b'ad:label_2': b'3', b'context:hexposure_clocation': b'1',
    b'ad:alldexposure_alocation': b'4324', b'context:dclick_otherad': b'0', b'user:user_id': b'10000022', 
    b'user:purchase_power': b'B', b'user:app_freshness': b'G', b'user:math_ability': b'E', 
    b'context:log_hourtime': b'18', b'user:app_type': b'3', b'ad:label_1': b'1',
     b'context:log_weektime': b'4', b'context:rclick_ad': b'[]', b'user:province_id': b'13', 
     b'ad:window_otherad': b'0', b'user:mobile_type': b'OPPO_R11s;7.1.1', b'user:test_timestamp':
      b'1560940904729', b'context:duplicate_tag': b'0', b'ad:label_7': b'-1', b'context:rclick_category': 
      b'[]', b'context:month_accuracy': b'0.0', b'user:city_id': b'181', b'ad:allhexposure_clocation': 
      b'9418410', b'context:dexposure_clocation': b'1'})
'''