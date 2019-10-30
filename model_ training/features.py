import pandas as pd
import asyncio
from feature.user.user_feature import UserFeature
from risk_platform.mysql_conn.bussiness_conn_pool import BussinessMysqlConn
from system_config import system_config
from risk_platform.thread_process_pool.multi_proccess_thread_run import task_run
from risk_platform.utils import time_utils


use_feature = ["last_overdue_days","pdl_last_payoff_dpd","installment_or_pdl_has_overdue_record",
               "installment_second_dpd","installment_first_dpd","pdl_payoff_peroids","is_pdl_unpay","pdl_payoff_avg_dpd",
               "installment_payoff_periods_lock","installment_last_100d_pd","installment_advanced_payoff_radio","last_one_year_has_pdl_bill",
               "pdl_last_100d_max_overdue_day",'pdl_last_100d_sum_overdue_day','last_180d_total_ptp_num','last_180d_total_ptp_ratio',
               'last_180d_total_feeckback_cnt', 'last_180d_min_level',"last_1_month_distinct_call_phone_number_cnt","device_related_uids_cnt",
               "current_repayment_date_debt",'installment_total_latefee','installment_pay_off_hdpd','avg_payoff_dpd','sum_overdue_days',
               'total_not_online_num_divide_effective_account_age','active_days_in_last90day_cnt','same_business_app_cnt',
               'recent_30_day_app_cnt','recent_30_same_business_app_cnt','is_bank_auth','installment_last_40d_dpd','user_active_days',
               'pdl_first_dpd','pdl_on_time_payoff_peroids','pdl_last_payoff_dpd','pdl_max_dpd','cash_loan_age','pdl_on_due_total_days','last_3_pdl_dpd',
               "installment_overdue_unpay_periods","installment_latefee_periods","sum_overdue_days","installment_ageing","unpaid_virtual_item","call_recharge_rate",
               "game_recharge_rate","flow_recharge_rate","installment_overdue_unpay","installment_unpay","avg_payoff_paid_up","max_theoretical_profit",
               "lst_180days_total_arrears_time","goods_total_num","get_user_use_most_periods","current_period_receive_phone_cnt","total_not_online_num",
               "total_not_online_num_divide_effective_account_age","ontime_repayment_num_last_12month","continue_no_overdue_bill",
               "overdue_less_than_30days_bill_rate",
               ]
bill_features = []
async def run():
    with BussinessMysqlConn() as conn:
        uid_list = conn.select_many_one_value(
            """select distinct uid from installmentdb.t_bill where repayment_date='2019-08-25' and 
            (status=1 or from_unixtime(floor((pay_off_time+8*3600000)/1000),'%%Y-%%m-%%d')>'2019-09-25') limit 50000""",)
    for uid in uid_list:
        feature = await UserFeature.get_features(uid, use_feature, country_id=1,end_time=time_utils.str_date_to_timestamp("2019-07-25",1))
        feature.update({"label":0})
        bill_features.append(feature)
    with BussinessMysqlConn() as conn:
        uid_list = conn.select_many_one_value(
            """select distinct uid from installmentdb.t_bill where repayment_date='2019-08-25' and status=2 and 
            from_unixtime(floor((pay_off_time+8*3600000)/1000),'%%Y-%%m-%%d')<'2019-09-25' limit 50000""",)
    for uid in uid_list:
        feature = await UserFeature.get_features(uid, use_feature, country_id=1,end_time=time_utils.str_date_to_timestamp("2019-07-25",1))
        feature.update({"label":1})
        bill_features.append(feature)

asyncio.get_event_loop().run_until_complete(run())
df = pd.DataFrame.from_records(bill_features)
df.to_csv("data.csv",index=False)
