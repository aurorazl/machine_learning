from sklearn.externals import joblib
import pandas as pd

use_feature = ["last_overdue_days","pdl_last_payoff_dpd","installment_or_pdl_has_overdue_record",
               "installment_second_dpd","installment_first_dpd","pdl_payoff_peroids","is_pdl_unpay","pdl_payoff_avg_dpd",
               "installment_payoff_periods_lock","installment_last_100d_pd","installment_advanced_payoff_radio","last_one_year_has_pdl_bill",
               "pdl_last_100d_max_overdue_day",'pdl_last_100d_sum_overdue_day','last_180d_total_ptp_num','last_180d_total_ptp_ratio',
               'last_180d_total_feeckback_cnt', 'last_180d_min_level',"last_1_month_distinct_call_phone_number_cnt","device_related_uids_cnt",
               "current_repayment_date_debt",]
df = pd.read_csv("./data.csv")
df = df.reindex(columns=use_feature+["label"])
X,y = df.iloc[:,:-1].values,df.iloc[:,-1].values

model = joblib.load("lgb.pkl")
print(model.predict_proba(X[[50001],]),y[50001])