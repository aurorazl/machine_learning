import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split,learning_curve,validation_curve,cross_val_predict,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,roc_curve,auc,roc_auc_score,precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

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
df = pd.read_csv("./data_2.csv")
X = df.reindex(columns=use_feature)
X=X.fillna(0)
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,init="k-means++",n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km = km.fit_predict(X)
print(km.inertia_)