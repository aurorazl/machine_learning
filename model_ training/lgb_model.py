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
df = df.reindex(columns=use_feature+["label"])
df = df.fillna(0)
# corr_matrix = df.corr()
# print(corr_matrix["label"].sort_values(ascending=False))

def search_params(clf,X_train,y_train):
    param_grid = [{
            # "clf__max_depth": range(2,10,1),
            # "clf__num_leaves": list(chain(range(2,20,4),range(20,200,50))),
            # "clf__learning_rate": [0.1,0.01,0.001],
            # "clf__n_estimators": [100,300,10],
            # "clf__boosting_type":["gbdt",'dart','goss'], # rf
            # "clf__min_child_samples":range(100,500,50),
            # "clf__colsample_bytree":np.linspace(0.6,1.0,5),
            # "clf__subsample":np.linspace(0.6,1.0,5),
            # 'clf__reg_alpha': [0.001, 0.01,0.1],
            # 'clf__reg_lambda': [0.001, 0.01, 0.1]
                   }]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, n_jobs=-1,scoring="accuracy")
    gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    print(gs.best_params_)
    print(gs.best_score_)
    cvres = gs.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)
    return clf

def print_feature_order(clf,feat_labels):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]))

X,y = df.iloc[:,:-1].values,df.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# sc = StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.fit_transform(X_test)
r = np.random.permutation(len(y_train))
X_train = X_train[r]
y_train = y_train[r]
lf = lgb.LGBMClassifier()
# lf = xgb.XGBClassifier()
# lf = RandomForestClassifier()
# lf = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
# lf = SVC(kernel='rbf',C=1.0,random_state=0)
# lf = LogisticRegression(random_state=0)
# lf = VotingClassifier(estimators=[
#     ('lgb', lgb.LGBMClassifier()),
#     ('xgb', xgb.XGBClassifier()),
#     ('rdf', RandomForestClassifier()),
    # ('knn', KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')),
    # ('svc', SVC(kernel='rbf',C=1.0,random_state=0)),
    # ('lr', LogisticRegression(random_state=0))
    # ],voting='hard')
lf = Pipeline([('scl',StandardScaler()),("clf",lf)])
# lf = search_params(lf,X_train,y_train)
lf.fit(X_train,y_train)
# print_feature_order(lf._final_estimator,use_feature)
print("train set accuracy is",lf.score(X_train,y_train))
print("test set accuracy is",lf.score(X_test,y_test))
# print("pre is",precision_score(y_test,lf.predict(X_test)))
# print("recall is",recall_score(y_test,lf.predict(X_test)))
# print("fi score is",f1_score(y_test,lf.predict(X_test)))
# print("auc is",roc_auc_score(y_true=y_test,y_score=lf.predict(X_test)))
# print("lift is",precision_score(y_test,lf.predict(X_test))/np.sum(y_test)*len(y_test))
# sample = [[1,1,1,1,1,1]]
# print(lf.predict_proba(sample))

# joblib.dump(lf, "lgb.pkl")

def show_learning_curve(clf,X_train,y_train):
    train_sizes,train_scores,test_scores = learning_curve(estimator=clf,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=10,n_jobs=1)
    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)
    plt.plot(train_sizes,train_mean,color="blue",marker='o',markersize=5,label='training accuracy')
    plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color="blue")
    plt.plot(train_sizes,test_mean,color="green",marker='s',markersize=5,label='validation accuracy',linestyle="--")
    plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color="green")
    plt.grid()
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.8,1.0])
    plt.show()

def show_valicate_curve(clf,X_train,y_train,param_name,param_range):
    train_scores, test_scores = validation_curve(estimator=clf, X=X_train, y=y_train,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=10,n_jobs=-1,scoring="accuracy")
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color="blue", marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    plt.plot(param_range, test_mean, color="green", marker='s', markersize=5, label='validation accuracy',
             linestyle="--")
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
    plt.grid()
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    # plt.ylim([0.8, 1.0])
    plt.show()

def show_roc_curve(clf,X_train,y_train):
    cv = StratifiedKFold(n_splits=3, random_state=1)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        probas = clf.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label="ROC fold %d (area = %0.2f)" % (i + 1, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", color=(0.6, 0.6, 0.6), label="random guessing")
    mean_tpr /= len(list(cv.split(X_train, y_train))) # 多折的平均值，用于画平均线，其实用cross_val_predict得到全部样本，再自己分割也行
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label="mean ROC (area=%0.2f)" % mean_auc, lw=2)
    plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=":", color="black", label="perfect performance")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("Receiver Operator Characteristic")
    plt.legend(loc="lower right")
    plt.show()

def show_precision_recall_curve(clf,X_train,y_train):
    y_probas_forest = cross_val_predict(clf, X_train, y_train, cv=3,method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_forest)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()

def show_ks_curve(clf,X_train,y_train):
    y_probas_forest = cross_val_predict(clf, X_train, y_train, cv=3,method="predict_proba")
    fpr, tpr, thresholds = roc_curve(y_train, y_probas_forest[:, 1], pos_label=1)
    plt.plot(thresholds, fpr, "b--", label="fpr")
    plt.plot(thresholds, tpr, "g-", label="tpr")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()

def ks_value(clf,X_test,y_test):
    y_probas_forest = cross_val_predict(clf, X_test, y_test, cv=3, method="predict_proba")
    fpr, tpr, thresholds = roc_curve(y_test, y_probas_forest[:, 1], pos_label=1)        # 在测试集上，根据之前预测结果和真实类标，来计算ks值
    return np.max(tpr-fpr)

# show_learning_curve(lf,X_train,y_train)
# show_valicate_curve(lf,X_train,y_train,"clf__max_depth",range(1,10,1))
# show_roc_curve(lf,X_train,y_train)
# show_precision_recall_curve(lf,X_train,y_train)
# show_ks_curve(lf,X_train,y_train)
print(ks_value(lf,X_test,y_test))


