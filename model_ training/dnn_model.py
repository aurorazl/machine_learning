import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout
import matplotlib.pyplot as plt

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
df = pd.read_csv("./data_100000.csv")
df = df.reindex(columns=use_feature+["label"])
df = df.fillna(0)
X,y = df.iloc[:,:-1].values,df.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

n_inputs = len(use_feature)
n_hidden1 = 500
n_hidden2 = 100
n_outputs=2
X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
Y = tf.placeholder(tf.int64,shape=(None),name='Y')
is_training = tf.placeholder(tf.bool,shape=(),name="is_training")

he_init = tf.contrib.layers.variance_scaling_initializer()

keep_prob = 0.5
with tf.name_scope("dnn"):
    with tf.contrib.framework.arg_scope([fully_connected],normalizer_fn=batch_norm):
        x_drop = dropout(X, keep_prob, is_training=is_training)
        hidden1 = fully_connected(x_drop,n_hidden1,scope='hidden1',weights_initializer=he_init,activation_fn=tf.nn.elu)
        hidden1_drop = dropout(hidden1,keep_prob, is_training=is_training)
        hidden2 = fully_connected(hidden1_drop,n_hidden2,scope='hidden2',weights_initializer=he_init,activation_fn=tf.nn.elu)
        hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
        logits = fully_connected(hidden2_drop,n_outputs,activation_fn=None,scope='outputs')

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate=0.01

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,Y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 50
batch_size = 200
data = []
data_2 = []

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        mini = np.array_split(range(y_train.shape[0]),batch_size)
        for idx in mini:
            X_batch,y_batch = X_train[idx],y_train[idx]
            sess.run(training_op, feed_dict={X: X_batch, Y: y_batch,is_training:True})
        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: y_batch,is_training:False})
        acc_test = accuracy.eval(feed_dict={X: X_test,Y: y_test,is_training:False})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        data.append(acc_train)
        data_2.append(acc_test)
    # save_path = saver.save(sess, "./my_model_final.ckpt")

plt.plot(range(n_epochs),data,color='blue',marker='o',label="train")
plt.plot(range(n_epochs),data_2,color='red',marker='s',label="test")
plt.legend(loc="upper left")
plt.show()