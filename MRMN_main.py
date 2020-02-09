#encoding:utf-8
import sys
import random

opt_num = 0
f_i1 = open("../data/uin_train_data")
train_data = []

for line in f_i1:
    new_line = line.strip("\n").split("\t")
    cur_data = []
    cur_data.append(int(new_line[0]))
    cur_data.append(int(new_line[1]))
    cur_data.append(int(new_line[2]))
    if new_line[-1] == "1":
        lamda_data = [0.25, -1000.0, -1000.0,-1000.0,-1000.0]
        #train_data.append(cur_data)
    if new_line[-1] == "2":
        lamda_data = [-1000.0,0.20,-1000.0,-1000.0,-1000.0]
    if new_line[-1] == "3":
        lamda_data = [-1000.0,-1000.0,0.15,-1000.0,-1000.0]
    if new_line[-1] == "4":
        lamda_data = [-1000.0,-1000.0,-1000.0,0.1,-1000.0]
    if new_line[-1] == "5":
        lamda_data = [-1000.0,-1000.0,-1000.0,-1000.0,0.05]
    cur_data.append(lamda_data)
    train_data.append(cur_data)
    opt_num += 1
    if opt_num % 10000 == 0:
        print (opt_num)
f_i1.close()

random.shuffle(train_data)

n_sample = len(train_data)


from MRMN_model import MRMN
import tensorflow as tf

class args:
    std = 0.1
    num_mem = 100
    embedding_size = 50
    type_size = 4
    constraint = True
    rnn_type = set(['PAIR'])
    margin = 0.2
    l2_reg = 0.001
    opt = 'SGD'
    clip_norm = 2
    dropout = 0.7
    learn_rate = 0.01
ar = args()

user_num = 86415
item_num = 26808

model = MRMN(user_num,item_num,ar)
model._build_list_network()
saver = tf.train.get_or_create_global_step()
init = tf.global_variables_initializer()
# Launch the graph.
sess = tf.Session()
sess.run(init)

print ("build finish")
batch_size = 64
batch_num = int((n_sample +  batch_size - 1)/batch_size)
print ("batch_num",batch_num)
Iter = 50

import pickle

for it in range(Iter):
    for i in range(batch_num):
        beg = i*batch_size
        end = min((i + 1)*batch_size,n_sample)
        cur_train_data = train_data[beg:end]
        feed_dict = model.get_list_feed_dict(cur_train_data)
        loss,_ = sess.run([model.cost,model.train_op],feed_dict)
        if i % 10 == 0:
            print (it,i,loss)
            
    if it % 2 == 0:
        i3 = 0
        f_i3 = open("../data/uin_test_data")
        test_pos_data = []
        for line in f_i3:
            new_line = line.strip("\n").split("\t")
            uin = int(new_line[0])
            item =int(new_line[1])
            label = int(new_line[2])
            test_pos_data.append([uin,item,label])
            i3 += 1
        f_i3.close()
        
        n_test_sample = len(test_pos_data)
        batch_size = 64
        test_batch_num = int((n_test_sample +  batch_size - 1)/batch_size)

        f_o3 = open("MRMN_ret/opt_test_score_out_" + str(it),"w")
        for i in range(test_batch_num):
            beg =  i*batch_size
            end = min((i + 1)*batch_size,n_test_sample)
            ops_batch = test_pos_data[beg:end]
            feed_dict = model.get_list_feed_dict(ops_batch,"")
            scores = sess.run(model.predict_op,feed_dict)
            for j in range(len(ops_batch)):
                uid = ops_batch[j][0]
                pid = ops_batch[j][1]
                ss = scores[j][0]
                f_o3.write(str(uid) + "\t" + str(pid) + "\t" + str(ss) + "\n")
        f_o3.close()
        
        i4 = 0
        f_i4 = open("../data/uin_test_neg_data")
        test_neg_data = []
        for line in f_i4:
            new_line = line.strip("\n").split("\t")
            uin = int(new_line[0])
            item =int(new_line[1])
            label = int(new_line[2])
            test_neg_data.append([uin,item,label])
            i4 += 1
        f_i4.close()
        
        n_test_sample = len(test_neg_data)
        batch_size = 64
        test_batch_num = int((n_test_sample +  batch_size - 1)/batch_size)

        f_o4 = open("MRMN_ret/neg_test_score_out_" + str(it),"w")
        for i in range(test_batch_num):
            beg =  i*batch_size
            end = min((i + 1)*batch_size,n_test_sample)
            ops_batch = test_neg_data[beg:end]
            feed_dict = model.get_list_feed_dict(ops_batch,"")
            scores = sess.run(model.predict_op,feed_dict)
            for j in range(len(ops_batch)):
                uid = ops_batch[j][0]
                pid = ops_batch[j][1]
                ss = scores[j][0]
                f_o4.write(str(uid) + "\t" + str(pid) + "\t" + str(ss) + "\n")
        f_o4.close()
        
i3 = 0
f_i3 = open("../data/uin_test_data")
test_pos_data = []
for line in f_i3:
    new_line = line.strip("\n").split("\t")
    uin = int(new_line[0])
    item =int(new_line[1])
    label = int(new_line[2])
    test_pos_data.append([uin,item,label])
    i3 += 1
f_i3.close()

n_test_sample = len(test_pos_data)
batch_size = 64
test_batch_num = int((n_test_sample +  batch_size - 1)/batch_size)

f_o3 = open("MRMN_ret/opt_test_score_out_" + str(Iter),"w")
for i in range(test_batch_num):
    beg =  i*batch_size
    end = min((i + 1)*batch_size,n_test_sample)
    ops_batch = test_pos_data[beg:end]
    feed_dict = model.get_list_feed_dict(ops_batch,"")
    scores = sess.run(model.predict_op,feed_dict)
    for j in range(len(ops_batch)):
        uid = ops_batch[j][0]
        pid = ops_batch[j][1]
        ss = scores[j][0]
        f_o3.write(str(uid) + "\t" + str(pid) + "\t" + str(ss) + "\n")
f_o3.close()

i4 = 0
f_i4 = open("../data/uin_test_neg_data")
test_neg_data = []
for line in f_i4:
    new_line = line.strip("\n").split("\t")
    uin = int(new_line[0])
    item =int(new_line[1])
    label = int(new_line[2])
    test_neg_data.append([uin,item,label])
    i4 += 1
f_i4.close()

n_test_sample = len(test_neg_data)
batch_size = 64
test_batch_num = int((n_test_sample +  batch_size - 1)/batch_size)

f_o4 = open("MRMN_ret/neg_test_score_out_" + str(Iter),"w")
for i in range(test_batch_num):
    beg =  i*batch_size
    end = min((i + 1)*batch_size,n_test_sample)
    ops_batch = test_neg_data[beg:end]
    feed_dict = model.get_list_feed_dict(ops_batch,"")
    scores = sess.run(model.predict_op,feed_dict)
    for j in range(len(ops_batch)):
        uid = ops_batch[j][0]
        pid = ops_batch[j][1]
        ss = scores[j][0]
        f_o4.write(str(uid) + "\t" + str(pid) + "\t" + str(ss) + "\n")
f_o4.close()
