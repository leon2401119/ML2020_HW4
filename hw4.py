from liblinearutil import *
import numpy as np
import os

TRAIN_PATH = f'/home/yangck/Desktop/liblinear-2.42/python/hw4_train.dat'
TEST_PATH = f'/home/yangck/Desktop/liblinear-2.42/python/hw4_test.dat'

def load_data(PATH):

    train_dataset = []

    with open(PATH) as f:
        while True:
            input = f.readline()
            if not len(input):
                return train_dataset

            split_input = input[:-1].split(' ')

            for id in range(len(split_input)):
                split_input[id] = float(split_input[id])


            train_dataset.append([np.array(split_input[:-1],dtype='float64'),split_input[-1]])



def feature_transform(dataset):
    for i, data in enumerate(dataset):
        temp = np.append(np.array([1.0],dtype='float64'),data[0])
        for i in range(0,6):
            for j in range(i,6):
                temp = np.append(temp, data[0][i]*data[0][j])

        data[0] = temp



dataset = load_data(TRAIN_PATH)
test_dataset = load_data(TEST_PATH)
feature_transform(dataset)
feature_transform(test_dataset)



## 16, 17
# y = [data[1] for data in dataset]
# x = [data[0] for data in dataset]
# test_y = [data[1] for data in test_dataset]
# test_x = [data[0] for data in test_dataset]
# prob = problem(y, x)
# param = parameter('-s 0 -c 0.00005 -e 0.000001 -q')
# model = train(prob, param)
# p_labs, p_acc, p_vals = predict(test_y, test_x, model)


## 18
# train_y = []
# train_x = []
# eval_y = []
# eval_x = []
# for i in range(120):
#     train_y.append(dataset[i][1])
#     train_x.append(dataset[i][0])
#
# for i in range(120,len(dataset)):
#     eval_y.append(dataset[i][1])
#     eval_x.append(dataset[i][0])
#
#
# prob = problem(train_y, train_x)
# param = parameter('-s 0 -c 0.00005 -e 0.000001 -q')
# model = train(prob, param)
# p_labs, p_acc, p_vals = predict(eval_y, eval_x, model)


## 19
# y = [data[1] for data in dataset]
# x = [data[0] for data in dataset]
# test_y = [data[1] for data in test_dataset]
# test_x = [data[0] for data in test_dataset]
# prob = problem(y, x)
# param = parameter('-s 0 -c 50 -e 0.000001 -q')
# model = train(prob, param)
# p_labs, p_acc, p_vals = predict(test_y, test_x, model)


## 20
error = 0
for j in range(5):
    train_y = []
    train_x = []
    eval_y = []
    eval_x = []
    val_list = list(range(j*40,j*40+40))
    for i in range(200):
        if i in val_list:
            eval_y.append(dataset[i][1])
            eval_x.append(dataset[i][0])
        else:
            train_y.append(dataset[i][1])
            train_x.append(dataset[i][0])

    prob = problem(train_y, train_x)
    param = parameter('-s 0 -c 0.00005 -e 0.000001 -q')
    model = train(prob, param)
    p_labs, p_acc, p_vals = predict(eval_y, eval_x, model)
    error += (100-p_acc[0])

print(error/5)
