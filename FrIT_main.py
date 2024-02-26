import numpy as np
import torch
from operator import truediv
import torch.utils.data as Data
from sklearn import metrics, preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import cv2
from IPython import display
from modelF import *
from func import *
import math
from torchsummary import summary
from IPython import display
import torch_optimizer as optim2
import time
import collections
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

global Dataset  
dataset = 'NA' 
perclass= 20
PATCH_LENGTH = 4
vit_layer=6


Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, VALIDATION_SPLIT,mask_train_hsi,mask_test_hsi= load_dataset(Dataset,perclass)
# mat_mask_test1 =sio.loadmat('content/linye/gt2.mat')
# mask_test_hsi1=mat_mask_test1['gt']
# mat_mask_test2 =sio.loadmat('content/linye/gt3.mat')
# mask_test_hsi2=mat_mask_test2['gt']
# mat_mask_test3 =sio.loadmat('content/linye/gt4.mat')
# mask_test_hsi3=mat_mask_test3['gt']

data_hsi = samele_wise_normalization(data_hsi)
data_hsi = sample_wise_standardization(data_hsi)
print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
mask_test = mask_test_hsi.reshape(np.prod(mask_test_hsi.shape[:2]),)
# mask_test1 = mask_test_hsi1.reshape(np.prod(mask_test_hsi1.shape[:2]),)
# mask_test2 = mask_test_hsi2.reshape(np.prod(mask_test_hsi2.shape[:2]),)
# mask_test3 = mask_test_hsi3.reshape(np.prod(mask_test_hsi3.shape[:2]),)
mask_train =mask_train_hsi.reshape(np.prod(mask_train_hsi.shape[:2]),)
CLASSES_NUM = max(mask_train)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 1
 #窗口半径
# for PATCH_LENGTH in range(5):
#     PATCH_LENGTH = PATCH_LENGTH+1 #窗口半径

lr, num_epochs, batch_size = 0.0001, 20, 32
loss = torch.nn.CrossEntropyLoss()
vit_layer=6

img_rows = 2*PATCH_LENGTH+1 #patch大小
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                        'constant', constant_values=0)

# train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
_, val_indices    = sampling(1,mask_train)
_, train_indices  = sampling(1,mask_train)
_, test_indices   = sampling(1,mask_test) 
# _, test_indices1   = sampling(1,mask_test1) 
# _, test_indices2   = sampling(1,mask_test2) 
# _, test_indices3  = sampling(1,mask_test3) 
_, total_indices  = sampling(1, gt)

TRAIN_SIZE = len(train_indices)
print('Train size: ', TRAIN_SIZE)
TEST_SIZE = len(test_indices)
print('Test size: ', TEST_SIZE)
# TEST_SIZE1 = len(test_indices1)
# print('Test size1: ', TEST_SIZE1)
# TEST_SIZE2 = len(test_indices2)
# print('Test size2: ', TEST_SIZE2)
# TEST_SIZE3 = len(test_indices3)
# print('Test size3: ', TEST_SIZE3)
VAL_SIZE = len(val_indices)
print('Validation size: ', VAL_SIZE)

print('-----Selecting Small Pieces from the Original Cube Data-----')
train_iter, valida_iter, test_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, val_indices, TEST_SIZE, total_indices, VAL_SIZE,whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, 16, gt) #batchsize in 1
# test_iter1 = generate_testiter(TEST_SIZE1, test_indices1, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, 16, gt) #batchsize in 1
# test_iter2 = generate_testiter(TEST_SIZE2, test_indices2, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, 16, gt) #batchsize in 1
# test_iter3 = generate_testiter(TEST_SIZE3, test_indices3, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, 16, gt) #batchsize in 1


net = SSRN_network(BAND, CLASSES_NUM).cuda()
# model = HYSN(BAND, CLASSES_NUM).cuda()
# summary(net, (1,9,9,200))
# print(model.children)

# model_vit = ViT(
#     image_size = 7,   #输入特征的窗口大小
#     patch_size = 1,   #裁成patch的大小
#     num_classes = CLASSES_NUM,
#     dim = 1024,       #trans的维度
#     depth = vit_layer,        #trans的深度
#     heads = 15,       #trans的头数  
#     mlp_dim = 2048,   #TRANS
#     channels = 24,    #输入特征的维度
#     dropout = 0.1,
#     emb_dropout = 0.1
# ).cuda()
# summary(model_vit,(24,7,7))
# print(model_vit.children)
# for PATCH_LENGTH in range(5):
#     PATCH_LENGTH = PATCH_LENGTH+1 #窗口半径
for index_iter in range(ITER):
    print('iter:', index_iter)
    model_vit = ViT(
        image_size = img_rows,   #输入特征的窗口大小
        patch_size = 1,   #裁成patch的大小
        num_classes = CLASSES_NUM,
        dim = 1024,       #trans的维度
        depth = vit_layer,        #trans的深度
        mlp_dim = 2048,   #TRANS
        channels = 24,    #输入特征的维度
        dropout = 0.1,
        emb_dropout = 0.1
    ).cuda()
    # summary(model_vit,(24,7,7))
    # print(model_vit.children)
    # net = SSRN_network(BAND, CLASSES_NUM)
    # net = HYSN(BAND, CLASSES_NUM)
    #optimizer = optim2.DiffGrad(net.parameters(), lr=lr, amsgrad=False) #, weight_decay=0.0001)
    optimizer = torch.optim.Adam(list(net.parameters())+list(model_vit.parameters()), lr= lr, betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    # # train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    # _, val_indices    = sampling(1,mask_train)
    # _, train_indices  = sampling(1,mask_train)
    # _, test_indices   = sampling(1,mask_test) 
    # _, total_indices  = sampling(1, gt)

    # TRAIN_SIZE = len(train_indices)
    # print('Train size: ', TRAIN_SIZE)
    # TEST_SIZE = len(test_indices)
    # print('Test size: ', TEST_SIZE)
    # VAL_SIZE = len(val_indices)
    # print('Validation size: ', VAL_SIZE)

    # print('-----Selecting Small Pieces from the Original Cube Data-----')
    # train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, val_indices, TOTAL_SIZE, total_indices, VAL_SIZE,whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, 16, gt) #batchsize in 1

    tic1 = time.process_time()
    train(net, model_vit, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.process_time()

    pred_test = []
    tic2 = time.process_time()
    with torch.no_grad():
        for X, y in test_iter:
            #X = X.permute(0, 3, 1, 2)
            X = X.to(device)
            net.eval()
            model_vit.eval()
            y_hat = net(X)
            y_hat = model_vit(y_hat)
            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
    toc2 = time.process_time()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1


    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)

    # torch.save(net.state_dict(), "content/linye/" + str(round(overall_acc, 3)) +Dataset+ '15vitlayer'+str(vit_layer)+'.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc

    print("--------" + " Training Finished-----------")
    record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,confusion_matrix,
                        'content/nashua/fft_'  + str(img_rows) + '_' + Dataset + 'fft20vit_layer' + str(vit_layer) + 'lr：' + str(lr) + '.txt')
    print('OA %4f, KAPPA %.4f, AA %.3f' % (overall_acc, kappa, average_acc))

    y_re=generate_png(test_iter, net, model_vit, gt_hsi, Dataset, device, test_indices)
    # y_re1=generate_png(test_iter1, net, model_vit, gt_hsi, Dataset, device, test_indices1)
    # y_re2=generate_png(test_iter2, net, model_vit, gt_hsi, Dataset, device, test_indices2)
    # y_re3=generate_png(test_iter3, net, model_vit, gt_hsi, Dataset, device, test_indices3)
    sio.savemat('content/nashua/fft_patch' +  str(img_rows) + '_'+Dataset + '_sample'+str(20) +str(vit_layer) +'.mat',{'map1':y_re})
    # sio.savemat('content/nashua/' +  str(img_rows) + '_'+Dataset + 'fft_sample20_layer' +str(vit_layer) +'.mat',{'map1':y_re,'map2':y_re1,'map3':y_re2,'map4':y_re3})