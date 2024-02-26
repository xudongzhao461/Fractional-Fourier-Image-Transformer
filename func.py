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
import math
from torchsummary import summary
from IPython import display
import torch_optimizer as optim2
import time
import collections
from torch import optim



def load_dataset(Dataset,perclass):

    if Dataset == 'HU':
        mat_data = sio.loadmat('content/houston/Houston_Merge.mat')
        mat_gt = sio.loadmat('content/houston/houston15_gt.mat')
        data_hsi = mat_data['data']
        gt_hsi = mat_gt['gt']
        
        mat_mask_train =sio.loadmat('content/houston/Houston_mask_train_20.mat')#
        mask_train_hsi=mat_mask_train['mask_train']
        mat_mask_test =sio.loadmat('content/houston/houston15_gt')#mask_test.mat')#
        mask_test_hsi=mat_mask_test['gt']
        TOTAL_SIZE = 15029 #样本总数
        VALIDATION_SPLIT = 0.1

    if Dataset == 'TR':
        mat_data = sio.loadmat('content/trento/Italy_HSI_Lidar_Merge.mat')
        mat_gt = sio.loadmat('content/trento/mask_gt.mat')
        data_hsi = mat_data['data']
        gt_hsi = mat_gt['gt']
        
        mat_mask_train =sio.loadmat('content/trento/mask_train_italy20.mat')#
        mask_train_hsi=mat_mask_train['mask_train']
        mat_mask_test =sio.loadmat('content/trento/mask_gt')#mask_test.mat')#
        mask_test_hsi=mat_mask_test['gt']
        TOTAL_SIZE = 30214 #样本总数
        VALIDATION_SPLIT = 0.1

    if Dataset == 'MU':
        mat_data = sio.loadmat('content/muufl/MUUF_merge.mat')
        # mat_data = sio.loadmat('content/muufl/muuflNor.mat')
        mat_gt = sio.loadmat('content/muufl/gt.mat')
        data_hsi = mat_data['data']
        # data_hsi = mat_data['tar']
        gt_hsi = mat_gt['gt']
        
        mat_mask_train =sio.loadmat('content/muufl/mask_train_20.mat')
        mask_train_hsi=mat_mask_train['mask_train']
        mat_mask_test =sio.loadmat('content/muufl/gt.mat')
        mask_test_hsi=mat_mask_test['gt']
        TOTAL_SIZE = 53687 #样本总数
        VALIDATION_SPLIT = 0.1

    if Dataset == 'NA':
        mat_data = sio.loadmat('content/nashua/data/NashuaGSMnormal.mat')
        # mat_data = sio.loadmat('content/muufl/muuflNor.mat')
        mat_gt = sio.loadmat('content/nashua/data/NashuaGT.mat')
        data_hsi = mat_data['data']
        # data_hsi = mat_data['tar']
        gt_hsi = mat_gt['gt']
        
        mat_mask_train =sio.loadmat('content/nashua/data/Nmask_train_'+str(perclass)+'.mat')
        mask_train_hsi=mat_mask_train['mask_train']
        mat_mask_test =sio.loadmat('content/nashua/data/NashuaGT.mat')
        mask_test_hsi=mat_mask_test['gt']
        TOTAL_SIZE = 229447 #样本总数
        VALIDATION_SPLIT = 0.1

    return data_hsi, gt_hsi, TOTAL_SIZE, VALIDATION_SPLIT,mask_train_hsi,mask_test_hsi

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, val_indices,  VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt, img_rows):
    y_train_ori = gt[train_indices] - 1
    y_test = gt[test_indices] - 1
    y_val_ori =gt[val_indices] - 1

    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    y_train=np.zeros(4*TRAIN_SIZE)
    y_train[0:TRAIN_SIZE]=y_train_ori
    y_train[TRAIN_SIZE:2*TRAIN_SIZE]=y_train_ori
    y_train[2*TRAIN_SIZE:3*TRAIN_SIZE]=y_train_ori
    y_train[3*TRAIN_SIZE:4*TRAIN_SIZE]=y_train_ori

    # test_data =  select_small_cubic_ori(TEST_SIZE, test_indices, whole_data,
    #                                                    PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    # np.save('testcube'+str(img_rows)+'.npy',test_data)
    test_data = np.load('testcube'+str(img_rows)+'.npy')
    val_data =  select_small_cubic(VAL_SIZE, val_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    y_val=np.zeros(4*VAL_SIZE)
    y_val[0:VAL_SIZE]=y_val_ori
    y_val[VAL_SIZE:2*VAL_SIZE]=y_val_ori
    y_val[2*VAL_SIZE:3*VAL_SIZE]=y_val_ori
    y_val[3*VAL_SIZE:4*VAL_SIZE]=y_val_ori
    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    x_val = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], INPUT_DIMENSION)
    
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=0, 
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=0, 
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False, 
        num_workers=0, 
    )

    return train_iter, valiada_iter, test_iter



def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    """
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))

def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes
    
def samplingtest(ground_truth):
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        labels_loc[i] = indexes
        nb_val = 0
        test[i] = indexes[nb_val:]
    test_indexes = []
    for i in range(m):
        test_indexes += test[i]
    return test_indexes


def set_figsize(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([255, 128, 0]) / 255.
        if item == 14:
            y[index] = np.array([255, 255, 255]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y



def generate_png(all_iter, net, model_vit, gt_hsi, Dataset, device, total_indices):
    pred_test = []
    for X, y in all_iter:
        #X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        model_vit.eval()
        y_hat = net(X)
        y_hat = model_vit(y_hat)
        pred_test.extend(y_hat.cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    print('------Get classification maps successful-------')
    return y_re

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic_ori(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((4*data_size, (2 * patch_length + 1), (2 * patch_length + 1), dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        tmph=select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
        
        small_cubic_data[i] = tmph       
        small_cubic_data[i+data_size] = np.flip(tmph, axis=0)
        noise = np.random.normal(0.0, 0.01, size=tmph.shape)
        small_cubic_data[i+2*data_size] = np.flip(tmph + noise, axis=1)
        k = np.random.randint(4)
        small_cubic_data[i+3*data_size] = np.rot90(tmph, k=k)
    return small_cubic_data

def generate_testiter( TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    y_test = gt[test_indices] - 1

    test_data =  select_small_cubic_ori(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    print(test_data.shape)

    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)


    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False, 
        num_workers=0, 
    )

    return test_iter#, all_iter #, y_test


def evaluate_accuracy(data_iter, net, model_vit, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            #X = X.permute(0, 3, 1, 2)
            X = X.to(device)
            y = y.to(device)
            net.eval()
            model_vit.eval()
            y_hat = net(X)
            y_hat = model_vit(y_hat)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train()
            model_vit.train()
            n += y.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, confusion_matrix, path):
    f = open(path, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)
    sentence10 = "The diagonal Confusion matrix: " + '\n' + str(confusion_matrix) + '\n'
    f.write(sentence10)
    sentence11 = "The element_acc_ae: " + '\n' + str(element_acc_ae) + '\n'
    f.write(sentence11)
    f.close()



def train(net, model_vit, train_iter, valida_iter, loss, optimizer, device, epochs, early_stopping=True,
          early_num=20):
    loss_list = [100]


    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X, y in train_iter:
            
            batch_count, train_l_sum = 0, 0
            #X = X.permute(0, 3, 1, 2)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = model_vit(y_hat)
            # print('y_hat:', y_hat.shape)
            # print('y:', y.shape)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, model_vit, loss, device)
        loss_list.append(valida_loss)

        
        train_loss_list.append(train_l_sum) # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, valida_acc, time.time() - time_epoch))

        # PATH = "./net_DBA.pt"
        # if loss_list[-1] <= 0.01 and valida_acc >= 0.95:
        #     torch.save(net.state_dict(), PATH)
        #     break

        # if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
        #     if early_epoch == 0: # and valida_acc > 0.9:
        #         torch.save(net.state_dict(), PATH)
        #     early_epoch += 1
        #     loss_list[-1] = loss_list[-2]
        #     if early_epoch == early_num:
        #         net.load_state_dict(torch.load(PATH))
        #         break
        # else:
        #     early_epoch = 0

    
    # set_figsize()
    # plt.figure(figsize=(8, 8.5))
    # train_accuracy =   plt.subplot(221)
    # train_accuracy.set_title('train_accuracy')
    # plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    # plt.xlabel('epoch')
    # plt.ylabel('train_accuracy')
    
    # test_accuracy =   plt.subplot(222)
    # test_accuracy.set_title('valida_accuracy')
    # plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    # plt.xlabel('epoch')
    # plt.ylabel('test_accuracy')

    # loss_sum =   plt.subplot(223)
    # loss_sum.set_title('train_loss')
    # plt.plot(np.linspace(1, epoch, len(train_loss_list)), train_loss_list, color='red')
    # plt.xlabel('epoch')
    # plt.ylabel('train loss')
    # # ls_plot = np.array(ls_plot)

    # test_loss =   plt.subplot(224)
    # test_loss.set_title('valida_loss')
    # plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    # plt.xlabel('epoch')
    # plt.ylabel('valida loss')
    # # ls_plot = np.array(ls_plot)

    # plt.show()
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))