from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader
from Network import Encoder
import torch.optim
from sklearn.metrics import confusion_matrix
import pandas as pd
from LPP_func import *
import matplotlib.ticker as ticker
import random

# 构造训练数据和测试数据
def creat_dataset(testindex, attribute_matrix1):
    path = './OGW_mat_data/'
    data1 = loadmat(path + 'data_ow_bubble.mat')['data_ow_bubble']
    data2 = loadmat(path + 'data_ow_plug.mat')['data_ow_plug']
    data3 = loadmat(path + 'data_ow_slug.mat')['data_ow_slug']
    data4 = loadmat(path + 'data_ow_wave.mat')['data_ow_wave']
    data5 = loadmat(path + 'data_ow_st.mat')['data_ow_st']
    data6 = loadmat(path + 'data_ow_ann.mat')['data_ow_ann']
    data7 = loadmat(path + 'data_wo_bubble.mat')['data_wo_bubble']
    data8 = loadmat(path + 'data_wo_plug.mat')['data_wo_plug']
    data9 = loadmat(path + 'data_wo_slug.mat')['data_wo_slug']
    data10 = loadmat(path + 'data_wo_ann.mat')['data_wo_ann']
    attribute_matrix = attribute_matrix1.values

    # 构造训练数据
    traindata = np.row_stack([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10])
    n1 = data1.shape[0]
    attribute1 = [attribute_matrix[0, :]] * n1
    n2 = data2.shape[0]
    attribute2 = [attribute_matrix[1, :]] * n2
    n3 = data3.shape[0]
    attribute3 = [attribute_matrix[2, :]] * n3
    n4 = data4.shape[0]
    attribute4 = [attribute_matrix[3, :]] * n4
    n5 = data5.shape[0]
    attribute5 = [attribute_matrix[4, :]] * n5
    n6 = data6.shape[0]
    attribute6 = [attribute_matrix[5, :]] * n6
    n7 = data7.shape[0]
    attribute7 = [attribute_matrix[6, :]] * n7
    n8 = data8.shape[0]
    attribute8 = [attribute_matrix[7, :]] * n8
    n9 = data9.shape[0]
    attribute9 = [attribute_matrix[8, :]] * n9
    n10 = data10.shape[0]
    attribute10 = [attribute_matrix[9, :]] * n10
    train_attributelabel = np.row_stack(
    [attribute1, attribute2, attribute3, attribute4, attribute5, attribute6, attribute7, attribute8, attribute9,
     attribute10])  # 属性矩阵

    label1 = [[0]] * n1
    label2 = [[1]] * n2
    label3 = [[2]] * n3
    label4 = [[3]] * n4
    label5 = [[4]] * n5
    label6 = [[5]] * n6
    label7 = [[6]] * n7
    label8 = [[7]] * n8
    label9 = [[8]] * n9
    label10 = [[9]] * n10
    trainlabel = np.row_stack([label1, label2, label3, label4, label5, label6, label7, label8, label9, label10])

    # 典型状态测试数据
    if testindex == 'data_ogw_test':
        Tdata1 = loadmat(path + testindex +'.mat')[testindex]
        testdata = np.row_stack([Tdata1])
        testdata = testdata.reshape(10,490,7)
        testdata = testdata[0:10,250:450,0:7]
        testdata = testdata.reshape(2000, 7)
        label1 = [[1]] * 200
        label2 = [[2]] * 200
        label3 = [[3]] * 200
        label4 = [[4]] * 200
        label5 = [[5]] * 200
        label6 = [[6]] * 200
        label7 = [[7]] * 200
        label8 = [[8]] * 200
        label9 = [[9]] * 200
        label10 = [[10]] * 200
        testlabel = np.row_stack([label1, label2, label3, label4, label5, label6, label7, label8, label9, label10])
        return traindata, trainlabel, train_attributelabel, testdata, testlabel

    # 过渡状态测试数据
    else:
        Tdata1 = loadmat(path + testindex +'.mat')[testindex]
        testdata = np.row_stack([Tdata1])
        testdata = testdata[0*490:7*490,:]
        testdata = testdata.reshape(7,490,7)
        testdata = testdata[0:7,150:250,0:7]    # 每组从490采样点中取100采样点
        testdata = testdata.reshape(700, 7)
        return traindata, trainlabel, train_attributelabel, testdata

# 数据标准化
def center_data(X):
    X_means = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    X_norm = (X - X_means[:, np.newaxis])/X_std[:, np.newaxis]
    return X_norm, X_means, X_std

# 特征-属性映射
def pre_attribute_model(model, traindata, train_attributelabel, testdata):
    print('Attribute prediction model: '+model)
    model_dict = {'SVR': SVR(kernel='rbf'), 'rf': RandomForestRegressor(n_estimators=200),
                  'Ridge': Ridge(alpha=1), 'Lasso': Lasso(alpha=0.1)}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    return test_pre_attribute

# 属性-状态映射
def pre_label(test_pre_attribute,attribute_matrix1):
    attribute_matrix1 = pd.DataFrame(attribute_matrix1.values)
    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix1.values - pre_res), axis=1)).argmin()
        label_lis.append(attribute_matrix1.index[loc]+1)
    label_lis = np.array(np.row_stack(label_lis))

    return label_lis


# main
if __name__ == '__main__':
    '''初始化'''
    random.seed(42)
    Trainflag = 0   # 是否需要训练, 0-不训练, 1-训练
    testindex = 'data_transition_3'     # data_ogw_test, data_transition_1, data_transition_3

    print("==========================[Test case] "+str(testindex)+"==========================")
    print("Data generating...")
    attribute_matrix = pd.read_excel('./OGW_attribute_new.xlsx', index_col='no')
    if testindex == 'data_ogw_test':
        traindata, trainlabel, train_attributelabel, testdata, testlabel = creat_dataset(testindex,attribute_matrix)
    else:
        traindata, trainlabel, train_attributelabel, testdata = creat_dataset(testindex, attribute_matrix)

    '''Training'''
    # 数据标准化
    traindata, X_means, X_std = center_data(traindata.T)
    traindata = traindata.T
    testdata = (testdata.T - X_means[:, np.newaxis])/X_std[:, np.newaxis]
    testdata = testdata.T

    # 数据-特征提取
    print("Feature extracting...")
    num1 = traindata.shape[0]
    num2 = traindata.shape[1]
    X_train = traindata
    Y_train = train_attributelabel
    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    # 网络训练, Xp & Xf共两个网络
    net1 = Encoder()
    net2 = Encoder()
    net_trained = Encoder()
    # 设置超参数
    epochs = 300
    learning_rate = 0.001
    batch_size = 64
    para_corr = 0.5
    para_lpp = 0.01
    para_cov = 0.5
    para_error = 1

    if Trainflag == 1:
        print("================= AgMRDCVA Training =================")
        criterion = torch.nn.L1Loss()
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 循环训练
        loss_history = []
        cv = []
        I = torch.eye(num2)
        for epoch in range(epochs):
            for data in train_loader:
                X_data, Y_data = data[0], data[1]
                Xp, Y1 = X_data[:batch_size // 2], Y_data[:batch_size // 2]
                Xf = X_data[batch_size // 2:]

                # Calculate LPP
                Xp_np = Xp.numpy()  # Convert to NumPy for cal_L function if needed
                L = cal_L(Xp_np, n_neighbors=5)

                # Forward pass
                output1, output_attri = net1(Xp)
                output2, attri2 = net2(Xf)

                # Calculate M_LPP
                M_LPP = torch.matmul(torch.matmul(output1.detach().t(), torch.FloatTensor(L)), output1.detach())

                # Calculate correlation matrix
                corr = np.corrcoef(output1.detach().numpy(), output2.detach().numpy())
                corr_pf = torch.FloatTensor(corr[:batch_size // 2, batch_size // 2:])

                # Calculate losses
                loss_error = criterion(output_attri, Y1)
                Cov = torch.matmul(output1.t(), output1) / (batch_size // 2)
                loss_cov = torch.norm(Cov - I, p="fro")
                loss_corr = 1 / (torch.diag(corr_pf).norm(p="fro"))
                loss_lpp = torch.norm(M_LPP, p="fro")

                # Total loss
                loss = para_cov * loss_cov + para_corr * loss_corr + para_lpp * loss_lpp + para_error * loss_error

                # Optimization step
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()

            if epoch % 1 == 0:
                print(
                    "Epoch number {}\n Loss_corr {}\n Loss_cov {}\n Loss_lpp {}\n Loss_all {}\n".format(
                        epoch, loss_corr.item(), loss_cov.item(), loss_lpp.item(), loss.item()))
                loss_history.append(loss.item())

        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Training Loss', color='blue')
        plt.title('Training Loss over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        torch.save(net1.state_dict(), 'weights/AgMRDCVA.pkl')       # 保存网络权重
    else:
        print("No training!")
        pass

    '''Testing'''
    # 加载网络权重
    net_trained.load_state_dict(torch.load(r'.\weights\AgMRDCVA.pkl'))

    # 加载测试数据
    num1 = testdata.shape[0]
    num2 = testdata.shape[1]
    X_test = testdata
    X_test = torch.FloatTensor(X_test)

    output1, output_attri = net_trained(X_test)
    output1_train, output_attri_train = net_trained(X_train)
    trainfeature = output1_train.detach().numpy()
    testfeature = output1.detach().numpy()

    # 预测属性
    print("Attribute predicting...")  # 包含已知类属性
    test_pre_attribute = pre_attribute_model('rf', trainfeature, train_attributelabel,
                                             testfeature)
    test_pre_attribute = np.array(test_pre_attribute)

    if testindex == 'data_ogw_test':
        print("Identification of typical flow states...")
        label_lis = pre_label(test_pre_attribute, attribute_matrix)
        test_pre_attribute = np.array(test_pre_attribute)
        Mean_attribute = np.average(test_pre_attribute, axis=0)

        C = confusion_matrix(testlabel, label_lis)/200
        Accuracy_aver = np.average(np.diag(C))
        print('Overall accuracy：' + str(Accuracy_aver))
        Accuracy = np.diag(C)
        print('Identification accuracy for each flow state：' + str(Accuracy))

        # 画图 典型混淆矩阵
        plt.matshow(C, cmap=plt.cm.Blues)
        def fmt1(x, pos):
            return round(x, 1)
        for i in range(len(C)):
            for j in range(len(C)):
                if round(C[j, i], 3) < 0.39:
                    plt.annotate(round(C[j, i], 3), xy=(i, j), horizontalalignment='center',
                                 verticalalignment='center', fontsize=14)
                else:
                    plt.annotate(round(C[j, i], 3), xy=(i, j), horizontalalignment='center',
                                 verticalalignment='center',
                                 color='w', fontsize=14)
        plt.colorbar(format=ticker.FuncFormatter(fmt1))
        plt.tick_params(labelsize=14)
        from pylab import *

        mpl.rcParams['font.sans-serif'] = ['SimHei']
        plt.ylabel('True label of flow state', fontsize=15)
        plt.xlabel('Predicted label of flow state', fontsize=15)
        plt.xticks(range(0, 10), labels=['1','2','3','4','5','6','7','8','9', '10'])
        plt.yticks(range(0, 10), labels=['1','2','3','4','5','6','7','8','9', '10'])
        plt.show()

        # 画图 典型类别
        fig3 = plt.figure(3, figsize=(10, 4), dpi=200)
        plt.imshow(label_lis.reshape([10, 200]), interpolation='nearest', cmap='tab10', origin='upper', aspect='auto')
        plt.xlabel("Samples", fontsize=15)
        mpl.rc('xtick', labelsize=15)
        plt.yticks(range(10), ('State1', 'State2', 'State3', 'State4', 'State5', 'State6', 'State7', 'State8', 'State9', 'State10'), fontsize=15)
        plt.show()

    else:
        # 属性滑窗平均
        p = 30
        attribute_roll = np.zeros([11, 620 - p])
        for i in range(11):
            d = pd.Series(test_pre_attribute[:, i])
            attribute_roll[i, :] = (np.array(d.rolling(p).mean())[p:620]) / 5

        # 属性演化热图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig1 = plt.figure(1, figsize=(9, 4), dpi=200)
        plt.imshow(attribute_roll, interpolation='nearest', cmap='Purples', origin='upper', aspect='auto')
        plt.xlabel("Samples", fontsize=15)
        plt.ylabel("State attributes", fontsize=15)
        plt.yticks(range(11), ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11'))
        mpl.rc('ytick', labelsize=15)
        plt.colorbar()
        plt.tight_layout(pad=0.5)
        plt.show()

