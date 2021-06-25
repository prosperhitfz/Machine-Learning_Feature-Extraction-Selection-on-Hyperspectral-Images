import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# 不显示warning
import warnings
warnings.filterwarnings('ignore')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def manual_pca(inner_data, feature_components):  # 手写pca特征维度压缩函数
    # 求数据矩阵每一列的均值
    data_mean = np.mean(inner_data, axis=0)
    # 计算数据矩阵每一列特征减去该列的特征均值
    data_mean_subtract = inner_data - data_mean
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    data_cov_matrix = np.cov(data_mean_subtract, rowvar=False)
    # 计算协方差矩阵的特征值eigenvalue及对应的特征向量feature vector
    cov_mat_eigenvalue, cov_mat_feature_vector = np.linalg.eig(np.mat(data_cov_matrix))
    # np.argsort()用于对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eigenvalue_index = np.argsort(cov_mat_eigenvalue)
    # 从排序后的矩阵最后一个开始自下而上选取最大的k个特征值，返回其对应的索引
    selected_eigenvalue_index = eigenvalue_index[:-(feature_components + 1): -1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    compression_matrix = cov_mat_feature_vector[:, selected_eigenvalue_index]
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为k
    data_descended = data_mean_subtract * compression_matrix
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return data_descended


def classification_by_mlp(inner_data, inner_label):  # 神经网络分类器（2层hidden layer，分别为100，50个神经元，共迭代一百万次）
    # 划分数据集、训练集
    # X_train, X_test, y_train, y_test = train_test_split(inner_data, inner_label, test_size=0.14, random_state=1)
    # print(y_test.shape[0])
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000000)
    mlp.fit(inner_data, inner_label)  # 训练数据拟合
    train_prediction_res = np.reshape(mlp.predict(inner_data), (-1, 1))
    test_prediction_res = np.reshape(mlp.predict(inner_data), (-1, 1))
    # print(test_prediction_res[1][0])
    # print(np.reshape(y_test, (1, -1)))
    # print(np.reshape(test_prediction_res, (1, -1)))
    print('训练集分类错误率（也称 经验误差）：', 1 - accuracy_score(inner_label, train_prediction_res))
    print('测试集分类错误率（也称 错误率、泛化误差）：', 1 - accuracy_score(inner_label, test_prediction_res))
    print('精度：', accuracy_score(inner_label, test_prediction_res))

    # 计算宏PR值和宏F1 Score（这里的宏意指多分类的PR参数和F1 Score，见周志华西瓜书第二章）
    record_list = np.zeros(inner_label.shape)  # 创建一个用于记录每类（标签）有多少个数据的数组
    for i in range(inner_label.shape[0]):
        record_list[inner_label[i][0]][0] += 1  # 记录每一类有多少数据
    P_mid = 0
    R_mid = 0
    time = 0  # 记录一共有多少个二值混淆矩阵
    for i in range(record_list.shape[0]):
        for j in range(record_list.shape[0]):
            TP = 0  # 初始化被正确预测为正类的正类样本数量
            FP = 0  # 初始化被错误预测为负类的正类样本数量
            TN = 0  # 初始化被正确预测为负类的负类样本数量
            FN = 0  # 初始化被错误预测为正类的负类样本数量
            if record_list[i][0] != 0 and record_list[j][0] != 0 and (i < j):
                time += 1
                for k in range(test_prediction_res.shape[0]):
                    if inner_label[k][0] == i and test_prediction_res[k][0] == i:  # 被正确预测为正类的正类样本数量
                        TP += 1
                    elif inner_label[k][0] == i and test_prediction_res[k][0] == j:  # 被错误预测为正类的负类样本数量
                        FN += 1
                    elif inner_label[k][0] == j and test_prediction_res[k][0] == i:  # 被错误预测为负类的正类样本数量
                        FP += 1
                    elif inner_label[k][0] == j and test_prediction_res[k][0] == j:  # 被正确预测为负类的负类样本数量
                        TN += 1
            if TP + FP == 0 or TP + FN == 0:
                continue
            P_mid += TP / (TP + FP)
            R_mid += TP / (TP + FN)
    P_macro = P_mid / time
    R_macro = R_mid / time
    F1_macro = 2 * P_macro * R_macro / (P_macro + R_macro)
    print('宏查准率（也称 macro-P）：', P_macro)
    print('宏查全率（也称 macro-R）：', R_macro)
    print('宏F1（也称 macro-F1 Score）：', F1_macro)
    return inner_label, test_prediction_res


def visualization_res_matrix(initial_matrix, classify_matrix):  # 将矩阵结果可视化
    initial_img = np.reshape(initial_matrix, (83, -1))  # 将类别矩阵恢复为原来的二维图像尺寸
    classify_img = np.reshape(classify_matrix, (83, -1))  # 将分类矩阵恢复为原来的二维图像尺寸
    plt.matshow(initial_img, cmap=plt.get_cmap('Oranges'), alpha=0.4)
    plt.title('测试集原始图像还原图示')
    plt.show()
    plt.matshow(classify_img, cmap=plt.get_cmap('Oranges'), alpha=0.4)
    plt.title('测试集预测图像还原图示')
    plt.show()


# 整个实验可以这样理解，data就是一个83*86的图像（对那个salinasA来说，那个大的salinas是512*217的尺寸）
# 其中每个像素有204维不同的光谱像素值
# labels是83*86的图像这里面的每一个像素点所属的哪个类别
# 因此，我是要对这83*86个像素点进行划分训练集和测试集并分类，其中每一个像素点有204维特征
# 因为204维特征太多了，不加处理全部考虑反倒会可能使得分类效果变差，所以要进行特征选择，从204维中挑出来一些强相关特征用于分类

# 引入高光谱图像矩阵
data_matrix = scipy.io.loadmat('D://hyperspectral_image_feature_selection//SalinasA_corrected.mat')
# 取矩阵中的图像元素并转换为数组类型，作为数据data
data = np.array(data_matrix['salinasA_corrected'])
print('data:\n', data)
print('data shape:', data.shape)
print('\n')
# 引入图像中每个像素点的标签矩阵并初始化，作为标签labels
label_matrix = scipy.io.loadmat('D://hyperspectral_image_feature_selection//SalinasA_gt.mat')
label = np.array(label_matrix['salinasA_gt'])
print('labels:\n', label)
print('labels shape:', label.shape)
print('\n')
# 将二维图像+特征转化为一个新的processed_data数组，其中每一行为要分类的像素点，每一列为一维特征
processed_data = np.reshape(data, (-1, data.shape[2]))
print('processed data:\n', processed_data)
print('processed data shape:', processed_data.shape)
print('\n')
# 将像素点的特征转换为一个一列的一维数组
processed_label = np.reshape(label, (-1, 1))
print('processed label:\n', processed_label)
print('processed label shape:', processed_label.shape)
print('\n')

# # 使用手写的PCA降维后的数据特征
# feature_descended_pca_manual = manual_pca(processed_data, feature_components=100)
# print('manual PCA descended feature:\n', feature_descended_pca_manual)
# print('manual PCA descended feature shape:', feature_descended_pca_manual.shape)
# # 神经网络训练并进行测试集分类，并给出相关评估参数
# print('手写的PCA特征降维后的模型分类结果评估相关参数:')
# ini_mat, pre_mat = classification_by_mlp(feature_descended_pca_manual, processed_label)
# print('\n')
# # 绘制测试集预测图分类结果和测试集原始图分类结果对比图示
# visualization_res_matrix(ini_mat, pre_mat)
#
# # 使用sklearn封装好的PCA降维后的数据特征
# feature_pca = PCA(n_components=100)
# feature_descended_pca = feature_pca.fit_transform(processed_data)
# print('PCA descended feature:\n', feature_descended_pca)
# print('PCA descended feature shape:', feature_descended_pca.shape)
# # 神经网络训练并进行测试集分类，并给出相关评估参数
# print('PCA特征降维后的模型分类结果评估相关参数:')
# ini_mat1, pre_mat1 = classification_by_mlp(feature_descended_pca, processed_label)
# print('\n')
# # 绘制测试集预测图分类结果和测试集原始图分类结果对比图示
# visualization_res_matrix(ini_mat1, pre_mat1)

# 使用sklearn封装好的Filter方法下的方差分析ANOVA法（f_classif）
# 注：本来是想用卡方检验（chi2）的，但是特征里面有负数，不可用
feature_filter_fclass = SelectKBest(score_func=f_classif, k=100)
feature_descended_filter_fclass = feature_filter_fclass.fit_transform(processed_data, processed_label)
print('filter ANOVA fclassif descended feature:\n', feature_descended_filter_fclass)
print('filter ANOVA fclassif descended feature shape:', feature_descended_filter_fclass.shape)
# 神经网络训练并进行测试集分类，并给出相关评估参数
print('Filter之方差分析法特征选择后的模型分类结果评估相关参数:')
ini_mat2, pre_mat2 = classification_by_mlp(feature_descended_filter_fclass, processed_label)
print('\n')
# 绘制测试集预测图分类结果和测试集原始图分类结果对比图示
visualization_res_matrix(ini_mat2, pre_mat2)

# 使用sklearn封装好的Wrapper方法下的递归特征分析RFE法
# 运行时速度慢???
feature_wrapper_rfe = RFE(estimator=LogisticRegression(), n_features_to_select=100, step=50)  # step是每代减少的特征数
feature_descended_wrapper_rfe = feature_wrapper_rfe.fit_transform(processed_data, processed_label)
print('wrapper RFE descended feature:\n', feature_descended_wrapper_rfe)
print('wrapper RFE descended feature shape:', feature_descended_wrapper_rfe.shape)
# 神经网络训练并进行测试集分类，并给出相关评估参数
print('Wrapper之递归特征分析法特征选择后的模型分类结果评估相关参数:')
ini_mat3, pre_mat3 = classification_by_mlp(feature_descended_wrapper_rfe, processed_label)
print('\n')
# 绘制测试集预测图分类结果和测试集原始图分类结果对比图示
visualization_res_matrix(ini_mat3, pre_mat3)

# 使用sklearn封装好的Embedded方法下的L1范数惩罚项特征选择法
# 运行时速度也慢???
feature_embedded_l1 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(processed_data, processed_label)  # L1范数
feature_descended_embedded_l1 = SelectFromModel(feature_embedded_l1, prefit=True, max_features=100)\
    .transform(processed_data)  # max_features用于限制所选择的特征在100个
print('embedded L1 descended feature:\n', feature_descended_embedded_l1)
print('embedded L1 descended feature shape:', feature_descended_embedded_l1.shape)
# 神经网络训练并进行测试集分类，并给出相关评估参数
print('Embedded之基于L1范数的特征选择法选择后的模型分类结果评估相关参数:')
ini_mat4, pre_mat4 = classification_by_mlp(feature_descended_embedded_l1, processed_label)
print('\n')
# 绘制测试集预测图分类结果和测试集原始图分类结果对比图示
visualization_res_matrix(ini_mat4, pre_mat4)

# 使用sklearn封装好的Embedded方法下基于随机森林的特征选择法
feature_embedded_random_forests = RandomForestClassifier().fit(processed_data, processed_label)  # 随机森林
feature_descended_embedded_random_forests = SelectFromModel(feature_embedded_random_forests,
                                                            prefit=True,
                                                            max_features=100)\
    .transform(processed_data)  # max_features用于限制所选择的特征在100个
print('embedded random forests descended feature:\n', feature_descended_embedded_l1)
print('embedded random forests descended feature shape:', feature_descended_embedded_l1.shape)
# 神经网络训练并进行测试集分类，并给出相关评估参数
print('Embedded之基于随机森林的特征选择法选择后的模型分类结果评估相关参数:')
ini_mat5, pre_mat5 = classification_by_mlp(feature_descended_embedded_random_forests, processed_label)
print('\n')
# 绘制测试集预测图分类结果和测试集原始图分类结果对比图示
visualization_res_matrix(ini_mat5, pre_mat5)
