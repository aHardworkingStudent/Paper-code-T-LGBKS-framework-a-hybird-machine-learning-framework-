# Thank you very much for your recognition of our work.
# As the technology used in this work is currently being applied in a Chinese company and related technologies are
# being applied for national invention patents, the complete data and code may affect the smooth development of
# future work. Therefore, the complete code cannot be released at this time, and we apologize for any inconvenience
# this may cause.

# However, in consideration of the needs of other researchers, we have provided approximately 80% of the source code
# used in this paper in the supplementary documentation, with the remaining 20% described in detail in the document
# using annotation. Readers can replicate our model more easily by referring to the existing code and the detailed
# description in the paper.

# Besides, We left the contact information in supplementary documentation and explained the reasons for retaining
# some of the code. If researchers have academic or teaching needs, we welcome them to send us an email.
# We will evaluate whether the reader's use is only for academic or teaching purposes before choosing to share
# all the code separately.

# E-mail:mengkunphd@163.com
# Researchers in need please feel free to contact us, thank you for your attention !


# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore")

df = pd.read_excel("dataSet.xlsx")
df_interpolate = df.interpolate('linear')

df = df_interpolate

df.iloc[:, 2:].hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
                    xlabelsize=8, ylabelsize=8, grid=False, figsize=(10, 10))
plt.tight_layout(rect=(0, 0, 1.5, 1.5))


# 无量纲化
def dimensionlessProcessing(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        # MEAN = d.mean()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame


def dimensionBack(y, origin_y=origin_y):
    max = origin_y.max()
    min = origin_y.min()
    mean = origin_y.mean()
    min_y = y.min()
    back = []
    for i in y:
        back.append(i * (max - min) + min)
    back = pd.Series(back)
    return back


# 2.数据归一化
def min_max(df, test_size):
    from sklearn.model_selection import train_test_split
    data = dimensionlessProcessing(df.iloc[:, 2:])
    X = data.iloc[:, 1:]  # 特征
    y = data.iloc[:, 0]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    return X, y, data, X_train, X_test, y_train, y_test


X, y, data, X_train, X_test, y_train, y_test = min_max(df, 0.2)


# 2.数据归一化
def min_max(df, test_size):
    from sklearn.model_selection import train_test_split
    data = dimensionlessProcessing(df.iloc[:, 2:])
    X = data.iloc[:, 1:]  # 特征
    y = data.iloc[:, 0]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    return X, y, data, X_train, X_test, y_train, y_test


X, y, data, X_train, X_test, y_train, y_test = min_max(df, 0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


plt.style.use("seaborn-darkgrid")


def draw_model(model, X, y, X_train, X_test, y_train, y_test, title, label):
    print("Training score: ", model.score(X_train, y_train))
    predictions = model.predict(X_test)
    pre = model.predict(X)
    pre1 = dimensionBack(pre)
    # print("Predictions: ", predictions)
    print('-----------------')
    r2score = r2_score(y_test, predictions)
    print("r2 score is: ", r2score)
    print('MAE:{}', mean_absolute_error(
        dimensionBack(y_test), dimensionBack(predictions)))
    print('MSE:{}', mean_squared_error(
        dimensionBack(y_test), dimensionBack(predictions)))
    print('RMSE:{}', np.sqrt(mean_squared_error(
        dimensionBack(y_test), dimensionBack(predictions))))
    print('MAPE:{}', mape(dimensionBack(y_test), dimensionBack(predictions)))
    # 真实值和预测值的差值
    sns.distplot(y_test - predictions)
    plt.figure(figsize=(15, 7.5), dpi=200)  # 创建画布
    plt.plot(np.arange(X.shape[0]), origin_y,
             color='k', label='Real y')  # 画出原始值的曲线
    plt.plot(np.arange(X.shape[0]), pre1, label=label)  # 画出每条预测结果线
    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 30})  # 标题
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 30}, frameon=True, fancybox=True,
               edgecolor='black')  # 图例位置
    plt.ylabel('Real and predicted value',
               fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 20})  # y轴标题
    plt.show()  # 展示图像


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
print(model)
print(linreg.intercept_)
print(linreg.coef_)
y_pred = linreg.predict(X_test)

pre = model.predict(X)

draw_model(model, X, y, X_train, X_test, y_train, y_test, 'Multiple linear regression result', 'MLR')


def KNN(X, y, X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsRegressor
    best_MAE = 100
    best_MAE_index = 1
    best_MSE = 100
    best_MSE_index = 1
    best_RMSE = 100
    best_RMSE_index = 1
    for K in range(2, 10):
        model = KNeighborsRegressor(n_neighbors=K)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        if mae < best_MAE:
            best_MAE = mae
            best_MAE_index = K
        if mse < best_MSE:
            best_MSE = mae
            best_MSE_index = K
        if rmse < best_RMSE:
            best_RMSE = mae
            best_RMSE_index = K

    model = KNeighborsRegressor(n_neighbors=best_RMSE_index)
    model.fit(X_train, y_train)
    draw_model(model, X, y, X_train, X_test, y_train, y_test, 'KNN regression result', 'KNN')
    return model


KNN_global = KNN(X, y, X_train, X_test, y_train, y_test)

# 导入库
from random import shuffle
import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
import sklearn.ensemble
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库

# 数据准备
# 1 2 3 4 5——mean 五折交叉验证
# 训练回归模型
n_folds = 4  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = sklearn.ensemble.GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合
cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, X, y, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(X, y).predict(X))  # 将回归训练中得到的预测y存入列表
# 模型效果指标评估
n_samples, n_features = X.shape  # 总样本量,总特征数
# explained_variance_score:解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
# 的方差变化，值越小则说明效果越差。
# mean_absolute_error:平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
# ，其其值越小说明拟合效果越好。
# mean_squared_error:均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
# 平方和的均值，其值越小说明拟合效果越好。
# r2_score:判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
# 变量的方差变化，值越小则说明效果越差。
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(y, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
print('cross validation result:')  # 打印输出标题
print(df1)  # 打印输出交叉检验的数据框
print(70 * '-')  # 打印分隔线
print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线
# 模型效果可视化
plt.figure(figsize=(15, 7.5), dpi=150)  # 创建画布
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
# 模型应用
print('regression prediction')

import scipy.stats as stats
from pprint import pprint
from sklearn import metrics
from openpyxl import load_workbook
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

'''TPE-RF'''
global best
best = 9999


def TPE4rf(max_evals, X, y, choose=0):
    def hyperopt_train_test(params):
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=10, shuffle=False)
        from sklearn import metrics
        clf = RandomForestRegressor(**params)
        from sklearn.model_selection import cross_validate
        cv_cross = cross_validate(
            clf, X, y, cv=kfold, scoring=('r2', 'neg_mean_squared_error'))
        return -(cv_cross['test_neg_mean_squared_error'].mean())

    space4rf = {
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'max_depth': hp.choice('max_depth', range(3, 20)),
        'max_features': hp.choice('max_features', range(3, 15)),
        'min_samples_split': hp.choice('min_samples_split', range(2, 15)),
    }

    best = 99999
    best_Params_auto = {'max_depth': 3, 'max_features': 3, 'min_samples_split': 7, 'n_estimators': 870}

    def f(params):
        global best
        acc = hyperopt_train_test(params)
        if acc < best:
            best = acc
            print('new best:', best, params)
            for key, value in params.items():
                if key in best_Params_auto.keys():
                    best_Params_auto[key] = params[key]
        return {'loss': acc, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(f, space4rf, algo=tpe.suggest,
                       max_evals=max_evals, trials=trials)

    if choose == 1:
        print('best_params:', best_params)
        parameters = list(best_params.keys())
        f, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
        cmap = plt.cm.jet
        for i, val in enumerate(parameters):
            print(i, val)
            xs = np.array([t['misc']['vals'][val]
                           for t in trials.trials]).ravel()
            ys = [t['result']['loss'] for t in trials.trials]
            ys = np.array(ys)
            axes[i].scatter(xs, ys, s=20, linewidth=0.01,
                            alpha=0.25, c=cmap(float(i) / len(parameters)))
            axes[i].set_title(val)

    return best_Params_auto


# Auto-search optimal hyperparameter with TPE
def hyperopt_train_test(params):
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=6, shuffle=False)
    from sklearn import metrics
    clf = RandomForestRegressor(**params)
    # X_train,X_test,Y_train, Y_test = train_test_split(X, Y, test_size=test_size,shuffle=False)
    # clf.fit(X_train,Y_train)
    # Y_pre = clf.predict(X_test)
    from sklearn.model_selection import cross_validate
    cv_cross = cross_validate(clf, X, y, cv=kfold, scoring=('r2', 'neg_mean_squared_error'))

    return -cv_cross['test_neg_mean_squared_error'].mean()


space4rf = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'max_depth': hp.choice('max_depth', range(3, 20)),
    'max_features': hp.choice('max_features', range(3, 15)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 15)),
}

best = 99999


def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc < best:
        best = acc
        print('new best:', best, params)
    return {'loss': acc, 'status': STATUS_OK}


trials = Trials()
best_params = fmin(f, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)

print('best_params:', best_params)
parameters = list(best_params.keys())
f, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    print(i, val)
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [t['result']['loss'] for t in trials.trials]
    ys = np.array(ys)
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)

best_Params_auto_rf = TPE4rf(300, X, y)
# 训练模型
rf1 = RandomForestRegressor(**best_Params_auto_rf)
# {'max_depth': 3, 'max_features': 3, 'min_samples_split': 7, 'n_estimators': 870}
rf = rf1.fit(X_train, y_train)

print("Training score: ", rf.score(X, y))
draw_model(rf, X, y, X_train, X_test, y_train, y_test, 'Random forest regression result', 'Random_forest')

global best
best = 9999


def TPE4xgb(max_evals, X, y, choose=0):
    def hyperopt_train_test(params):
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=10, shuffle=False)
        from sklearn import metrics
        clf = XGBRegressor(**params)
        from sklearn.model_selection import cross_validate
        cv_cross = cross_validate(
            clf, X, y, cv=kfold, scoring=('r2', 'neg_mean_squared_error'))
        return -(cv_cross['test_neg_mean_squared_error'].mean())

    space4xgb = {
        'max_depth': hp.choice('max_depth', range(3, 20)),
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'subsample': hp.uniform('subsample', 0.6, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.65, 0.95),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.21),
    }

    best = 99999
    best_Params_auto = {'colsample_bytree': 0.7841394718769811, 'learning_rate': 0.12895025652481837,
                        'max_depth': 3, 'n_estimators': 733, 'subsample': 0.7878949670020676}

    def f(params):
        global best
        acc = hyperopt_train_test(params)
        if acc < best:
            best = acc
            print('new best:', best, params)
            for key, value in params.items():
                if key in best_Params_auto.keys():
                    best_Params_auto[key] = params[key]
        return {'loss': acc, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(f, space4xgb, algo=tpe.suggest,
                       max_evals=max_evals, trials=trials)

    if choose == 1:
        print('best_params:', best_params)
        parameters = list(best_params.keys())
        f, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
        cmap = plt.cm.jet
        for i, val in enumerate(parameters):
            print(i, val)
            xs = np.array([t['misc']['vals'][val]
                           for t in trials.trials]).ravel()
            ys = [t['result']['loss'] for t in trials.trials]
            ys = np.array(ys)
            axes[i].scatter(xs, ys, s=20, linewidth=0.01,
                            alpha=0.25, c=cmap(float(i) / len(parameters)))
            axes[i].set_title(val)

    return best_Params_auto


best_Params_auto_xgb = TPE4xgb(300, X, y)

# 训练模型
xgb = XGBRegressor(**best_Params_auto_xgb)
# {'colsample_bytree': 0.6888810537357809, 'learning_rate': 0.16704758920419038, 'max_depth': 10, 'n_estimators': 443, 'subsample': 0.8793233427076314}
xgb = xgb.fit(X_train, y_train)

print("Training score: ", xgb.score(X_train, y_train))

draw_model(xgb, X, y, X_train, X_test, y_train, y_test, 'XGBoost regression result', 'XGBoost')

global best
best = 9999


def TPE4lgb(max_evals, X, y, choose=0):
    def hyperopt_train_test(params):
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=10, shuffle=False)
        from sklearn import metrics
        clf = LGBMRegressor(**params)
        from sklearn.model_selection import cross_validate
        cv_cross = cross_validate(
            clf, X, y, cv=kfold, scoring=('r2', 'neg_mean_squared_error'))
        return -(cv_cross['test_neg_mean_squared_error'].mean())

    space4lgb = {
        'max_depth': hp.choice('max_depth', range(3, 20)),
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'subsample': hp.uniform('subsample', 0.6, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.65, 0.95),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.21),
    }

    best = 99999
    best_Params_auto = {'colsample_bytree': 0.7841394718769811, 'learning_rate': 0.12895025652481837,
                        'max_depth': 3, 'n_estimators': 733, 'subsample': 0.7878949670020676}

    def f(params):
        global best
        acc = hyperopt_train_test(params)
        if acc < best:
            best = acc
            print('new best:', best, params)
            for key, value in params.items():
                if key in best_Params_auto.keys():
                    best_Params_auto[key] = params[key]
        return {'loss': acc, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(f, space4lgb, algo=tpe.suggest,
                       max_evals=max_evals, trials=trials)

    if choose == 1:
        print('best_params:', best_params)
        parameters = list(best_params.keys())
        f, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
        cmap = plt.cm.jet
        for i, val in enumerate(parameters):
            print(i, val)
            xs = np.array([t['misc']['vals'][val]
                           for t in trials.trials]).ravel()
            ys = [t['result']['loss'] for t in trials.trials]
            ys = np.array(ys)
            axes[i].scatter(xs, ys, s=20, linewidth=0.01,
                            alpha=0.25, c=cmap(float(i) / len(parameters)))
            axes[i].set_title(val)

    return best_Params_auto


# lgb1= LGBMRegressor(**best_params)
# {'colsample_bytree': 0.7108129813542299, 'learning_rate': 0.01003066208766242, 'max_depth': 5, 'n_estimators': 115, 'subsample': 0.7552434962742499}
best_Params_auto_lgb = TPE4lgb(300, X, y)
# 训练模型
lgb1 = LGBMRegressor(**best_Params_auto_lgb)
lgb = lgb1.fit(X_train, y_train)

print("Training score: ", lgb.score(X_train, y_train))
draw_model(lgb, X, y, X_train, X_test, y_train, y_test, 'LightGBM regression result', 'LightGBM')

print("best_Params_auto_xgb:{}".format(best_Params_auto_xgb))
print("best_Params_auto_lgb:{}".format(best_Params_auto_lgb))
print("best_Params_auto_rf:{}".format(best_Params_auto_rf))

rf = RandomForestRegressor(**{'max_depth': 18, 'max_features': 3, 'min_samples_split': 3, 'n_estimators': 376})
# {'colsample_bytree': 0.6888810537357809, 'learning_rate': 0.16704758920419038, 'max_depth': 10, 'n_estimators': 443, 'subsample': 0.8793233427076314}
rf = rf.fit(X_train, y_train)

print("Training score: ", rf.score(X_train, y_train))

draw_model(rf, X, y, X_train, X_test, y_train, y_test, 'Random forests regression result', 'Random forests')

xgb = XGBRegressor(**{'colsample_bytree': 0.6626929239294624, 'learning_rate': 0.15449783083198493, 'max_depth': 3,
                      'n_estimators': 912, 'subsample': 0.726613502522596})
# {'colsample_bytree': 0.6888810537357809, 'learning_rate': 0.16704758920419038, 'max_depth': 10, 'n_estimators': 443, 'subsample': 0.8793233427076314}
xgb = xgb.fit(X_train, y_train)

print("Training score: ", xgb.score(X_train, y_train))

draw_model(xgb, X, y, X_train, X_test, y_train, y_test, 'XGBoost regression result', 'XGBoost')

# 训练模型
lgb1 = LGBMRegressor(**{'colsample_bytree': 0.6626929239294624, 'learning_rate': 0.15449783083198493, 'max_depth': 3,
                        'n_estimators': 912, 'subsample': 0.726613502522596})
lgb = lgb1.fit(X_train, y_train)

print("Training score: ", lgb.score(X_train, y_train))
draw_model(lgb, X, y, X_train, X_test, y_train, y_test, 'LightGBM regression result', 'LightGBM')

from keras import initializers
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dropout
from keras.layers.core import Dense
import tensorflow as tf

# 构建神经网络
train_data = X
train_targets = y


def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu',
                    input_shape=(train_data.shape[1],),
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)),
              )
    # model.add(Dropout(rate = 0.5))
    model.add(Dense(64, activation='relu',
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)),
              )
    # model.add(Dropout(rate = 0.5))
    # model.add(Dense(32, activation='relu',
    #                         kernel_initializer=initializers.TruncatedNormal(mean=0.0,stddev=0.05,seed=None)),
    #                         )
    # model.add(Dropout(rate = 0.5))
    # !网络的最后一层只有一个单元,没有激活,是一个线性层
    # !这是标量回归（标量回归是预测单一连续值的回归）的典型设置
    model.add(Dense(1))
    # !编译网络用的是mse损失函数,即均方误差（MSE, mean squared error）
    # !预测值与目标值之差的平方,这是回归问题常用的损失函数

    # !平均绝对误差（MAE, mean absolute error）
    # !是预测值与目标值之差的绝对值
    optimizer = tf.keras.optimizers.Adam(0.004)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


num_epochs = 200
DNN = build_model()
history = DNN.fit(X_train, y_train,
                  epochs=num_epochs,
                  batch_size=10,
                  validation_split=0.2,
                  verbose=0,
                  )

# !使用验证集验证
val_mse, val_mae = DNN.evaluate(X_test, y_test, verbose=0)

# print("MSE:{},MAE:{}".format(val_mse, val_mae))
# print("Training score: ", model.score(X_train, y_train))
# predictions = model.predict(X_test)
# pre = model.predict(X)
# # print("Predictions: ", predictions)
# print('-----------------')
# r2score = r2_score(y_test, predictions)
# print("r2 score is: ", r2score)
# print('MAE:{}', mean_absolute_error(y_test, predictions))
# print('MSE:{}', mean_squared_error(y_test, predictions))
# print('RMSE:{}', np.sqrt(mean_squared_error(y_test, predictions)))
# # 真实值和预测值的差值
# sns.distplot(y_test - predictions)
# plt.figure(figsize=(15, 7.5), dpi=200)  # 创建画布
# plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
# plt.plot(np.arange(X.shape[0]), pre, label='DNN')  # 画出每条预测结果线
# plt.title('DNN regression result')  # 标题
# plt.legend(loc='upper right')  # 图例位置
# plt.ylabel('real and predicted value')  # y轴标题
# plt.show()  # 展示图像

model = DNN
predictions = model.predict(X_test)
pre = model.predict(X)
pre1 = dimensionBack(pre)
# print("Predictions: ", predictions)
print('-----------------')
r2score = r2_score(y_test, predictions)
print("r2 score is: ", r2score)
print('MAE:{}', mean_absolute_error(
    dimensionBack(y_test), dimensionBack(predictions)))
print('MSE:{}', mean_squared_error(
    dimensionBack(y_test), dimensionBack(predictions)))
print('RMSE:{}', np.sqrt(mean_squared_error(
    dimensionBack(y_test), dimensionBack(predictions))))
print('MAPE:{}', mape(dimensionBack(y_test), dimensionBack(predictions)))
# 真实值和预测值的差值


plt.figure(figsize=(15, 7.5), dpi=200)  # 创建画布
plt.plot(np.arange(X.shape[0]), origin_y,
         color='k', label='Real y')  # 画出原始值的曲线
plt.plot(np.arange(X.shape[0]), pre1, label='DNN')  # 画出每条预测结果线
plt.title('DNN regression result', fontdict={'family': 'Times New Roman', 'size': 30})  # 标题
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.legend(loc='upper right', prop={'family': 'Times New Roman',
                                    'size': 30}, frameon=True, fancybox=True, edgecolor='black')  # 图例位置
plt.ylabel('Real and predicted value', fontdict={
    'family': 'Times New Roman', 'weight': 'normal', 'size': 20})  # y轴标题
plt.show()  # 展示图像

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('train_valid_loss.png')
plt.show()


def plotTestSet(y_test, predictions, model_name):
    '''绘制回归常用的点线图'''
    plt.figure(figsize=(7.5, 6), dpi=100)  # 创建画布
    plt.plot(np.arange(max(dimensionBack(y_test))), color='red', label="$y=x$")
    plt.scatter(dimensionBack(y_test), dimensionBack(
        predictions), s=30, marker='o', c='blue', label="Data")
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.legend(prop={'family': 'Times New Roman',
                     'size': 20}, frameon=True, fancybox=True, edgecolor='black')  # 图例位置
    plt.ylabel('Original value', fontdict={
        'family': 'Times New Roman', 'weight': 'normal', 'size': 20})  # y轴标题
    plt.xlabel('Predicted value', fontdict={
        'family': 'Times New Roman', 'weight': 'normal', 'size': 20})  # y轴标题
    plt.title(model_name, fontdict={
        'family': 'Times New Roman', 'weight': 'normal', 'size': 20})
    plt.show()  # 展示图像


def model_effect(model, X, y, X_train, X_test, y_train, y_test, model_name):
    print("Training score: ", model.score(X_train, y_train))
    predictions = model.predict(X_test)
    pre = model.predict(X)
    pre1 = dimensionBack(pre)
    # print("Predictions: ", predictions)
    print('-----------------')
    r2score = r2_score(y_test, predictions)
    print("r2 score is: ", r2score)
    print('MAE:{}', mean_absolute_error(
        dimensionBack(y_test), dimensionBack(predictions, )))
    print('MSE:{}', mean_squared_error(
        dimensionBack(y_test), dimensionBack(predictions)))
    print('RMSE:{}', np.sqrt(mean_squared_error(
        dimensionBack(y_test), dimensionBack(predictions))))
    print('MAPE:{}', mape(dimensionBack(y_test), dimensionBack(predictions)))
    # 真实值和预测值的差值分布可视化
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    sns.distplot(y_test - predictions)
    plt.figure(figsize=(15, 7.5), dpi=200)  # 创建画布

    '''绘制回归常用的点线图'''
    plotTestSet(y_test, predictions, model_name)


from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
svr_poly = SVR(kernel='poly')

y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
msetest_rbf = mean_squared_error(y_test, y_rbf)
msetest_lin = mean_squared_error(y_test, y_lin)
msetest_poly = mean_squared_error(y_test, y_poly)

print(msetest_rbf)
print(msetest_lin)
print(msetest_poly)
# draw_model(svr_rbf,X,y, X_train, X_test, y_train, y_test,'svr_rbf regression result','svr_rbf')
# draw_model(svr_lin,X,y, X_train, X_test, y_train, y_test,'svr_lin regression result','svr_lin')
# draw_model(svr_poly,X,y, X_train, X_test, y_train, y_test,'svr_poly regression result','svr_poly')

model_effect(svr_rbf, X, y, X_train, X_test, y_train, y_test, 'svr_rbf regression result')
model_effect(svr_lin, X, y, X_train, X_test, y_train, y_test, 'svr_lin regression result')
model_effect(svr_poly, X, y, X_train, X_test, y_train, y_test, 'svr_poly regression result')

import shap

explainer = shap.TreeExplainer(lgb)
# 初始化解释器
shap.initjs()  # 载入JS
shap_values = explainer.shap_values(X)  # 计算SHAP值

# summarize the effects of all the features
shap.summary_plot(shap_values, df.iloc[:, 3:])

shap.summary_plot(shap_values, df.iloc[:, 3:], plot_type="bar")

shap_interaction_values = explainer.shap_interaction_values(df.iloc[:, 3:])
shap.summary_plot(shap_interaction_values, df.iloc[:, 3:])

shap_values2 = explainer(df.iloc[:, 3:])

clustering = shap.utils.hclust(df.iloc[:, 3:], df.iloc[:, 2])
shap.plots.bar(shap_values2,
               clustering=clustering,
               clustering_cutoff=0.5)

shap.plots.bar(shap_values2[1], show_data=True)

shap.plots.waterfall(shap_values2[5])

for i in range(len(df.iloc[:, 3:].columns)):
    shap.dependence_plot('rank({})'.format(i), shap_values, X, feature_names=df.iloc[:, 3:].columns)

explainer1 = shap.TreeExplainer(xgb)  # 初始化解释器

shap_values = explainer1.shap_values(X)  # 计算SHAP值

# summarize the effects of all the features
shap.summary_plot(shap_values, df.iloc[:, 3:])

shap.summary_plot(shap_values, df.iloc[:, 3:], plot_type="bar")

shap_values2 = explainer(df.iloc[:, 3:])

clustering = shap.utils.hclust(df.iloc[:, 3:], df.iloc[:, 2])
shap.plots.bar(shap_values2,
               clustering=clustering,
               clustering_cutoff=0.5)

shap.plots.bar(shap_values2[1], show_data=True)

shap.plots.waterfall(shap_values2[5])

shap.plots.waterfall(shap_values2[5])

for i in range(len(df.iloc[:, 3:].columns)):
    shap.dependence_plot('rank({})'.format(i), shap_values, X, feature_names=df.iloc[:, 3:].columns)
