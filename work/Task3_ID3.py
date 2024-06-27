import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_csv('../NBA_Season_Stats.csv')
data = data.fillna(0)
X_columns = ['Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK']

# Age(年龄) PTS(总得分) TRB (总篮板数) AST (助攻) STL (抢断) BLK (盖帽)
X = data[X_columns]
y = data['Pos']
y_str_list = y.unique().tolist()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# 数据标准化处理
scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = pd.DataFrame(X, columns=X_columns)

# 为每一列特征绘制分布图
def drawFeatures():
    for column in X.columns:
        plt.figure(figsize=(8, 6))  # 设置图形大小
        sns.histplot(X[column], kde=True)  # 使用seaborn绘制直方图和核密度估计
        plt.title(f'Distribution of {column}')  # 设置图形标题
        plt.xlabel('Value')  # X轴标签
        plt.ylabel('Frequency')  # Y轴标签
        plt.grid(True)  # 显示网格
        plt.savefig(f'../result/feature_of_{column}.png')
        plt.clf()

# 连续数据离散化，确保每组大约有相同数量的数据点
def data_Discretization(X, n):
    for column in X.columns:
        # 设置分位数
        quantiles = np.linspace(0, 1, n+1)  # 生成0.2, 0.4, 0.6, 0.8
        bins = X[column].quantile(quantiles)  # 计算边界值
        labels = [str(i) for i in range(n)]
        # print(labels)
        new_column = f"{column}_level"
        # 使用pd.cut进行分桶
        X[column] = pd.cut(X[column], bins=bins, labels=labels, right=False)

# drawFeatures()
# data_Discretization(X, 7)
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用交叉验证来找到最优的剪枝参数
def find_Bestparams():
    param_grid = {
        'min_samples_leaf': [40, 50, 60],
        'min_samples_split': [40, 60, 80, 100, 120],
        'max_depth': [5, 10, 15],
        'ccp_alpha': [0, 0.001, 0.01, 0.1, 0.2],
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    # 获取最优参数
    best_params = grid_search.best_params_
    # 打印最优参数
    print("最优参数:", best_params)
    return clf

# 交叉验证寻找最优参数
# clf = find_Bestparams()

clf = DecisionTreeClassifier(criterion='entropy', max_depth=15,
                             min_samples_leaf=40, min_samples_split=40)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 打印决策树的规则
def drawID3Tree():
    tree_rules = export_text(clf, feature_names=list(X.columns))
    # print(tree_rules)
    fig = plt.figure(figsize=(40, 16))
    _ = tree.plot_tree(
        clf,
        feature_names=X_columns,
        class_names=y_str_list,
        filled=True,
        # max_depth=2
    )
    # Save picture
    fig.savefig("../result/All_decistion_tree.png")

def drawConfusionMatrix():
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵
    ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(values_format='.0f',                                                                     cmap='Blues')  # 可以根据需要调整参数
    plt.savefig('../result/confusion_matrix_ID3.png')
    plt.clf()

drawConfusionMatrix()