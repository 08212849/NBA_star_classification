import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_csv('../NBA_Season_Stats.csv')
data = data.fillna(0)
X = data.drop(['Pos', 'Player', 'Tm'], axis=1)
y = data['Pos']
# 处理缺失值
X_columns = X.columns

# 数据标准化处理
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)

def showCorrelation(X):
    '''
    展示各变量间的相关性，绘制热力图
    :param X_np: numpy.ndarray对象的数据
    :param X_columns: 数据列名
    :return:
    '''
    # 计算相关矩阵
    corr_matrix = X.corr()
    def drawHeatmap():
        # 绘制热力图
        plt.figure(figsize=(15, 15))  # 可以根据需要调整图形的大小
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
        plt.savefig('../result/heatmap1.png')
        plt.clf()
    def printCorr():
        glossary = {
            "Age": "球员的年龄。",
            "G": "球员参加的比赛场次",
            "MP": "球员平均每场比赛的出场时间",
            "FG": "投篮命中数",
            "FGA": "投篮出手数",
            "FG%": "投篮命中率，计算方式为 FG / FGA。",
            "3P": "三分球命中数",
            "3PA": "三分球出手数",
            "3P%": "三分球命中率",
            "2P": "两分球命中数",
            "2PA": "两分球出手数",
            "2P%": "两分球命中率",
            "eFG%": "有效投篮命中率",
            "FT": "罚球命中数",
            "FTA": "罚球出手数",
            "FT%": "罚球命中率",
            "ORB": "进攻篮板数",
            "DRB": "防守篮板数",
            "TRB": "总篮板数",
            "AST": "助攻数",
            "STL": "抢断数",
            "BLK": "盖帽数",
            "TOV": "失误数",
            "PF": "犯规数",
            "PTS": "总得分"
        }
        # 输出相关性大于0.95的变量对
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):  # 避免重复和对角线元素
                if corr_matrix.iloc[i, j] > 0.95:
                    print(f"变量 {corr_matrix.columns[i]}( {glossary[corr_matrix.columns[i]]})和变量 "
                          f"{corr_matrix.columns[j]} ({glossary[corr_matrix.columns[j]]})的相关性为 "
                          f"{corr_matrix.iloc[i, j]:.2f}")
    # drawHeatmap()
    # printCorr()

def process_PCA(df_scaled):
    pca = PCA(n_components=20)
    pca.fit(df_scaled)
    df_pca = pca.transform(df_scaled)
    def printRatio():
        # 查看每个主成分解释的方差比例
        print("每个主成分解释的方差比例:")
        sum = 0.0
        for i, var in enumerate(pca.explained_variance_ratio_, 1):
            sum += var
            if i < 11:
                print(f"主成分{i}: {var:.4f} sum :{sum:.4f}")
        # 打印前几个主成分的DataFrame
        print(df_pca.head())
    df_pca = pd.DataFrame(df_pca, columns=[f'PC{i}' for i in range(1, len(pca.components_) + 1)])
    # printRatio()
    return df_pca

# showCorrelation(X)
X = process_PCA(df_scaled)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

def printResult(nb):
    y_pred = nb.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4], target_names=label_encoder.classes_))

def drawConfusionMatrix(nb):
    y_pred = nb.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 绘制混淆矩阵
    ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(values_format='.0f',                                                                     cmap='Blues')  # 可以根据需要调整参数
    plt.savefig('../result/confusion_matrix_gnb.png')
    plt.clf()

def bayes_gnb():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    printResult(gnb)
    # drawConfusionMatrix(gnb)

def bayes_clf(X_train):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    printResult(clf)

def bayes_bnl():
    bnl = BernoulliNB()
    bnl.fit(X_train, y_train)
    printResult(bnl)

bayes_clf(X_train)
bayes_gnb()
bayes_bnl()