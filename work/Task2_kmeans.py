import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import accuracy_score, silhouette_score
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_csv('../NBA_Season_Stats.csv')

X_columns = ['Pos', 'Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK']
# Pos(位置) Age(年龄) PTS(总得分) TRB (总篮板数) AST (助攻) STL (抢断) BLK (盖帽)
X = data[X_columns]

label_encoder = LabelEncoder()
X['Pos'] = label_encoder.fit_transform(X['Pos'])

def process_PCA(df_scaled):
    pca = PCA(n_components=2)
    pca.fit(df_scaled)
    def drawLoadings():
        df = pd.DataFrame(df_scaled, columns=X_columns)
        loadings = pd.DataFrame(pca.components_, columns=df.columns)
        print(loadings)
        loadings.plot(kind='bar')
        plt.show()
        plt.clf()
    # drawLoadings()
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
    df_pca = pd.DataFrame(df_pca, columns=[f'PCA{i}' for i in range(1, len(pca.components_) + 1)])
    # printRatio()
    return df_pca

# 数据标准化处理
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)
# X = process_PCA(df_scaled)

def drawElbow():
    '''
    肘部法确定K-means的K值，绘制误差平方变化图
    '''
    inertia = []
    K_range = 10
    for k in range(2, K_range):
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=0).fit(df_scaled)
        inertia.append(np.sqrt(kmeans.inertia_))
    plt.plot(range(2, K_range), inertia, 'o-')
    plt.xlabel('k')
    plt.ylabel('误差平方和')
    plt.savefig('../result/findK_Elbow.png')
    plt.clf()

def drawSilscore():
    '''
    使用轮廓系数法确定K值
    '''
    sil_score = []
    K_range = 10
    for k in range(2, K_range):
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=0).fit(df_scaled)
        sil_score.append(silhouette_score(X, kmeans.labels_))
    plt.plot(range(2, K_range), sil_score, 'o-')
    plt.xlabel('k')
    plt.ylabel('轮廓系数')
    plt.savefig('../result/findK_Silscore.png')
    plt.clf()

# drawElbow()
# drawSilscore()

kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
k_fit = kmeans.fit(X)
predictions = k_fit.labels_

def drawCluster_2D():
    '''
    数据PCA降维至2D，绘制聚类图像
    :return:
    '''
    pca = PCA(n_components=2)
    pca.fit(X)
    df_pca = pca.transform(df_scaled)
    df_pca = pd.DataFrame(df_pca, columns=[f'PCA{i}' for i in range(1, len(pca.components_) + 1)])
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue=predictions, palette='tab10', alpha=0.8)
    plt.savefig('../result/cluster_PCA2D')
    plt.clf()

def drawCluster_3D():
    '''
    数据PCA降维至3D，绘制聚类图像
    :return:
    '''
    pca = PCA(n_components=3)
    pca.fit(X)
    df_pca = pca.transform(df_scaled)
    df_pca = pd.DataFrame(df_pca, columns=[f'PCA{i}' for i in range(1, len(pca.components_) + 1)])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = df_pca['PCA1']
    y = df_pca['PCA2']
    z = df_pca['PCA3']
    ax.scatter(x, y, z, c=predictions, cmap="jet", marker="o")
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.savefig('../result/cluster_PCA3D')
    plt.clf()

drawCluster_2D()
drawCluster_3D()
