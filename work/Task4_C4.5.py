import math
import operator
import pandas as pd
from Task4_treePlot import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 加载数据集
def createDataSet():
    data = pd.read_csv('../NBA_Season_Stats.csv')
    X_columns = ['Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'Pos']
    X = data[X_columns]
    label_encoder = LabelEncoder()
    X['Pos'] = label_encoder.fit_transform(X['Pos'])
    dataSet = X.values.tolist()
    labels = ['Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK']
    return dataSet, labels, label_encoder

# 计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  # 为所有可能分类创建字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # 以2为底数求对数
    return shannonEnt

# 依据特征划分数据集  axis代表第几个特征  value代表该特征所对应的值  返回的是划分后的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择信息增益比最大属性作为分裂节点
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainrate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历特征 第i个
        featureSet = set([example[i] for example in dataSet])  # 第i个特征取值集合
        newEntropy = 0.0
        splitinfo = 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 该特征划分所对应的entropy
            splitinfo -= prob * math.log(prob, 2)

        if not splitinfo:
            splitinfo = -0.99 * math.log(0.99, 2) - 0.01 * math.log(0.01, 2)
        infoGain = baseEntropy - newEntropy
        infoGainrate = float(infoGain) / float(splitinfo) # 信息增益比
        if infoGainrate > bestInfoGainrate:
            bestInfoGainrate = infoGainrate
            bestFeature = i
    return bestFeature

# 创建树的函数代码   python中用字典类型来存储树的结构 返回的结果是myTree-字典
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分  返回类标签-叶子节点
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有的特征时返回出现次数最多的

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 得到的列表包含所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    # print(myTree)
    return myTree

# 多数表决的方法决定叶子节点的分类 ----  当所有的特征全部用完时仍属于多类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 排序函数 operator中的
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 自底向上剪枝
def prune_downtoup(inputTree, dataSet, featLabels, count):
    # global num
    firstStr =  list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():  # 走到最深的非叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            tempcount = []  # 本将的记录
            rightcount = 0
            wrongcount = 0
            tempfeatLabels = featLabels[:]
            subDataSet = splitDataSet(dataSet, featIndex, key)
            tempfeatLabels.remove(firstStr)
            getCount(secondDict[key], subDataSet, tempfeatLabels, tempcount)
            tempnum = 0.0
            wrongnum = 0.0
            old = 0.0
            # 标准误差
            standwrong = 0.0
            for var in tempcount:
                tempnum += var[0] + var[1]
                wrongnum += var[1]
            old = float(wrongnum + 0.5 * len(tempcount)) / float(tempnum)
            standwrong = math.sqrt(tempnum * old * (1 - old))
            # 假如剪枝

            new = float(wrongnum + 0.5) / float(tempnum)
            if tempnum*new <= tempnum*old + standwrong :  # 要确定新叶子结点的类别

                # 误判率最低的叶子节点的类为新叶子结点的类

                # 在count的每一个列表类型的元素里再加一个标记类别的元素。

                wrongtemp = 1.0
                newtype = -1
                for var in tempcount:
                    if float(var[1] + 0.5) / float(var[0] + var[1]) < wrongtemp:
                        wrongtemp = float(var[1] + 0.5) / float(var[0] + var[1])
                        newtype = var[-1]
                secondDict[key] = str(newtype)
                tempcount = []  # 这个相当复杂，因为如果发生剪枝，才会将它置空，如果不发生剪枝，那么应该保持原来的叶子结点的结构

            for var in tempcount:
                count.append(var)

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            continue
        rightcount = 0
        wrongcount = 0
        subDataSet = splitDataSet(dataSet, featIndex, key)
        for eachdata in subDataSet:
            if str(eachdata[-1]) == str(secondDict[key]):
                rightcount += 1
            else:
                wrongcount += 1
        count.append([rightcount, wrongcount, secondDict[key]])  # 最后一个为该叶子结点的类别

# 计算任意子树正确率
def getCount(inputTree, dataSet, featLabels, count):
    # global num
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        rightcount = 0
        wrongcount = 0
        tempfeatLabels = featLabels[:]
        subDataSet = splitDataSet(dataSet, featIndex, key)
        tempfeatLabels.remove(firstStr)
        if type(secondDict[key]).__name__ == 'dict':
            # 如果是子树结点，递归调用
            getCount(secondDict[key], subDataSet, tempfeatLabels, count)
        else:
            for eachdata in subDataSet:
                if str(eachdata[-1]) == str(secondDict[key]):
                    rightcount += 1
                else:
                    wrongcount += 1
            count.append([rightcount, wrongcount, secondDict[key]])

# 自顶向下剪枝
def prune_uptodown(inputTree, dataSet, featLabels):
    firstStr = list(inputTree.keys()) [0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 如果是子树则计算该结点错误率
            tempfeatLabels = featLabels[:]
            subDataSet = splitDataSet(dataSet, featIndex, key)
            tempfeatLabels.remove(firstStr)
            tempcount = []
            # tempcount保存了所有子树结点正确与错误的个数、以及该子树对应分裂属性
            getCount(secondDict[key], subDataSet, tempfeatLabels, tempcount)
            print(tempcount)
            tempnum, wrongnum, standwrong = 0.0, 0.0, 0.0
            for var in tempcount:
                tempnum += var[0] + var[1]
                wrongnum += var[1]
            treeErr = float(wrongnum + 0.5 * len(tempcount)) / float(tempnum)
            standwrong = math.sqrt(tempnum * treeErr * (1 - treeErr))  # 方差
            # 如果用叶结点代替子树结点
            nodeErr = float(wrongnum + 0.5) / float(tempnum)
            # 判断条件对应公式(2.4)
            if tempnum*nodeErr <= tempnum*treeErr - standwrong:  # 要确定新叶子结点的类别
                # 误判率最低的叶子节点的类为新叶子结点的类
                # 在count的每一个列表类型的元素里再加一个标记类别的元素。
                # print(key,old,new)
                print('cut')
                wrongtemp = 1.0
                newtype = -1
                for var in tempcount:
                    if float(var[0] + var[1]) == 0:
                        continue
                    if float(var[1] + 0.5) / float(var[0] + var[1]) < wrongtemp:
                        wrongtemp = float(var[1] + 0.5) / (float(var[0] + var[1]) )
                        newtype = var[-1]
                secondDict[key] = str(newtype)

# 根据中间值列表将原始数值映射到离散化的数值
def getsortNum(mapfeatures,num):
    if (num < mapfeatures[0] or num > mapfeatures[len(mapfeatures) - 1]):
        return num
    for i in range(len(mapfeatures) - 1):
        if num > mapfeatures[i] and num <= mapfeatures[i + 1]:
            return mapfeatures[i + 1]

# 处理连续型属性
def handleContinuousNumber(dataset, index, num_intervals=20):
    # 提取特定索引下的所有特征值，并对其进行排序
    features = sorted({data[index] for data in dataset})
    # 计算每个区间的边界
    interval_width = (features[-1] - features[0]) / num_intervals
    interval_boundaries = [features[0] + i * interval_width for i in range(num_intervals + 1)]
    # 定义一个函数，将数值映射到区间编号
    def getsortNum(value, boundaries):
        for i, boundary in enumerate(boundaries[1:]):
            if value <= boundary:
                return i
        return num_intervals - 1
    # 更新数据集中的连续值
    for i in range(len(dataset)):
        dataset[i][index] = getsortNum(dataset[i][index], interval_boundaries)

# 预测函数
def predict(example, tree):
    current_tree = tree
    X_columns = ['Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'Pos']
    while isinstance(current_tree, dict):
        key = list(current_tree.keys())[0]
        value = example[X_columns.index(key)]
        if value not in current_tree[key]:
            for i in current_tree[key]:
                current_tree = current_tree[key][i]
                break
        else:
            current_tree = current_tree[key][value]
    return current_tree

# 计算准确率的函数
def calculate_accuracy(dataset, tree, label_encoder):
    correct_predictions = 0
    y_pred = []
    y_test = []
    total_samples = len(dataset)
    for example in dataset:
        prediction = predict(example, tree)
        actual = example[-1]
        if prediction == actual:
            correct_predictions += 1
        y_pred.append(prediction)
        y_test.append(actual)
    accuracy = correct_predictions / total_samples
    def drawConfusionMatrix():
        conf_matrix = confusion_matrix(y_test, y_pred)
        # 绘制混淆矩阵
        ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(values_format='.0f', cmap='Blues')  # 可以根据需要调整参数
        plt.savefig('../result/confusion_matrix_C45.png')
        plt.clf()
    # drawConfusionMatrix()
    return accuracy


if __name__ == '__main__':
    global num
    num = 0
    dataset, features, label_encoder = createDataSet()
    for a in range(len(features)):
        handleContinuousNumber(dataset[:], a, num_intervals=12)  # 1
    X_train, X_test = train_test_split(dataset, test_size=0.2)
    features4uptodown = features.copy()
    tree = createTree(X_train, features)  # 2
    # print(tree)
    createPlot(tree)

    # 自顶向下剪枝
    # prune_uptodown(tree, dataset, features4uptodown)  # 3
    # print(tree)
    # createPlot(tree)

    accuracy = calculate_accuracy(X_test, tree, label_encoder)
    print(f"分类准确率: {accuracy:.2f}")

