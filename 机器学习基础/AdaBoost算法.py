import numpy as np
from  sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from pytorch基础知识.图片和视频教学.TorchVision对象检测微调教程 import predictions


# 读取数据
def loadDataSet(fileName):
    numFeat=len((open(fileName).readline().split('\t')))
    dataMat,labelMat=[],[]
    fr=open(fileName)
    for line in fr.readline():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__ == '__main__':
    dataArr,classLabel=loadDataSet('horseColicTraining2.txt')
    testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
    bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm='SAME',estimator=10)
    bdt.fit(dataArr,classLabel)
    predictions=bdt.predict(dataArr)
    errArr=np.mat(np.ones(len(dataArr),1))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabel].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))
