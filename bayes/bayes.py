#coding=utf-8
from numpy import *

# 创建一些实验样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 创建一个包含所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 词表到向量的转换
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: % is not in my Vocabulary!" % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 拉布拉斯平滑
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0 + (numTrainDocs - sum(trainCategory))
    p1Denom = 2.0 + sum(trainCategory)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
        else:
            p0Num += trainMatrix[i]
    p1Vec = log(p1Num/p1Denom)
    p0Vec = log(p0Num/p0Denom)
    return p1Vec, p0Vec, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2classify, p1Vec, p0Vec, pClass1):
    p1 = sum(vec2classify * p1Vec) + log(pClass1)
    p0 = sum(vec2classify * p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else: return 0

def testingNB():
    listOPosts, listClasses = loadDataSet() # 导入数据
    myVocabList = createVocabList(listOPosts) # 不含重复的词汇list
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p1V, p0V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmatian']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p1V, p0V, pAb))
    testEntry = ['love', 'stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p1V, p0V, pAb))

# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p1V, p0V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p1V, p0V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount)/len(testSet))

