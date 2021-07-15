import re
from numpy import random
import pandas as pd
import mcm_2021 as mcm
import numpy as np
from numpy.core import zeros

DATASET_FILE_PATH = r'E:\Git\Python\mcm\2021_MCM_Problem_C_Data\2021MCMProblemC_DataSet.csv'
common = ['with', 'hornet', 'this', 'have', 'were', 'when', 'they', 'what', 'which', 'This', 'then', 'after', 'from',
          'about', '/><br', 'these', 'them', 'because', 'also', 'They', 'that']


def setOfWords2Vec(vocabList, inputSet):
    # print(type(inputSet))
    returnVec = [0] * len(vocabList)
    # print("inputSet:",inputSet)
    # sentence = str(inputSet).split(' ')
    sentence = re.split('[,|.|!| ]', str(inputSet))
    # print(sentence)
    for word in sentence:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def sortVocabList(dataSet, vocabList):  # get keyword and it's count
    sortList = [0] * len(vocabList)
    for word in vocabList:
        cnt = dataSet.count(word)
        wordPos = vocabList.index(word)
        # print(wordPos)
        sortList[wordPos] = cnt
    arr = np.asarray(sortList)
    brr = np.argsort(arr)
    keyword = []
    cntArr = brr[-1:-50:-1]
    for i in cntArr:
        keyword.append(vocabList[i])
        print(vocabList[i], ":", dataSet.count(vocabList[i]))
    # print("keyword:", keyword)
    return keyword


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # print(numTrainDocs,numWords)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    print("pAb=", pAbusive)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += 1
            # p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == -1:
            p0Num += trainMatrix[i]
            p0Denom += 1
            # p0Denom += sum(trainMatrix[i])
        else:
            # 对于未确定以及未处理的数据按一定概率判断
            p = random.randint(10) + trainCategory[i] * 10
            if p > 10:
                p1Num += trainMatrix[i]
                p1Denom += 1
            else:
                p0Num += trainMatrix[i]
                p0Denom += 1
    # print("p1Num:",p1Num,"p0Num:",p0Num)
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def Bayesian():
    print("Comment processing is in progress...")
    myreview, myvoc, myclass = pretreatment()
    trainMat = []
    for postinDoc in myreview:
        trainMat.append(setOfWords2Vec(myvoc, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, myclass)
    arr = np.argsort(p0v)
    brr = arr[-1:-10:-1]
    bad_word = []
    bad_p = []
    # print("Bad Notes")
    # print(brr)
    for i in brr:
        # print(p1v[i], myvoc[i])
        bad_p.append(p0v[i])
        bad_word.append(myvoc[i])
    crr = np.argsort(p1v)
    drr = crr[-1:-10:-1]
    # print("Good Notes")
    # print(drr)
    good_word = []
    good_p = []
    for i in drr:
        # print(p1v[i], myvoc[i])
        good_p.append(p1v[i])
        good_word.append(myvoc[i])
    return bad_p, bad_word, good_p, good_word


def get_Notes(data):
    my_Notes = list(data['Notes'])
    my_Voc = []
    my_Word = []
    for sentence in my_Notes:
        # word_list = str(sentence).split(' ')
        word_list = re.split('[,|.| ]', str(sentence))
        for word in word_list:
            if len(word) > 3:
                if word not in common:
                    my_Word.append(word)
                    if word not in my_Voc:
                        my_Voc.append(word)
    return my_Notes, my_Voc, my_Word


def pretreatment():
    data_set = mcm.read_data(DATASET_FILE_PATH)
    data_set['class'] = data_set['Lab Status'].apply(
        lambda x: 1 if x == 'Positive ID' else 0 if x == 'Unprocessed' else -1 if x == 'Negative ID' else 0.5)
    my_Notes, my_Voc, my_Word = get_Notes(data_set)
    return my_Notes, my_Voc, list(data_set['class'])


if __name__ == '__main__':
    Bayesian()
