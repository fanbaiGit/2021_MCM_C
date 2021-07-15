# C:\\Users\\18395\\Documents\\Git\\VScode\\python\\source\\review_body.txt
import numpy as np
import xlrd
from numpy.core import zeros

common = ['Stars', 'stars', 'Five', 'five', 'baby', 'this', 'pacifier', 'This', 'pacifiers', 'microwave', 'have', 'that', 'That', 'with', 'just', 'would', 'your', 
            'were', 'when', 'they', 'what','which', 'than', 'then', 'after', 'from', 'about', '/><br', 'these', 'them', 'because', 'also', 'They', 'time', 'does', 
            'still', 'used', 'hair', 'dryer', ]
product = ['test', 'pacifier', 'microwave', 'hair_dryer']
tot_word = 0
tot_review = 0
tot_voc = 0


def getData(filename):
    workbook = xlrd.open_workbook(filename) 
    sheet = workbook.sheet_by_index(0)
    all_rate = []  
    all_propa = []
    rate = []
    for i in range(1, sheet.nrows):
        all_rate.append(sheet.cell_value(i, 0))
        all_propa.append(sheet.cell_value(i, 6))
    for star in all_rate:
        #print("star: ",star)
        if star <= 3:
            classes = 0
        else:
            classes = 1
        rate.append(classes)
    return rate, all_propa, all_rate


def getReview(filename):
    workbook = xlrd.open_workbook(filename)  # 填路径
    sheet = workbook.sheet_by_index(0)
    all_review_head = []
    all_review_body = []
    reviewHead = []
    for i in range(1, sheet.nrows):
        all_review_head.append(sheet.cell_value(i, 6))
        all_review_body.append(sheet.cell_value(i, 3))
    for sentence in all_review_head:
        word = str(sentence).split(' ')
        # print(word)
        for w in word:
            # print(w)
            if len(w) >= 4:
                if w not in common:
                    reviewHead.append(w)
    # print(reviewHead)
    return all_review_body, all_review_head


def loadDataSet(data):  # 加载评论
    review = []
    for line in data:
        #print("line :", line)
        odom = str(line).split(' ')  # 将单个数据分隔开存好
        for word in odom:
            if len(word) >= 4:
                if word not in common:
                    #print("word :", word)
                    review.append(word)
    # print(review)
    tot_word = len(review)
    print("total word is: ", tot_word)
    tot_review = len(data)
    print("total review is: ", tot_review)
    return review


def creatVocabList(dataSet):  # 创建单词表
    vocabSet = []
    for document in dataSet:
        if document not in vocabSet:
            vocabSet.append(document)
    tot_voc = len(vocabSet)
    print("total vocabulary is: ", tot_voc)
    return vocabSet


def cntVoc(review, keyWord):
    cnt = 0
    for word in review:
        if word in keyWord:
            cnt += 1
    print("The number of keyword in review is:", cnt)
    print("The rate of keyword in review is:", float(cnt/len(review)))
    return cnt


def cntWord(review, mystar, keyWord):
    i = 0
    # print("len.mystar:",len(mystar))
    cnt = np.zeros((len(keyWord), 5))
    for centence in review:
        centence = centence.split(' ')
        # print(centence)
        star = int(mystar[i])
        # print("star:",star)
        for word in centence:
            if word in keyWord:
                # print("word:",word)
                pos = keyWord.index(word)
                # print("pos:",pos)
                cnt[pos][star-1] += 1
        i += 1
    # print(cnt)
    for word in keyWord:
        print("word:", word)
        pos = keyWord.index(word)
        for i in range(5):
            print(i+1, " star:", int(cnt[pos][i]))
    return cnt


def setOfWords2Vec(vocabList, inputSet):
    # print(type(inputSet))
    returnVec = [0]*len(vocabList)
    # print("inputSet:",inputSet)
    sentence = str(inputSet).split(' ')
    # print(sentence)
    for word in sentence:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def sortVocabList(dataSet, vocabList):  # get keyword and it's count
    sortList = [0]*len(vocabList)
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
        print(vocabList[i],":",dataSet.count(vocabList[i]))
    #print("keyword:", keyword)
    return keyword


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # print(numTrainDocs,numWords)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    print("pAb=", pAbusive)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += 1
            #p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += 1
            #p0Denom += sum(trainMatrix[i])
    # print("p1Num:",p1Num,"p0Num:",p0Num)
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive


def Bayesian(myreview, myvoc, myclass):
    trainMat = []
    for postinDoc in myreview:
        trainMat.append(setOfWords2Vec(myvoc, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, myclass)
    arr = np.argsort(p0v)
    brr = arr[-1:-10:-1]
    print("Bad review")
    print(brr)
    for i in brr:
        print(p0v[i])
        print(myvoc[i])
    crr = np.argsort(p1v)
    drr = crr[-1:-10:-1]
    print("Good review")
    print(drr)
    for i in drr:
        print(p1v[i])
        print(myvoc[i])


def DataAnalysis():
    myclass, myreview, myrate = getData(
        "C:\\Users\\18395\\Documents\\Git\\VScode\\python\\source\\useful_datas_without_not_buying_pacifier.xls")
            # useful_datas_without_not_buying_pacifier.xls
            # useful_datas_without_not_buying_microwave.xls
            # useful_datas_without_not_buying_hair_dryer.xls
    # print("Myclass:",myclass)
    # print(myrate)
    myword = loadDataSet(myreview)
    myvoc = creatVocabList(myword)
    #Bayesian(myreview, myvoc, myclass)
    keyWord = sortVocabList(myword, myvoc)
    # cntVec = cntWord(myreview, myrate, keyWord)


def DataSelect():
    reviewBody,  reviewHead = getReview(
        "C:\\Users\\18395\\Documents\\Git\\VScode\\python\\\source\\useful_datas_without_not_buying_hair_dryer.xls")
    # useful_datas_without_not_buying_pacifier.xls
    # useful_datas_without_not_buying_microwave.xls
    # useful_datas_without_not_buying_hair_dryer.xls
    print("reviewBody:")
    myReviewBody = loadDataSet(reviewBody)
    myVocBody = creatVocabList(myReviewBody)
    print("\nreviewHead:")
    myReviewHead = loadDataSet(reviewHead)
    myVocHead = creatVocabList(myReviewHead)
    print('\n')
    myReview = myReviewBody+myReviewHead
    # print(myReviewBody+myReviewHead)
    myVoc = creatVocabList(myReview)
    keyWord = sortVocabList(myReview, myVoc)
    print("reviewBody:")
    cnt1 = cntVoc(myReviewBody, keyWord)
    print("\nreviewHead:")
    cnt2 = cntVoc(myReviewHead, keyWord)

    # print(reviewHead)

print("Writed by: fb")
# DataSelect()
DataAnalysis()
