from numpy import *
import matplotlib.pyplot as plt
import numpy as np

x0 = []
x1 = []
x2 = []
x3 = []
cnt = 0


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    global cnt
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            cnt += 1
            x0.append(weights[0])
            x1.append(weights[1])
            x2.append(weights[2])
            x3.append(weights[3])
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest(Predic_Train, Predic_Test):
    frTrain = Predic_Train
    frTest = Predic_Test
    trainingSet = []
    trainingLabels = []
    for index, row in frTrain.iterrows():
        lineArr = []
        for i in range(1, len(row)):
            lineArr.append(row[i])
        trainingSet.append(lineArr)
        trainingLabels.append(row[0])
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for index, row in frTest.iterrows():
        numTestVec += 1.0
        lineArr = []
        for i in range(1, len(row)):
            lineArr.append(row[i])
        if int(classifyVector(array(lineArr), trainWeights)) != int(row[0]):
            # print(row)
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("trainWeights:", trainWeights, len(trainWeights))
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def draw():
    global cnt
    print("cnt:", cnt)
    x = np.linspace(1, cnt, cnt)
    plt.subplot(4, 1, 1)
    plt.plot(x, x0, color='r')
    plt.subplot(4, 1, 2)
    plt.plot(x, x1, color='b')
    plt.subplot(4, 1, 3)
    plt.plot(x, x2, color='g')
    plt.subplot(4, 1, 4)
    plt.plot(x, x3, color='y')
    plt.show()


def multiTest(Predic_Train, Predic_Test):
    numTests = 1
    errorSum = 0.0
    for k in range(numTests):
        print(f"{k}th Test...")
        errorSum += colicTest(Predic_Train, Predic_Test)
    # draw()
    with open('logRegres.txt', 'w')as fp:
        print(x0[-1], x1[-1], x2[-1], x3[-1], errorSum / float(numTests), file=fp)
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))
