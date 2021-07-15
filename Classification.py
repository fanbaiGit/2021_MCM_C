import pandas as pd
import numpy as np
import LogRegres
import Prediction
import mcm_2021 as mcm

Weights = []

print("get Coefficient")
with open('logRegres.txt', 'r')as fp:
    line = fp.readline()
    list = line[:-1].split(" ")
    for i in range(4):
        Weights.append(float(list[i]))
    print(Weights, len(Weights))

print('pretreatment...')
df = mcm.read_data('csv/all_data.csv')
classify = Prediction.extract_val(df)
_classify = (classify - classify.min()) / (classify.max() - classify.min())
# for index, row in classify.iterrows():
#     print(index, row)
# print(classify.head())
print(_classify.shape)

print('classify...')
with open('Class_Result.txt', 'w') as fp:
    for index, row in classify.iterrows():
        lineArr = []
        for i in range(1, len(row)):
            lineArr.append(row[i])
        # print(lineArr)
        p = sum(np.array(lineArr, dtype='float64') * Weights)
        row[0] = LogRegres.classifyVector(np.array(lineArr), Weights)
        print(f"index:{index} \tlabel:{row[0]} \tpriority:{p}", file=fp)
        # print(f"index:{index} \tlabel:{row[0]} \tpriority:{p}")
print("Done")
