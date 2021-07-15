import pandas as pd
import mcm_2021 as mcm
import Bayesian
from numpy import *
import re
import LogRegres
import Update
from math import cos, sin, sqrt, pi, atan2

POSITIVEID_PATH = r'E:\Git\Python\mcm\csv\PositiveID.csv'
NEGATIVEID_PATH = r'E:\Git\Python\mcm\csv\NegativeID.csv'
POS_NEG_PATH = r'E:\Git\Python\mcm\csv\pos_neg.csv'


def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dLat = (lat2 - lat1) * pi / 180.0
    dLon = (lon2 - lon1) * pi / 180.0

    a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1 * pi / 180.0) * cos(
        lat2 * pi / 180.0) * sin(dLon / 2) * sin(dLon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    dist = R * c
    return dist


def cal_dir(lon1, lat1, lon2, lat2):
    a = 1 if lon1 >= lon2 else -1
    b = 1 if lat1 <= lat2 else -1
    return a, b


# kind为1 预测模型
# kind为0 分类模型
def extract_val(data):
    # if kind == 1:
    # 先从总体数据中找到每年的positive数据的中心位置
    data_positive = Update.pretreatment(data)
    years, avg_lat, avg_lon = Update.cal_avg_by_year(data_positive)

    # 按着第一题的模型每年平均位置移动
    if len(years) == 0:
        print("agflog")
        years = [2019, 2020]
        avg_lon = [-122.96909459999999, -122.59492977777778]
        avg_lat = [49.024412000000005, 48.959598666666665]
        max_year = 2020
    else:
        max_year = max(years)
    next_lat = avg_lat[years.index(max_year)] - 5.389340107145173
    next_lon = avg_lon[years.index(max_year)] - 15.008311598496988

    years.append(max_year + 1)
    avg_lon.append(next_lon)
    avg_lat.append(next_lat)

    data = mcm.get_year(data)
    Predic = pd.DataFrame()

    # get label
    Predic['label'] = data['Lab Status'].apply(lambda x: 1 if x == 'Positive ID' else 0)

    # get norm_d
    norm_list = []
    norm2_list = []
    for index, row in data.iterrows():
        a, b = 1, 1
        x, y = -29, -29
        if row['year'] in years:
            # 训练模型中使用
            index = years.index(row['year'])
            a, b = cal_dir(row['Longitude'], row['Latitude'], avg_lon[index], avg_lat[index])
            x = haversine(row['Longitude'], 0, avg_lon[index], 0)
            y = haversine(0, row['Latitude'], 0, avg_lat[index])
            # print(index, row['year'], a, b, x, y,avg_lat[index], row['Latitude'])
        # elif row['year'] > max_year:
        #     # 为预测模型中使用
        #     a, b = cal_dir(row['Longitude'], row['Latitude'], next_lon, next_lat)
        #     x = haversine(row['Longitude'], 0, next_lon, 0)
        #     y = haversine(0, row['Latitude'], 0, next_lat)
        val = mcm.norm_d(a * x, b * y) * 10000
        val2 = val ** 2
        norm_list.append(val)
        norm2_list.append(val2)
    Predic['norm'] = norm_list
    # print(Predic['norm'].head())
    Predic['norm2'] = norm2_list

    # get with_photo
    with_photo_list = mcm.natual_join()
    # print("with_photo_list:", len(with_photo_list))
    predic_photo_list, quality_photo_list = mcm.Predict_picture_quality()
    with_photo = []
    for index, row in data.iterrows():
        if row['GlobalID'] in predic_photo_list:
            # print("with_photo:", quality_photo_list[predic_photo_list.index(row['GlobalID'])])
            with_photo.append(quality_photo_list[predic_photo_list.index(row['GlobalID'])])
        elif row['GlobalID'] in with_photo_list:
            with_photo.append(1)
        else:
            with_photo.append(0)
    Predic['with_photo'] = with_photo
    # Predic['with_photo'] = data['GlobalID'].apply(lambda x: 1 if x in with_photo_list else -1)
    # print(Predic['with_photo'].describe())

    # get review score
    bad_p, bad_word, good_p, good_word = Bayesian.Bayesian()
    review = []
    for index, row in data.iterrows():
        line = re.split('[,|.|!| ]', str(row['Notes']))
        p = len(line)
        if p == 0:
            p = -10
        else:
            for word in line:
                if len(word) > 3:
                    if word in good_word:
                        p += good_p[good_word.index(word)] * 10
                    if word in bad_word:
                        p -= bad_p[bad_word.index(word)] * 20
        review.append(p)
    Predic['review'] = review
    # print(Predic['review'].describe())
    return Predic


def predict(df):
    print("start predict...")
    Predic_Test = df.loc[:int(0.30 * len(df)), :]
    Predic_Train = df.loc[int(0.30 * len(df)):len(df), :]
    print("Predic_Train:", Predic_Train.shape, "Predic_Test:", Predic_Test.shape)
    LogRegres.multiTest(Predic_Train, Predic_Test)
    print("Done")


def main():
    df = mcm.read_data('csv/all_data.csv')

    print("Extract feature values from all data")
    Predic = extract_val(df)
    mcm.print_info(Predic)
    df = (Predic - Predic.min()) / (Predic.max() - Predic.min())
    print(df.head())
    mcm.print_info(df)

    predict(df)


if __name__ == '__main__':
    main()
