import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import Prediction

DATASET_FILE_PATH = r'E:\Git\Python\mcm\2021_MCM_Problem_C_Data\2021MCMProblemC_DataSet.csv'
GlobalID_FILE_PATH = r'E:\Git\Python\mcm\2021_MCM_Problem_C_Data\2021MCM_ProblemC_ Images_by_GlobalID.csv'
PHOTO_FILE_PATH = r'E:\Git\Python\mcm\2021MCM_ProblemC_Files'
PHOTO_Unverified_FILE_PATH = r'E:\Git\Python\mcm\2021MCM_Files'




def Predict_picture_quality():
    filename = []
    quality = []
    GlobalID = []
    tmp = []

    with open('image_result.txt', 'r') as fp:
        while True:
            line = fp.readline()
            if line:
                line = line[:-1]
                if len(line) > 1:
                    if line[0] == '-':
                        line = line.strip('-')
                        # print(line)
                        filename.append(line)
                    elif line[0] == 'a':
                        list = line.split('=')
                        str = list[-1][1:-1]
                        quality.append(float(str))
            else:
                break

    df = pd.read_csv('2021_MCM_Problem_C_Data/2021MCM_ProblemC_ Images_by_GlobalID.csv')
    for index, row in df.iterrows():
        if row['FileName'] in filename:
            GlobalID.append(row['GlobalID'])
            tmp.append(quality[filename.index(row['FileName'])])
    # print(len(GlobalID))

    return GlobalID, tmp


def read_data(FILE_PATH):
    print("read data from ", FILE_PATH)
    df = pd.read_csv(FILE_PATH, encoding='unicode_escape')
    # print(df.describe())
    return df


def ClassificationByFilename(data_set, data_ID=None):
    Status_list = ['Positive ID', 'Unprocessed', 'Negative ID', 'Unverified']
    Pos_data = data_set[data_set['Lab Status'].apply(lambda x: Status_list[0] in x)]
    Unp_data = data_set[data_set['Lab Status'].apply(lambda x: Status_list[1] in x)]
    Neg_data = data_set[data_set['Lab Status'].apply(lambda x: Status_list[2] in x)]
    Unv_data = data_set[data_set['Lab Status'].apply(lambda x: Status_list[3] in x)]

    # _data_set = data_set[data_set['Lab Status'].apply(lambda x: Status_list[2] in x)]
    # GlobalID_list = list(_data_set['GlobalID'])
    # print(len(GlobalID_list))
    # print(GlobalID_list[:10])
    # _data_ID = data_ID[data_ID['GlobalID'].apply(lambda x: x in GlobalID_list)]
    # filename_list = list(_data_ID['FileName'])
    # print(len(filename_list))
    # print(filename_list[:10])

    # for filename in os.listdir(PHOTO_FILE_PATH):
    #     if filename not in filename_list:
    #         os.remove(PHOTO_FILE_PATH + "/" + filename)

    return Pos_data, Unp_data, Neg_data, Unv_data


# `2021mcm_problemc_ images_by_globalid`表中存在一个GlobalID对应多个图片的情况
# 使用mysql命令：
# select count(*),GlobalID
# from `2021mcm_problemc_ images_by_globalid`
# group by GlobalID
# having count(*)>1;
def natual_join():
    data_set = read_data(DATASET_FILE_PATH)
    data_ID = read_data(GlobalID_FILE_PATH)
    GlobalID_list = list(data_ID['GlobalID'])
    _data_set = data_set[data_set['GlobalID'].apply(lambda x: x in GlobalID_list)]
    # print(_data_set.shape)
    with_photo_list = list(_data_set['GlobalID'])
    return with_photo_list


def get_year(data):
    # 只计算2017年以后的数据
    date = list(data['Detection Date'])
    year = []
    for d in date:
        d_list = d.split("/")
        if len(d_list) == 3:
            if len(d_list[0]) > 2:
                year.append(int(d_list[0]))
            else:
                year.append(int(d_list[-1]))
        else:
            year.append(-1)
    data['year'] = year
    # data = data[data['year'].apply(lambda x: 2017 <= x)]
    return data


def pint(r, x, y, c='r'):
    # 点的横坐标为a
    a = np.arange(x - r, x + r, 0.0001)
    # 点的纵坐标为b
    b = np.sqrt(abs(np.power(r, 2) - np.power((a - x), 2)))
    plt.plot(a, y + b, color=c, linestyle='-')
    plt.plot(a, y - b, color=c, linestyle='-')


def draw_plot_by_time(data):
    data = get_year(data)
    Pos_data, Unp_data, Neg_data, Unv_data = ClassificationByFilename(data)
    Pos_data['month'] = pd.to_datetime(Pos_data['Detection Date']).dt.month
    Pos_data1 = Pos_data[Pos_data['year'] == 2019]
    Pos_data1 = Pos_data1[Pos_data1['Longitude'] > -123]
    Pos_data2 = Pos_data[Pos_data['year'] == 2020]
    print(Pos_data2.columns)
    Pos_data3 = Pos_data2.groupby('month')
    # print(Pos_data3.head())
    # month_pos = []
    # for name, group in Pos_data3:
    #     print(type(group))
    #     print(name)
    #     print(group)

    # print(Pos_data1.shape)
    # print(Pos_data2.shape)

    plt.figure(figsize=(20, 10), dpi=100)
    plt.xlim(-123.2, -122)
    plt.ylim(48.5, 49.5)
    plt.plot(list(Pos_data1['Longitude']), list(Pos_data1['Latitude']), 'c*-', linewidth=2)
    plt.plot(list(Pos_data2['Longitude']), list(Pos_data2['Latitude']), 'm.-.', linewidth=1)
    for i in range(len(Pos_data1)):
        # print(float(Pos_data1.iloc[i, [-3]]), "\n", float(Pos_data1.iloc[i, [-4]]))
        pint(0.3, float(Pos_data1.iloc[i, [-3]]), float(Pos_data1.iloc[i, [-4]]), 'r')
    for i in range(len(Pos_data2)):
        pint(0.3, float(Pos_data2.iloc[i, [-3]]), float(Pos_data2.iloc[i, [-4]]), 'g')

    plt.xlabel("Longitude", fontdict={'size': 16})
    plt.ylabel("Latitude", fontdict={'size': 16})
    plt.title("Wasp report schematic", fontdict={'size': 20})
    plt.show()

    def plt_bar_by_year(data):
        years = data.groupby('year')['GlobalID'].count()
        x = [2007, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        plt.figure(figsize=(30, 30), dpi=100)
        plt.bar(x, list(years))
        plt.xlabel("Year", fontdict={'size': 16})
        plt.ylabel("Number", fontdict={'size': 16})
        plt.title("Number of reports per year", fontdict={'size': 20})
        plt.show()

    def plt_plot_by_year(data):
        Latitude_20, Longitude_20, Latitude_19, Longitude_19, Latitude_18, Longitude_18 = [], [], [], [], [], []

        for index, row in data.iterrows():
            if row['year'] == 2020:
                Latitude_20.append(row['Latitude'])
                Longitude_20.append(row['Longitude'])
            elif row['year'] == 2019:
                Latitude_19.append(row['Latitude'])
                Longitude_19.append(row['Longitude'])
            else:
                Latitude_18.append(row['Latitude'])
                Longitude_18.append(row['Longitude'])

        plt.figure(figsize=(30, 30), dpi=100)
        plt.scatter(Longitude_20, Latitude_20, s=30, c='r', alpha=0.4)
        plt.scatter(Longitude_19, Latitude_19, s=70, c='y', alpha=0.6)
        plt.scatter(Longitude_18, Latitude_18, s=60, c='g', alpha=1)

        plt.xlabel("Longitude", fontdict={'size': 16})
        plt.ylabel("Latitude", fontdict={'size': 16})
        plt.title("Wasp report schematic", fontdict={'size': 20})
        plt.show()

    def plt_bar_by_month(data):
        _data = data[data['year'].apply(lambda x: x == 2020)]
        _data['month'] = pd.to_datetime(_data['Detection Date']).dt.month
        print_info(_data)

        cnt_month = _data.groupby('month')['GlobalID'].count()
        x = []
        for i in range(12):
            x.append(i + 1)

        plt.figure(figsize=(30, 30), dpi=100)
        plt.bar(x, list(cnt_month))
        plt.xlabel("month", fontdict={'size': 16})
        plt.ylabel("Number", fontdict={'size': 16})
        plt.title("Number of reports per month in 2020", fontdict={'size': 20})
        plt.show()


def draw_plot_by_Status(data):
    data = get_year(data)
    pos_Latitude, pos_Longitude, unp_Latitude, unp_Longitude, neg_Latitude, neg_Longitude, unv_Latitude, unv_Longitude = [], [], [], [], [], [], [], []

    for index, row in data.iterrows():
        if row['Lab Status'] == 'Positive ID':
            pos_Latitude.append(row['Latitude'])
            pos_Longitude.append(row['Longitude'])
        elif row['Lab Status'] == 'Unprocessed':
            unp_Latitude.append(row['Latitude'])
            unp_Longitude.append(row['Longitude'])
        elif row['Lab Status'] == 'Negative ID':
            neg_Latitude.append(row['Latitude'])
            neg_Longitude.append(row['Longitude'])
        elif row['Lab Status'] == 'Unverified':
            unv_Latitude.append(row['Latitude'])
            unv_Longitude.append(row['Longitude'])

    l = len(pos_Latitude)
    arr_Latitude = np.array(pos_Latitude)
    arr_Longitude = np.array(pos_Longitude)
    avg_Latitude = arr_Latitude.sum() / l
    avg_Longitude = arr_Longitude.sum() / l

    print("avg:", avg_Longitude, avg_Latitude)

    plt.figure(figsize=(30, 30), dpi=100)
    plt.ylim(45, 50)
    plt.xlim(-126, -116)
    plt.scatter(avg_Longitude, avg_Latitude, s=500, c='y', alpha=1, marker='*')
    plt.scatter(pos_Longitude, pos_Latitude, s=30, c='r', alpha=0.6)
    plt.scatter(unp_Longitude, unp_Latitude, s=14, c='y', alpha=0.2)
    plt.scatter(neg_Longitude, neg_Latitude, s=16, c='g', alpha=0.2)
    plt.scatter(unv_Longitude, unv_Latitude, s=8, c='b', alpha=0.2)
    pint(0.1, avg_Longitude, avg_Latitude)
    pint(0.2, avg_Longitude, avg_Latitude)
    pint(0.3, avg_Longitude, avg_Latitude)
    plt.xlabel("Longitude", fontdict={'size': 16})
    plt.ylabel("Latitude", fontdict={'size': 16})
    plt.title("Wasp report schematic", fontdict={'size': 20})
    plt.show()


def get_dis(data):
    data_pos = data[data['Lab Status'].apply(lambda x: x == 'Positive ID')]
    l = len(data_pos)
    # print_info(data)
    avg_Latitude = data_pos['Latitude'].sum() / l
    avg_Longitude = data_pos['Longitude'].sum() / l
    print("avg:", avg_Longitude, avg_Latitude)


def print_info(data):
    print("shape:", data.shape)
    print("describe:", data.describe())
    print("info:", data.info())


def cal_Coefficient():
    # 19年确定点的中心位置
    lat_19 = 48.9931665
    lon_19 = -122.7255847

    # 读取20年确定点的数据
    data = read_data('csv/PositiveID.csv')
    get_year(data)
    data_20 = data[data['year'] == 2020]
    print("data_20:", data_20.shape)

    # 求20年点距离19年中心点的距离
    reX = []
    reY = []
    for index, row in data_20.iterrows():
        reX.append(Prediction.haversine(row['Longitude'], 0, lon_19, 0))
        reY.append(Prediction.haversine(0, row['Latitude'], 0, lat_19))

    Y = np.array(reY)
    X = np.array(reX)

    mu1 = np.mean(X)
    mu2 = np.mean(Y)
    s1 = np.std(X)
    s2 = np.std(Y)
    rho = np.linalg.det(np.corrcoef(np.vstack((X, Y))))

    print(mu1, mu2, s1, s2, rho)
    print(X)
    print(Y)


def norm_d(x1, x2, mu1=15.008311598496988, mu2=5.389340107145173, s1=8.403641199246648, s2=7.04191983342959,
           rho=0.6878240999385998):
    # 方程
    num = ((1) / (2 * np.pi * s1 * s2 * np.sqrt(1 - rho ** 2)))
    A = ((x1 - mu1) ** 2) / (s1 ** 2)
    B = 2 * rho * (((x1 - mu1) * (x2 - mu2)) / (s1 * s2))
    C = ((x2 - mu2) ** 2) / (s2 ** 2)
    D = -1 / (2 * (1 - rho ** 2)) * (A - B + C)
    pdf = num * np.exp(D)
    return pdf


def cal_Integral():
    # 求积分
    print("cal...")
    cnt = 0
    area = 30 * 30 * np.pi
    X, Y = [], []
    x = np.linspace(-30, 30, 300)
    for xi in x:
        r = int(np.sqrt(900 - xi * xi))
        y = np.linspace(-r, r, 20 * r)
        for yi in y:
            cnt += 1
            X.append(xi)
            Y.append(yi)
    X = np.array(X)
    Y = np.array(Y)
    z = norm_d(X, Y)
    areai = area / cnt
    Z = z.sum() * areai
    print("Integral:(Accuracy)", Z)
    print("cnt:", cnt)


def draw_3D():
    # 定义三维数据
    x = np.arange(-30, 30, 0.01)
    y = x
    X, Y = np.meshgrid(x, y)
    R = norm_d(X, Y)
    R1 = norm_d(X, Y, mu1=14.597177385940919, mu2=4.781305181452621, s1=15.337164747003376, s2=12.506264043482428,
                rho=0.5781894448127752)
    R2 = norm_d(X, Y, mu1=10.235100933152076, mu2=2.3520955416246285, s1=20.891126306598224, s2=18.3129061112368,
                rho=0.452421931711422)
    R3 = norm_d(X, Y, mu1=14.489782943989791, mu2=5.309087521805507, s1=20.820856258871498, s2=17.652408372342496,
                rho=0.6080960016028978)
    R4 = norm_d(X, Y, mu1=15.001700455387253, mu2=6.946429137465749, s1=20.208896466474098, s2=17.141221944979872,
                rho=0.6040536952740688)

    r = 30
    u = np.linspace(0, 2 * np.pi, 50)  # 把圆分按角度为50等分
    h = np.linspace(0, 0.0002, 20)  # 把高度1均分为20份
    a = np.sin(u) * r
    b = np.cos(u) * r
    z = 0.000

    # 作图
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X, Y, R, alpha=0.6, cmap=cm.coolwarm)
    ax3.plot(a,b,z)
    cset = ax3.contour(X, Y, R, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    cset = ax3.contour(X, Y, R, 10, zdir='y', offset=30, cmap=cm.coolwarm)


    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.show()


def find():
    Unprocessed = read_data('csv/Unprocessed.csv')
    key = ['syrphid']
    ID_list = list(Unprocessed['GlobalID'])
    print(len(ID_list))
    # for index, row in Unprocessed.iterrows():
    #     line = re.split('[,|.| ]', str(row[1]))
    #     for i in range(len(line)):
    #         if line[i] in key:
    #             ID_list.append(row[0])

    data = read_data(r'2021_MCM_Problem_C_Data/2021MCM_ProblemC_ Images_by_GlobalID.csv')
    filename_list = []
    for index, row in data.iterrows():
        if row[1] in ID_list:
            filename_list.append(row[0])
    print(len(filename_list))

    # FILE_PATH = r'2021MCM_ProblemC_Files'
    # for filename in os.listdir(FILE_PATH):
    #     if filename not in filename_list:
    #         os.remove(FILE_PATH + "/" + filename)


def main():
    data_set = read_data(DATASET_FILE_PATH)
    data_ID = read_data(GlobalID_FILE_PATH)
    print("READ DONE")

    # draw_plot_by_Status(data_set)

    draw_plot_by_time(data_set)

    # ClassificationByFilename(data_set, data_ID)


if __name__ == '__main__':
    draw_3D()
