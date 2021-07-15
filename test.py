import mcm_2021 as mcm
import Prediction
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
from math import cos, sin, sqrt, atan2, pi

mu1 = 15.008311598496988
mu2 = 5.389340107145173
s1 = 8.403641199246648
s2 = 7.04191983342959
rho = 0.6878240999385998


def get_ellipse(a, b, e_angle, e_x=160.86016534033422, e_y=6.957614820052047):
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * cos(angles)
        or_y = b * sin(angles)
        length_or = sqrt(or_x * or_x + or_y * or_y)
        or_theta = atan2(or_y, or_x)
        new_theta = or_theta + e_angle / 180 * pi
        new_x = e_x + length_or * cos(new_theta)
        new_y = e_y + length_or * sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y


# return 两个数组为x,y
def cal_20():
    lat = 49.0113
    lon = -124.04158

    df = mcm.read_data('csv/PositiveID.csv')
    df = mcm.get_year(df)
    df = df[df['year'] == 2020]

    df['reX'] = df['Longitude'].apply(lambda x: Prediction.haversine(x, 0, lon, 0))
    df['reY'] = df['Latitude'].apply(lambda x: Prediction.haversine(0, x, 0, lat))

    X = np.array(list(df['reX']))
    Y = np.array(list(df['reY']))
    return X, Y


def update_Coefficient(x_this, y_this, x_next, y_next):
    global mu1, mu2, s1, s2, rho
    x1 = np.array(x_this).mean()
    x2 = np.array(x_next)
    y1 = np.array(y_this).mean()
    y2 = np.array(y_next)
    X = x2 - x1
    Y = y2 - y1
    mu1 = np.mean(X)
    mu2 = np.mean(Y)
    s1 = np.std(X)
    s2 = np.std(Y)
    rho = np.linalg.det(np.corrcoef(np.vstack((X, Y))))
    print(f"update Coefficient:mu1:{mu1}, mu2:{mu2}, s1:{s1}, s2:{s2}, rho:{rho}")


def next_year(X, Y, N=1000, k=100, width_min=150, width_max=200, height_min=0, height_max=50):
    x = random.uniform(width_min, width_max, size=N)
    y = random.uniform(height_min, height_max, size=N)
    # print(len(X))

    S = []
    x_next = []
    y_next = []

    # print("cal...")
    for i in range(N):
        s = 0
        for j in range(len(X)):
            p = mcm.norm_d(x[i] - X[j], y[i] - Y[j])
            s += k * p
        if s >= 0.2:
            S.append(s)
            x_next.append(x[i])
            y_next.append(y[i])

    print("number of Wasp:", len(S))
    print("Average survival rate:", np.array(S).mean())
    return x_next, y_next, S


def get_Range(x, y):
    return np.array(x).max(), np.array(x).min(), np.array(y).max(), np.array(y).min()


def draw(x_next, y_next, x_this=None, y_this=None, e=39.97, i=1):
    ellipsis_x1, ellipsis_y1 = get_ellipse(s1, s2, e)
    ellipsis_x2, ellipsis_y2 = get_ellipse(s1 * 2, s2 * 2, e)
    ellipsis_x3, ellipsis_y3 = get_ellipse(s1 * 3, s2 * 3, e)
    avg_x = np.array(x_next).mean()
    avg_y = np.array(y_next).mean()
    x_max, x_min, y_max, y_min = get_Range(x_this, y_this)

    plt.figure(figsize=(20, 10), dpi=100)
    plt.xlim(x_min-10, x_max+20)
    plt.ylim(y_min-10, y_max+20)
    plt.scatter(x_this, y_this, c='g', s=100, alpha=1)
    plt.scatter(x_next, y_next, c='r', s=20, alpha=0.8)
    plt.scatter(avg_x, avg_y, s=400, c='b', marker='*', alpha=1)
    if i == 1:
        plt.plot(ellipsis_x1, ellipsis_y1, c='m')
        plt.plot(ellipsis_x2, ellipsis_y2, c='m')
        plt.plot(ellipsis_x3, ellipsis_y3, c='m')
    plt.xlabel("X", fontdict={'size': 16})
    plt.ylabel("Y", fontdict={'size': 16})
    plt.title(f"Predict the distribution after {i} years", fontdict={'size': 20})
    plt.show()


def Iteration(x_this, y_this, i):
    x_max, x_min, y_max, y_min = get_Range(x_this, y_this)
    x_next, y_next, S_next = next_year(x_this, y_this, width_max=x_max+30, width_min=x_min-30, height_max=y_max+30,
                                       height_min=y_min-30)
    if len(x_next) == 0:
        return x_next, y_next
    update_Coefficient(x_this, y_this, x_next, y_next)
    draw(x_next=x_next, y_next=y_next, x_this=x_this, y_this=y_this, i=i)

    data = pd.DataFrame()
    data['x'] = x_next
    data['y'] = y_next
    data['s'] = S_next
    avg = data['x'].mean()

    data1 = data[data['x'] <= avg]
    data2 = data[data['x'] > avg]
    k = len(data) * 0.40
    n1 = k * len(data1) / (len(data1) + len(data2))
    n2 = k * len(data1) / (len(data1) + len(data2))
    data1 = data1.sort_values(by='s', ascending=False)
    data2 = data2.sort_values(by='s', ascending=False)
    data1 = data1.iloc[:int(n1) + 1, :]
    data2 = data2.iloc[:int(n2) + 1, :]
    data = pd.concat([data1, data2])
    x_next_Eliminated = list(data['x'])
    y_next_Eliminated = list(data['y'])
    print("number of Wasp after Eliminated:", len(x_next_Eliminated))
    return x_next_Eliminated, y_next_Eliminated


N = 5
x_this, y_this = cal_20()
for i in range(N):
    print(f"Predict the distribution after {i+1} year")
    x_next, y_next = Iteration(x_this, y_this, i + 1)
    if len(x_next) == 0:
        print("After the model predicted that the wasp had disappeared, the prediction stopped")
        break
    x_max, x_min, y_max, y_min = get_Range(x_next, y_next)
    print(f"Range of Wasp:[{x_min},{x_max}][{y_min},{y_max}]\n")
    x_this, y_this = x_next, y_next
