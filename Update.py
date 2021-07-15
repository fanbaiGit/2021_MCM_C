import mcm_2021 as mcm
import numpy as np
import pandas as pd


def get_avg(mdata):
    avg_latitude = mdata['Latitude'].mean()
    avg_longitude = mdata['Longitude'].mean()
    print("year:", mdata.iloc[0, 6], "avg_latitude:", avg_longitude, "avg_longitude", avg_latitude)
    return avg_latitude, avg_longitude


def cal_avg_by_year(data):
    data_grouped = data.groupby('year')
    years = []
    avg_lat = []
    avg_lon = []
    for year, group in data_grouped:
        years.append(year)
        a, b = get_avg(group)
        avg_lat.append(a)
        avg_lon.append(b)
    # print(max(years))
    return years, avg_lat, avg_lon


def pretreatment(data):
    df = data.loc[:, ['GlobalID', 'Detection Date', 'Lab Status', 'Latitude', 'Longitude', 'Notes']]
    df = df[df['Lab Status'] == 'Positive ID']
    df = mcm.get_year(df)
    # print(df.shape)
    return df


if __name__ == '__main__':
    df = mcm.read_data('csv/all_data.csv')
    df = pretreatment(df)
    cal_avg_by_year(df)
