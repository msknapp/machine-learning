import os
import numpy as np
import pandas as pd
from home_prices.dataload import load_training_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import math

features, values = load_training_data()
ok = features['GrLivArea'] + features['TotalBsmtSF'] < 6000
features = features[ok]
values = values[ok]


def get_neighborhood_colors(column: str = "Neighborhood"):
    # Could also doc olumn = 'MSZoning'
    x: pd.Series = features[column]
    enc = OrdinalEncoder()
    colors = enc.fit_transform(x.to_numpy().reshape([len(x), 1]))
    return colors


def get_prices(take_log: bool = False, thousands: bool = False):
    y = values
    if thousands:
        y = y / 1000.0
    if take_log:
        y = np.log(y)
    return y


def get_total_area(thousands: bool = False):
    den = 1000.0 if thousands else 1.0
    return (features['GrLivArea'] + features['TotalBsmtSF']) / den


def get_overall_quality(squared: bool = False, take_log: bool = False, get_exp: bool = False):
    x = features['OverallQual']
    if squared:
        x = np.square(x)
    if take_log:
        x = np.log(x)
    if get_exp:
        x = np.exp(x)
    return x


def plot_price_vs_quality_violin(ax):
    violin_data = []
    oq = features["OverallQual"]
    for i in range(1, 11):
        indices = oq == i
        y = values[indices]
        # sqf = features['GrLivArea'] + features['TotalBsmtSF']
        # sqf = sqf[indices]
        y = np.log(y)
        violin_data.append(y)

    ax.violinplot(violin_data)
    ax.set_title("price vs. overall quality")
    ax.set_xlabel("overall quality")
    ax.set_ylabel("log of price")
    ax.yaxis.grid(True)


def plot_price_vs_quality(ax):
    oq = features["OverallQual"]
    x = np.exp(oq / 10.0)
    y = get_prices(take_log=False, thousands=True)
    ax.scatter(x, y)
    ax.set_title("price vs. overall quality")
    ax.set_xlabel("log (overall quality)")
    ax.set_ylabel("price in thousands")
    ax.yaxis.grid(True)


def plot_lot_area_vs_area(ax):
    total_area = get_total_area(True)
    neighborhood_colors = get_neighborhood_colors()
    axs[1, 1].scatter(total_area * np.log(features['OverallQual']) / 100.0, values, s=np.sqrt(features['LotArea']),
                      c=neighborhood_colors)


def plot_price_vs_area(ax):
    total_area = get_total_area(thousands=True)
    # total_bedrooms = features['BedroomAbvGr']
    # q = get_overall_quality(squared=False, take_log=True)
    ax.scatter(total_area, get_prices(take_log=False, thousands=True), c=features['GarageCars'])
    ax.set_title("price vs. area")
    ax.set_xlabel("area in 1000 ft2")
    ax.set_ylabel("price in thousands")
    ax.grid()


def plot_price_vs_combination(ax):
    total_area = get_total_area(thousands=False) / 6000.0
    overall_qual = get_overall_quality(squared=False, take_log=False, get_exp=False)
    oq = np.exp(overall_qual / 10.0) / math.e
    age = features['YrSold'] - features['YearBuilt']
    age_factor = np.exp(-age/65)
    # total_bedrooms = features['BedroomAbvGr']
    # q = get_overall_quality(squared=False, take_log=True)
    ax.scatter(total_area * oq * age_factor, get_prices(take_log=False, thousands=True), c=features['GarageCars'])
    ax.set_title("price vs. area")
    ax.set_xlabel("area in 1000 ft2")
    ax.set_ylabel("price in thousands")
    ax.grid()


def get_data_over_months(per_log_quality: bool = False, per_square_foot: bool = False, thousands: bool = False):
    total_area = get_total_area(thousands=True)
    avg_by_month = []
    for year in range(2006, 2011):
        for month in range(1, 13):
            filter1 = features["YrSold"] == year
            filter2 = features["MoSold"] == month
            v = values[filter1]
            v = v[filter2]
            if thousands:
                v /= 1000.0
            if per_log_quality:
                f1 = features['OverallQual'][filter1][filter2]
                v = v / (np.log(f1))
            if per_square_foot:
                f2 = total_area[filter1][filter2]
                v /= f2

            count = len(v)
            avg = np.average(v)
            tm = (12.0 * (year - 2006) + month) / 12.0
            avg_by_month.append([tm, avg, count])
    df = pd.DataFrame(avg_by_month, dtype=float, columns=['time', 'value', 'count'])
    return df


def plot_price_over_time(ax):
    abm = get_data_over_months(True, True, False)

    ax.scatter(abm['time'], abm['value'], s=abm['count'])
    ax.set_title("average price over the years")
    ax.set_xlabel("Year")
    ax.set_ylabel("average price in 1000s USD")
    ax.grid()


def plot_price_vs_age(ax):
    age = features['YrSold'] - features['YearBuilt']
    sqf = get_total_area(thousands=True)
    qual = get_overall_quality(squared=False, take_log=False, get_exp=True)
    prices = get_prices(take_log=False, thousands=True)
    y = prices
    x = np.exp(-age/65)
    ax.scatter(x, y)
    ax.set_title("price related to age")
    ax.set_xlabel("e ^ (-age / 65)")
    ax.set_ylabel("price in thousands of USD")
    ax.grid()


fig, axs = plt.subplots(nrows=2, ncols=2)
plot_price_vs_age(axs[0, 0])
plot_price_vs_combination(axs[0, 1])
plot_price_vs_area(axs[1, 0])
plot_price_vs_quality(axs[1, 1])
fig.show()
print("ok")
