import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import regex as re
from operator import itemgetter


def get_folder():
    return os.getcwd() + "/hw2/marathon_files/"

def get_files():
    return os.listdir(get_folder())
means_all = []
medians = []

def read_files():
    orderedfiles = []
    for file in get_files():

        year = int(re.search(r'\d{4}', file).group())
        orderedfiles.append((file,year))


    orderedfiles.sort(key=lambda x:x[1])
    print(orderedfiles)


    for file, year in orderedfiles:

        
        df = pd.read_csv(get_folder() + file)
        def convert_to_seconds(time):
            hours, minutes, seconds = map(int, time.split(":"))
            return hours * 3600 + minutes * 60 + seconds

        df['OfficialTime'] = df['OfficialTime'].apply(convert_to_seconds)

        usa_data = df.loc[df['CountryOfCtzAbbrev'] == "USA"]
        top_usa_data = usa_data.iloc[0:1000]


        # mean 
        medians.append(df['AgeOnRaceDay'].median())

        means_all.append(df['OfficialTime'].mean())
        

        # print(df.head())
        

read_files()
def seconds_to_time_format(seconds: float) -> str:
        """
        Converts seconds to time format.

        Args:
            seconds: A float representing the number of seconds.

        Returns:
            A string representing the time format.
        """
        hours: int = seconds // 3600
        hours: str = f"{int(hours):02d}"
        minutes: int = (seconds % 3600) // 60
        minutes: str = f"{int(minutes):02d}"
        seconds: int = (seconds % 3600) % 60
        seconds: str = f"{int(seconds):02d}"
        return f"{hours}:{minutes}:{seconds}"

def linear_regression(x_list, y_list, target_x):
        model = LinearRegression()
        lr = LinearRegression()
        x_array = np.array(x_list).reshape(-1, 1)
        y_array = np.array(y_list)
        lr.fit(x_array, y_array)

        # Predict the target value
        target_y = lr.predict([[target_x]])[0]

        # Predict the target value
        # target_y = lr.predict([[target_x]])[0]

        # Convert the target value to second

        return (seconds_to_time_format(target_y), target_y)


years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

def max_min_normalize(lst):
        """
        Normalizes a list of values.

        Args:
            lst: A list of floats representing the values.

        Returns:
            A list of floats representing the normalized values.
        """
        norm =  []
        mn: float = min(lst)
        mx: float = max(lst)
        for x in lst:
            n: float = (x - mn) / (mx - mn)
            norm.append(n)
        return norm

def plot_means():
     means_norm = max_min_normalize(means_all)
     medians_norm = max_min_normalize(medians)

     plt.plot(years, means_norm, label="Means")
     plt.plot(years, medians_norm, label="Medians")
     plt.legend()
     plt.show()

plot_means()


# print("MEANS \n")
# print(means)
# print("\n")

# print("LINEAR REGRESSION \n")
# print(linear_regression(years, means, 2020))












def variance(values, mean):
    return sum([(val-mean)**2 for val in values])

def covariance(x, mean_x, y , mean_y):
    covariance = 0.0
    for r in range(len(x)):
        covariance = covariance + (x[r] - mean_x) * (y[r] - mean_y)
    return covariance

def get_coef(df):
    mean_x = sum(df['x']) / float(len(df['x']))
    mean_y = sum(df['y']) / float(len(df['y']))
    variance_x = variance(df['x'], mean_x)
    #variance_y = variance(df['y'], mean_y)
    covariance_x_y = covariance(df['x'],mean_x,df['y'],mean_y)
    m = covariance_x_y / variance_x
    c = mean_y - m * mean_x
    return m,c

def get_y(x,m,c):
    return m*x+c
