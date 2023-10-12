import csv
import os
import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression



class YearData:
    def __init__(self, year: str, data: List[List[str]], analyzer) -> None:
        """
        Initializes a YearData object.

        Args:
            year: A string representing the year of the data.
            data: A list of lists representing the data.
            analyzer: A BMarathonAnalyzer object used for cleaning the data.
        """
        self.year = year
        self.analyzer = analyzer

        # Convert data to a dictionary, clean data, and store it in self.data
        # This prevents us from having to clean the data every time we want to use it
        dct_data = self.lst_to_dct(data)
        cleaned_dct_data = self.clean_numerics(dct_data)
        self.data = cleaned_dct_data

    def get_data(self) -> Dict[str, List[Union[int, str]]]:
        """
        Returns the cleaned data as a dictionary.

        Returns:
            A dictionary containing the cleaned data.
        """
        return self.data

    def get_year(self) -> str:
        """
        Returns the year of the data.

        Returns:
            A string representing the year of the data.
        """
        return self.year

    def lst_to_dct(self, lst: List[List[str]]) -> Dict[str, List[str]]:
        """
        Converts a list of lists to a dictionary.

        Args:
            lst: A list of lists representing the data.

        Returns:
            A dictionary containing the data.
        """
        headers = lst[0]
        dct = {h: [] for h in headers}
        data = lst[1:]
        for row in data:
            for i in range(len(row)):
                dct[headers[i]].append(row[i])

        return dct

    def clean_numerics(self, dct: Dict[str, List[str]]) -> Dict[str, List[Union[int, str]]]:
        """
        Cleans numeric data in the dictionary.

        Args:
            dct: A dictionary containing the data.

        Returns:
            A dictionary containing the cleaned data.
        """
        for key in dct:
            if key == self.analyzer.OFFICIAL_TIME_HEADER:
                dct[key] = [self.convert_to_seconds(time) for time in dct[key]]
            else:
                try:
                    dct[key] = [self.clean_numeric(s) for s in dct[key]]
                except:
                    pass
        return dct

    def clean_numeric(self, s: str) -> int:
        """
        Cleans a numeric string.

        Args:
            s: A string representing a numeric value.

        Returns:
            An integer representing the cleaned numeric value.
        """
        s = s.replace("$", "")
        s = s.replace("%", "")
        s = s.replace(",", "")
        return int(s)

    def convert_to_seconds(self, time: str) -> int:
        """
        Converts a time string to seconds.

        Args:
            time: A string representing a time in the format "HH:MM:SS".

        Returns:
            An integer representing the time in seconds.
        """
        hours, minutes, seconds = map(int, time.split(":"))
        return hours * 3600 + minutes * 60 + seconds


class BMarathonAnalyzer:
    """
    A class for analyzing Boston Marathon data.
    """

    def __init__(self, target_directory: str) -> None:
        """
        Initializes a BMarathonAnalyzer object.

        Args:
            target_directory: A string representing the path to the target directory.
        """
        self.current_file_directory: str = os.path.dirname(__file__)
        self.parent_directory: str = self.current_file_directory + "/"
        self.target_directory_path: str = self.parent_directory + target_directory
        self.RANK_HEADER: str = "RankOverall"
        self.NAME_HEADER: str = "FullName"
        self.BIB_HEADER: str = "BibNumber"
        self.AGE_HEADER: str = "AgeOnRaceDay"
        self.GENDER_HEADER: str = "Gender"
        self.CITY_HEADER: str = "City"
        self.STATE_HEADER: str = "StateAbbrev"
        self.COUNTRY_CITIZEN_HEADER: str = "CountryOfCtzAbbrev"
        self.CITIZENSHIP_HEADER: str = "CitizenOf"
        self.OFFICIAL_TIME_HEADER: str = "OfficialTime"
        self.RANK_HEADER: str = "RankOverall"
        self.GENDER_RANK_HEADER: str = "RankGender"
        self.YEARS: List[int] = []
        # gets all the files in the target directory
        files: List[str] = self.get_files()

        # creates a list of YearData objects
        self.data: List[YearData] = []

        for f in files:
            year, data = self.read_csv(f)
            self.data.append(YearData(year, data, self))
            self.YEARS.append(int(year))

    def get_files(self) -> List[str]:
        """
        Gets all the files in the target directory.

        Returns:
            A list of strings representing the filenames in the target directory.
        """
        files: List[str] = os.listdir(self.target_directory_path)
        return files

    def read_csv(self, filename: str) -> Tuple[str, List[List[str]]]:
        """
        Reads a CSV file and returns the year and data.

        Args:
            filename: A string representing the filename.

        Returns:
            A tuple containing a string representing the year and a list of lists representing the data.
        """
        # regex the year from the filename
        year: str = re.search(r"\d{4}", filename).group()
        data: List[List[str]] = []
        path_to_file: str = self.target_directory_path + "/" + filename
        with open(path_to_file, "r") as infile:
            csvfile = csv.reader(infile)
            for row in csvfile:
                data.append(row)

        return year, data

    def get_data_year(self, target_year: int) -> Dict[str, List[str]]:
        """
        Gets the data for a specific year.

        Args:
            target_year: An integer representing the target year.

        Returns:
            A dictionary containing the data for the target year.
        """
        # get all the data
        for year_data in self.data:
            if int(year_data.get_year()) == int(target_year):
                return year_data.get_data()

    def order_data(self, data: Dict[str, List[str]], feature: str, ascending: bool = True) -> Dict[str, List[str]]:
        """
        Orders the data based on a feature.

        Args:
            data: A dictionary containing the data.
            feature: A string representing the feature to order by.
            ascending: A boolean representing whether to order in ascending or descending order.

        Returns:
            A dictionary containing the ordered data.
        """
        # get the feature data
        feature_data: List[str] = data[feature]
        # get the ordered indices based on sorted feature data
        ordered_indices: List[int] = []
        if ascending:
            ordered_indices = sorted(range(len(feature_data)), key=lambda k: feature_data[k])
        else:
            # descending order, so we negate the feature data
            ordered_indices = sorted(range(len(feature_data)), key=lambda k: -feature_data[k])

        # create a new dictionary with the ordered data
        ordered_data: Dict[str, List[str]] = {}
        for key, value in data.items():
            ordered_data[key] = [value[i] for i in ordered_indices]
        return ordered_data

    def get_stats_tuple(self, data: List[str]) -> Tuple[float, float]:
        """
        Calculates the mean and median of a list of data.

        Args:
            data: A list of strings representing the data.

        Returns:
            A tuple containing the mean and median of the data.
        """
        mean: float = statistics.mean(data)
        median: float = statistics.median(data)
        return mean, median

    def filter_data(self, data: Dict[str, List[str]], feature: str, targeted_feature_value: str) -> Dict[str, List[str]]:
        """
        Filters the data based on a feature and a targeted feature value.

        Args:
            data: A dictionary containing the data.
            feature: A string representing the feature to filter by.
            targeted_feature_value: A string representing the targeted feature value.

        Returns:
            A dictionary containing the filtered data.
        """
        filtered_data: Dict[str, List[str]] = {}
        targeted_indices: List[int] = []
        for i in range(len(data[feature])):
            if data[feature][i] == targeted_feature_value:
                targeted_indices.append(i)
        for key, value in data.items():
            filtered_data[key] = [value[i] for i in targeted_indices]
        return filtered_data

    def get_top_rank_x(self, data: Dict[str, List[str]], x: int) -> Dict[str, List[str]]:
        """
        Gets the top x ranked data.

        Args:
            data: A dictionary containing the data.
            x: An integer representing the number of top ranked data to get.

        Returns:
            A dictionary containing the top x ranked data.
        """
        # get the top 100
        data = self.order_data(data, self.RANK_HEADER, True)
        # get the top 100 rows
        top_data: Dict[str, List[str]] = {}
        for key, value in data.items():
            top_data[key] = value[:x]
        return top_data

    def count_feature_occurrences(self, data: Dict[str, List[str]], feature_name: str) -> Dict[str, int]:
        """
        Counts the number of occurrences of a feature in the data.

        Args:
            data: A dictionary containing the data.
            feature_name: A string representing the name of the feature to count.

        Returns:
            A dictionary containing the count of the feature.
        """
        # Get the list of feature values from the data
        feature_values: List[str] = data[feature_name]

        # Count the occurrences of each feature value
        feature_counts: Counter = Counter(feature_values)

        return feature_counts

    def seconds_to_time_format(self, seconds: float) -> str:
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

    def get_years_data(self, years: List[int]) -> List[Dict[str, List[str]]]:
        """
        Gets the data for a list of years.

        Args:
            years: A list of integers representing the years.

        Returns:
            A list of dictionaries containing the data for each year.
        """
        # sort from least to greatest year
        years.sort()
        years_data: List[Dict[str, List[str]]] = []
        for year in years:
            years_data.append(self.get_data_year(year))
        return years_data

    def get_feature_metric_over_years(self, metric_name: str, feature_name: str, filter_feature_names: List[str], filter_feature_values: List[str], years: List[int], top_x_ranked: int = None) -> List[float]:
        """
        Gets a feature metric over a list of years.

        Args:
            metric_name: A string representing the name of the metric to calculate.
            feature_name: A string representing the name of the feature to calculate the metric for.
            filter_feature_names: A list of strings representing the names of the features to filter by.
            filter_feature_values: A list of strings representing the targeted feature values to filter by.
            years: A list of integers representing the years to calculate the metric for.
            top_x_ranked: An integer representing the number of top ranked data to use.

        Returns:
            A list of floats representing the metric for each year.
        """
        feature_metrics: List[float] = []
        years_data: List[Dict[str, List[str]]] = self.get_years_data(years)

        for i in range(len(years_data)):
            year_data = years_data[i]
            print(years[i])
            
            if top_x_ranked is not None:
                # Get the top x ranked data for each year
                year_data = self.get_top_rank_x(year_data, top_x_ranked)

            
            # set the filtered data to the year data, then filter it
            filtered_data: Dict[str, List[str]] = year_data

            for i in range(len(filter_feature_names)):
                filter_feature_name: str = filter_feature_names[i]
                filter_feature_value: str = filter_feature_values[i]
                filtered_data = self.filter_data(filtered_data, filter_feature_name, filter_feature_value)

            metric_value: float = 0
            print("FEATURE NAME")
            print(filtered_data[feature_name])

            if metric_name == "mean":
                metric_value = self.get_stats_tuple(filtered_data[feature_name])[0]
            elif metric_name == "median":
                metric_value = self.get_stats_tuple(filtered_data[feature_name])[1]
            feature_metrics.append(metric_value)

        return feature_metrics

    def regression_plot(self, xvals: List[float], yvals: List[float], x_label: str, y_label: str, title: str) -> None:
        """
        Creates a regression plot.

        Args:
            xvals: A list of floats representing the x values.
            yvals: A list of floats representing the y values.
            x_label: A string representing the x-axis label.
            y_label: A string representing the y-axis label.
            title: A string representing the title of the plot.
        """
        fg = sns.regplot(x=xvals, y=yvals, scatter_kws={'s': 10, "color": "black"}, line_kws={"color": "blue"})
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        fg.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: self.seconds_to_time_format(x)))
        fg.set_xticks(xvals)
        fg.set_xticklabels([str(int(x)) for x in xvals], rotation=0)
        fg.legend(labels=["Data Points", "Regression Line"])
        plt.show()

    def linear_regression(self, xvals: List[float], yvals: List[float], target_x: float) -> str:
        """
        Performs a linear regression and predicts a target value.

        Args:
            xvals: A list of floats representing the x values.
            yvals: A list of floats representing the y values.
            target_x: A float representing the target x value.

        Returns:
            A string representing the predicted target value in time format.
        """
        # Create a linear regression object and fit the data
        lr = LinearRegression()
        x_array = np.array(xvals).reshape(-1, 1)
        y_array = np.array(yvals)
        lr.fit(x_array, y_array)

        # Predict the target value
        target_y = lr.predict([[target_x]])[0]

        # Convert the target value to seconds
        return self.seconds_to_time_format(target_y)

    def max_min_normalize(self, lst: List[float]) -> List[float]:
        """
        Normalizes a list of values.

        Args:
            lst: A list of floats representing the values.

        Returns:
            A list of floats representing the normalized values.
        """
        norm: List[float] = []
        mn: float = min(lst)
        mx: float = max(lst)
        for x in lst:
            n: float = (x - mn) / (mx - mn)
            norm.append(n)
        return norm

    def line_plot(self, xvals: List[float], list_y_vals: List[List[float]], y_val_labels : List[str], x_label: str, y_label: str, title: str) -> None:
        """
        Creates a line plot.

        Args:
            xvals: A list of floats representing the x values.
            list_y_vals: A list of lists representing the y values.
            y_val_labels: A list of strings representing the y value labels.
            x_label: A string representing the x-axis label.
            y_label: A string representing the y-axis label.
            title: A string representing the title of the plot.
        """

        custom_palette = sns.color_palette("husl", len(list_y_vals))

        # Set the custom color palette
        sns.set_palette(custom_palette)
        for i in range(len(list_y_vals)):
            sns.lineplot(x=xvals, y=list_y_vals[i], label=y_val_labels[i])
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # manually set the xticks to get all the years
        plt.xticks(xvals)
        plt.title(title)
        plt.show()

    def get_correlation_with_years(self, years: List[int], feature_values: List[float]) -> float:
        """
        Calculates the correlation between a feature and a list of years.

        Args:
            years: A list of integers representing the years.
            feature_values: A list of floats representing the feature values.

        Returns:
            A float representing the correlation between the feature and the years.
        """
        return round(statistics.correlation(feature_values, years), 4)
    
    

    

bmarathon = BMarathonAnalyzer("marathon_files")

def main():
        # Get data for the year 2013
    data_2013 = bmarathon.get_data_year("2013")

    # Get the top 1000 runners for the year 2013
    top_1000_runners_2013 = bmarathon.get_top_rank_x(data_2013, 1000)

    # Calculate the mean finish time for the top 1000 runners in 2013
    mean_finish_time_2013 = bmarathon.get_stats_tuple(top_1000_runners_2013[bmarathon.OFFICIAL_TIME_HEADER])[0]

    # Print the mean finish time in a human-readable format
    print(bmarathon.seconds_to_time_format(mean_finish_time_2013))


    # Get data for the year 2010
    data_2010 = bmarathon.get_data_year("2010")

    # Get the top 1000 runners for the year 2010
    top_1000_runners_2010 = bmarathon.get_top_rank_x(data_2010, 1000)

    # Calculate the median age of the top 1000 runners in 2010
    median_age_2010 = bmarathon.get_stats_tuple(top_1000_runners_2010[bmarathon.AGE_HEADER])[1]

    # Print the median age
    print(median_age_2010)


    # Get data for the year 2023
    data_2023 = bmarathon.get_data_year("2023")

    # Count the number of runners from each country in the 2023 data
    country_counts_2023 = bmarathon.count_feature_occurrences(data_2023, bmarathon.COUNTRY_CITIZEN_HEADER)

    # Get the name of the second most common country
    second_most_common_country_2023 = country_counts_2023.most_common(2)[1][0]

    # Print the name of the second most common country
    print(second_most_common_country_2023)


    # Get the mean finish times for runners from the USA for each year
    mean_finish_times_usa = bmarathon.get_feature_metric_over_years("mean", bmarathon.OFFICIAL_TIME_HEADER, [bmarathon.COUNTRY_CITIZEN_HEADER], ["USA"], bmarathon.YEARS, 1000)

    # Print the mean finish times
    print(mean_finish_times_usa)

    # Plot the mean finish times for the top 1000 runners over the years
    bmarathon.regression_plot(bmarathon.YEARS, mean_finish_times_usa, "Year", "Mean Finish Time", "Mean Time of Top 1000 Runners Over Years")

    # Print the predicted finish time for the year 2020
    print("MEAN FINISH TIMES")
    print(mean_finish_times_usa)
    print("TARGET TIME")



    print(bmarathon.linear_regression(bmarathon.YEARS, mean_finish_times_usa, 2020))


    # Get the mean finish times for all runners for each year
    mean_finish_times_all = bmarathon.get_feature_metric_over_years("mean", bmarathon.OFFICIAL_TIME_HEADER, [], [], bmarathon.YEARS)

    # Normalize the mean finish times
    normalized_finish_times = bmarathon.max_min_normalize(mean_finish_times_all)

    # Get the median ages for all runners for each year
    median_ages_all = bmarathon.get_feature_metric_over_years("median", bmarathon.AGE_HEADER, [], [], bmarathon.YEARS)


    # Normalize the median ages
    normalized_ages = bmarathon.max_min_normalize(median_ages_all)

    # Plot the normalized mean finish times and median ages over the years
    bmarathon.line_plot(bmarathon.YEARS, [normalized_finish_times, normalized_ages], ["Finish Times", "Ages"], "Year", "Normalized Value", "Mean Finish Time vs Median Age Over Years")



if __name__ == "__main__":
    main()