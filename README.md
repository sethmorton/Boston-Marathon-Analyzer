# Boston Marathon Analyzer

The Boston Marathon Analyzer is a Python library that allows you to analyze and visualize data from the Boston Marathon. It provides a simple interface for reading and cleaning data from CSV files, as well as for performing various analyses on the data.

## Installation

To install the Boston Marathon Analyzer, you can use pip:


bash
pip install boston-marathon-analyzer
​


## Usage

To use the Boston Marathon Analyzer, you first need to import the necessary classes:


python
from boston_marathon_analyzer import BMarathonAnalyzer, YearData
​


Next, you can create a BMarathonAnalyzer object and use it to analyze the data:


python
# Create a BMarathonAnalyzer object
analyzer = BMarathonAnalyzer("path/to/target/directory")
​
# Get the data for a specific year
year_data = analyzer.get_data_for_year(2020)
​
# Get the official times for the top 10 finishers in 2020
top_10_times = year_data.get_official_times(10)
​
# Get the average official time for all finishers in 2020
average_time = year_data.get_average_official_time()
​


## Contributing

If you would like to contribute to the Boston Marathon Analyzer, please feel free to submit a pull request on the GitHub repository.

## License

The Boston Marathon Analyzer is licensed under the MIT License.

## Acknowledgements

The Boston Marathon Analyzer uses data from the [Boston Athletic Association](https://www.baa.org/). The data is available for free and can be downloaded from the BAA's website.

The Boston Marathon Analyzer is inspired by the [Boston Marathon Analysis](https://github.com/mattdonders/boston-marathon-analysis) project by Matt Donders. The project provides a comprehensive analysis of the Boston Marathon, including data visualization and statistical analysis.