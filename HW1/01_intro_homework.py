import numpy
# You need to install Python, NumPy, Pandas, Matplotlib and Seaborn.
# For that, you can use the instructions from 06-environment.md.

import pandas as pd
import numpy as np
import matplotlib as mplib
import seaborn

if __name__ == '__main__':
    filename = 'car_fuel_efficiency.csv'

    # check the version of pandas installed
    print(f"Pandas Version: {pd.__version__}")

    '''
    Read the provided CSV into the dataframe. It will have the following columns:
    engine_displacement: Integer
    num_cylinders: Float
    horsepower: Float
    vehicle_weight: Float
    acceleration: Float
    model_year: Integer
    origin: String, country of origin
    fuel_type: String
    drivetrain: String
    num_doors: Float, can go negative
    fuel_efficiency_mpg: Float
    '''
    car_df = pd.read_csv(filename)

    # How many records are in the dataset?
    print(f"Number of records using df.shape: {car_df.shape[0]}") # df.shape[1] gets number of columns

    # How many fuel types are presented in the dataset?
    print(f"Number of fuel types present: {car_df['fuel_type'].nunique()}")

    # How many columns in the dataset have missing values?
    # isnull returns a set of all the values in the dataframe, any() checks if any of them are null, sum() totals them.
    null_columns = car_df.isnull().any().sum()
    print(f"Number of columns missing values in the dataset: {null_columns}")

    # What's the maximum fuel efficiency of cars from Asia?
    filter_asia = car_df[car_df['origin'] == 'Asia']
    print(f"Maximum fuel efficiency of cars from Asia: {filter_asia['fuel_efficiency_mpg'].max()}")

    '''
    *** Median value of horsepower ***
    1.) Find the median value of horsepower column in the dataset.
    2.) Next, calculate the most frequent value of the same horsepower column.
    3.) Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.
    4.) Now, calculate the median value of horsepower once again.
        
        Has the median horsepower changed?
            Yes, it increased
            Yes, it decreased
            No
    '''

    print(f"Median horsepower of dataset before filling nulls: {car_df['horsepower'].median()}")
    # The value that appears most in a dataset is the mode, we can grab that using pandas
    most_freq_hp = car_df['horsepower'].mode()

    # now, mode() will return a series because there might be more than one mode.

    # we need to pick a value. We are going to delete any zeroes as they won't help
    most_freq_hp = most_freq_hp[most_freq_hp != 0]

    # After, we will take the average (mean) of the modes (but this is really for edge cases)
    average_mode = most_freq_hp.mean()

    # fill all null values on horsepower column
    car_df['horsepower'] = car_df['horsepower'].fillna(average_mode)
    print(f"Median horsepower of dataset after filling nulls: {car_df['horsepower'].median()}")

    '''
    *** Linear Regression ***
    1.) Select all the cars from Asia
    2.) Select only columns vehicle_weight and model_year
    3.) Select the first 7 values
    4.) Get the underlying NumPy array. Let's call it X.
    5.) Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
    6.) Invert XTX.
    7.) Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
    8.) Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
    9.) What's the sum of all the elements of the result?
    '''

    # we can just use our filter_asia df that we made earlier for step one
    sub_df = filter_asia[['vehicle_weight', 'model_year']] # 2
    array = sub_df.head(7).to_numpy() # 3 and 4
    xtx = np.dot(array.T, array) # 5
    inverted_xtx = np.linalg.inv(xtx) # 6
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200]) # 7
    w = np.dot(np.dot(inverted_xtx, array.T), y) # 8
    sum_of_all_results = np.sum(w) # 9
    print(f"Sum of linear regression: {sum_of_all_results}")