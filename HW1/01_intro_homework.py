
# You need to install Python, NumPy, Pandas, Matplotlib and Seaborn.
# For that, you can use the instructions from 06-environment.md.

import pandas as pd
import numpy as np
import matplotlib as mplib
import seaborn

if __name__ == '__main__':
    filename = 'car_fuel_efficiency.csv'

    car_df = pd.read_csv(filename)
