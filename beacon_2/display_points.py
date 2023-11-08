import pandas as pd
import random
from matplotlib import pyplot as plt


def main():
    df = pd.read_csv('beacon_readings.csv')
    df['Position X'] = df.apply(lambda row : row['Position X'] + random.randint(-6, 6), axis = 1)
    df['Position Y'] = df.apply(lambda row : row['Position Y'] + random.randint(-6, 6), axis = 1)

    plt.scatter(df['Position X'], df['Position Y'])
    plt.xlabel('Positon X')
    plt.ylabel('Position Y')
    plt.show()

if __name__ == '__main__':
    main()
