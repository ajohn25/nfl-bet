# Team Project EDA QB
# usage:  
#
# 
# History
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
# Name		Date		Description
# Aashish   11/28/2020 	Adapt Dr. Lindo's NFL QB graphs for Project EDA
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -

#import numpy as np 
import random as rd 
import pandas as pd 
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 15)

location = 'data/'
filename = 'nflqb2019.csv'

# This is a function to return general
# statistics about a data set
def f_calcStats(data):
    v_stats = data.describe()
    return v_stats

# This function will read the csv file
# and return a Pandas DataFrame
def f_getData():
    v_df = pd.read_csv(location+filename)
    v_df.set_index('Player', inplace=True)
    return v_df

# This function is going to return
# the data set sorted by a column
def f_sortData(data, col):
    data_sorted = data.sort_values(by=[col], ascending=False)
    return data_sorted

# - - - - - - - - - - - - - - - - - - - - - - - -
# This function is going to display
# a chart based on two dimensions
def plotDataPoints(data, col_x, col_y, col_z, c, qb, dimensions):
    x = []
    y = []

    x = data[col_x]
    y = data[col_y]

    plt.xlabel('QB ' + dimensions[0])
    plt.ylabel('QB ' + dimensions[1])
    plt.title('QB '+qb+' - '+dimensions[0]+ ' vs. ' + dimensions[1])

    plt.scatter(x, y, s=90, color=c, alpha=0.5, edgecolor='none')
    plt.savefig(location+qb+'_scatterplot_nfl.png')

def analyzeQB(qb, stats, dimensions):
    w = []
    x = []
    y = []
    z = []
    player = qb

    for k, v in stats.items():
        if 'Wk' in k:
            w.append(v)
        elif dimensions[0] in k:
            x.append(v)
        elif dimensions[1] in k:
            y.append(v)
        elif dimensions[2] in k:
            z.append(v)           

    df = pd.DataFrame(zip(w,x,y,z), columns=['w', 'x', 'y','z'])

    return df
# - - - - - - - - - - - - - - - - - - - - - - - -
def main():
    v_df_data = f_getData()
    v_df_data = v_df_data.fillna(0)

    # - print out the columns - 
    #for col in v_df_data.columns:
    #    print(col, end=' ')

    # loop through the index and analyze each
    # quarter backs 17 week season
    try:
        d_list =['Att', 'Yds', 'WinLoss']
        max_x_axis = 70
        max_y_axis = 500

        for index in v_df_data.index.to_list():
            v_df_qb = analyzeQB(index, v_df_data.loc[index], d_list)

            v_df_wins = v_df_qb['z'] == 'W'
            v_df_qb_wins = v_df_qb[v_df_wins]

            v_df_loss = v_df_qb['z'] == 'L'
            v_df_qb_loss = v_df_qb[v_df_loss]

            # set a fixed axis - easier to compare
            # with each player 3 dimentions 

            fig1 = plt.figure()

            plt.xlim(0, max_x_axis)
            plt.ylim(0, max_y_axis)

            plotDataPoints(v_df_qb_wins, 'x', 'y', 'z', 'green', index, d_list)
            plotDataPoints(v_df_qb_loss, 'x', 'y', 'z', 'red', index, d_list)

            plt.grid(True)
            plt.draw()
            plt.pause(20.00)
            plt.close(fig1)
        
    except KeyError:
        print('KeyError, Skip This Row')

    plt.show()

# -  -  main  -  -
print('\n - - -  Begin - - - ')

main()

print('\n - - -  End - - - \n')


