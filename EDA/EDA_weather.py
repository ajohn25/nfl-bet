
import requests
import json
#import nflgame
import pandas as pd
import matplotlib.pyplot as plt

def jprint(obj): # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

def plotDataPoints(xdata, ydata, xlbl, ylbl, title, mstyle): # Scatter Plot
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.plot (xdata, ydata, mstyle)


def main():
    df = pd.read_csv('data/nflbet.csv')
    print (df)

    home = df['score_home']
    away = df['score_away']
    odds = df['over_under_line']
    temp = df['weather_temperature']
    humid = df['weather_humidity']
    wind = df['weather_wind_mph']

    #print (home)

    # dome = df['weather_detail'] == 'DOME'

    diff = home + away - odds

    plotDataPoints(temp, diff, "Temp (°F)", "Difference between Final Score and Over/Under Odds", "Stadium Temperature vs. Game Outcome", "g.")
    plt.show()
    plotDataPoints(humid, diff, "Humidity (%)", "Difference between Final Score and Over/Under Odds", "Stadium Humidity vs. Game Outcome", "g.")
    plt.show()
    plotDataPoints(wind, diff, "Wind Speed (mph)", "Difference between Final Score and Over/Under Odds", "Wind Speed vs. Game Outcome", "g.")
    plt.show()

main()
