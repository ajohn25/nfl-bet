# Team Project Data Acquisition
# usage:  
#
# 
# History
# -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
# Name		Date		Description
# Aashish     	11/16/2020 	Initial framework to run acquisition
    
from team_utility import nflDataAcquisition

def acquire(teamfilename, qbfilename, yearfilename, outputfilename): # Correlate and clean datasets
    print("Running correlate, results in " + outputfilename + ".csv")
    nflData = nflDataAcquisition("data/" + teamfilename + ".csv")
    corr_df = nflData.correlate("data/" + teamfilename + ".csv", "data/" + qbfilename + ".csv", "data/" + yearfilename + ".csv")
    clean_df = nflData.cleanDf(corr_df)
    nflData.writeout(clean_df, "data/" + outputfilename + ".csv")

acquire("teams", "qbstats", "2019", "nfl_acquisition")