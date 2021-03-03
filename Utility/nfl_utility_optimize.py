
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame, read_csv
from numpy import vstack
from datetime import datetime
import jinja2
import pandas as pd
import numpy as np

class dataScienceOptimize:
	def __init__(self, f):
		self.X = ''
		self.y = ''

		self.X_train = ''
		self.y_train = ''

		self.X_test = ''
		self.y_test = ''

		self.__df = self.dataAcquisition(f)

	def dataAcquisition(self, f):
		file = f 
		self.__df = read_csv(file)
		self.datahead()
		return self.__df

	def datainfo(self):
		return self.__df.info()

	def datahead(self):
		return self.__df.head()

	def featureSelection(self, feature1, feature2, cls, train_pct):
		array1 = self.__df[feature1].to_numpy()
		array2 = self.__df[feature2].to_numpy()
		self.X = vstack((array1,array2)).T
		self.y = self.__df[cls]
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0, train_size= train_pct)

	def readcsv (self, filename): # Convert csv to dataframe
		csv_df = pd.read_csv(filename)
		return csv_df

	def writeout (self, df, filename): # Convert dataframe to csv, excluding row labels
		df.to_csv(filename, index = False)

class nflDataAcquisition (dataScienceOptimize):
	def splitWeeks (self, stats, team): # Split QB stats by week
		# Arrays to hold stats
		Team = [team] * 16
		Wk = []
		Comp = []
		Att = []
		Pct = []
		Yds = []
		TD = []
		Int = []
		# Reference arrays
		stats_arrays = [Team, Wk, Comp, Att, Pct, Yds, TD, Int]
		stats_columns = ["Team", "Wk", "Comp", "Att", "Pct", "Yds", "TD", "Int"]

		for k, v in stats.items(): # Loop through df
			for column in stats_columns:
				if column in k: # Check if k begins with default column headings
					i = stats_columns.index(column)
					stats_arrays[i].append(v)          

		df = pd.DataFrame(zip(Team, Wk, Comp, Att, Pct, Yds, TD, Int), columns= stats_columns)
		return df

	def cleanDf (self, df):
		clean_df = df.dropna()
		return clean_df

	def correlate(self, teamfilename, qbfilename, yearfilename):
		# Lookup df for team names/IDs
		teams_df = self.readcsv(teamfilename)

		# QB Stats
		qb_df = self.readcsv(qbfilename)
		qb_df.set_index('Team', inplace = True)
		qb_df.drop(columns = ['Player'], inplace = True)

		# Betting Stats
		bet_df = self.readcsv(yearfilename)
		bet_df = bet_df.iloc[:, 2:11] # Remove unnecessary data from df

		# Lookup home team IDs
		bet_df = bet_df.merge(teams_df, left_on = ['team_home'], right_on = ['team_name'])
		bet_df = bet_df.drop(columns = ['team_name'])
		bet_df = bet_df.rename(columns = {'team_id' : 'team_id_home'})

		# Lookup away team IDs
		bet_df = bet_df.merge(teams_df, left_on = ['team_away'], right_on = ['team_name'],)
		bet_df = bet_df.drop(columns = ['team_name'])
		bet_df = bet_df.rename(columns = {'team_id' : 'team_id_away'})

		# Calculate whether teams went over/under
		odds = bet_df['over_under_line']
		home = bet_df['score_home']
		away = bet_df['score_away']
		diff = home + away - odds
		bet_df['Odds Difference'] = diff

		# Label whether game went Over/Under
		bet_df.loc[(bet_df['Odds Difference'] > 0),'Over/Under']='Over'
		bet_df.loc[(bet_df['Odds Difference'] < 0),'Over/Under']='Under'
		bet_df.drop(columns = ['Odds Difference'], inplace = True)

		# Format bet_df
		bet_df = bet_df.iloc[:, np.r_[0, 9:12]] # Df now has week, home, away, over/under
		bet_df = bet_df[bet_df['schedule_week'].apply(lambda x: x.isnumeric())] # Remove playoff rows
		bet_df = bet_df.astype({"schedule_week": int}) # Cast to int for merging later

		# Split QB Data by weeks
		total_df = pd.DataFrame()
		for index in qb_df.index.to_list():
			team_df = self.splitWeeks(qb_df.loc[index], index) # Split for each team
			total_df = pd.concat([total_df, team_df]) # Add to growing dataframe
		
		# Merge for home teams
		total_df = pd.merge(left = total_df, right = bet_df, how = 'left', left_on = ['Team', 'Wk'], right_on = ['team_id_home','schedule_week'])
		total_df.drop(total_df.columns[[8, 9, 10]], axis = 1, inplace = True) 

		# Merge for away teams
		total_df = pd.merge(left = total_df, right = bet_df, how = 'left', left_on = ['Team', 'Wk'], right_on = ['team_id_away','schedule_week'])
		total_df.loc[total_df["Over/Under_x"].isnull(),'Over/Under_x'] = total_df['Over/Under_y']

		# Format total_df
		total_df = total_df.iloc[:, 2:9]
		total_df = total_df.rename(columns = {'Over/Under_x' : 'Class'})
		return total_df

import configparser

def getConfigsJobs():
	cf = './config/teamproject.conf'
	config = configparser.ConfigParser()
	config.read(cf)
	jobs = config['execution_plan'].getint('num_jobs')
	precision = config['execution_plan'].getint('PRECISION')
	return jobs, precision

def getConfigs(j):
	cf = './config/teamproject.conf'
	config = configparser.ConfigParser()
	config.read(cf)
	job = j
	fn = config[job]['filename']
	f1 = config[job]['feature_col1']
	f2 = config[job]['feature_col2']
	c  = config[job]['classifi_col']
	k  = config[job].getint('k')
	return fn, f1, f2, c, k

class getProjectReport: 
	def getReportHeading(self,jobs,filename,k,feature_col1,feature_col2,nbr_acc,nbz_acc,svm_acc,kmm_acc,nbr_f1,nbz_f1,svm_f1,kmm_f1):
		open("./reports/" + str(datetime.now()) + "report.txt","w")
		templateLoader = jinja2.FileSystemLoader(searchpath="./")
		templateEnv = jinja2.Environment(loader=templateLoader)
		TEMPLATE_FILE = "./reports/report_template.txt"

		template = templateEnv.get_template(TEMPLATE_FILE)
		outputText = template.render(job_name=filename,time=datetime.now(),k=k,f1=feature_col1,f2=feature_col2,KNN_acc=nbr_acc,NBz_acc=nbz_acc,SVM_acc=svm_acc,KMeans_acc=kmm_acc,KNN_f1=nbr_f1,NBZ_f1=nbz_f1,SVM_f1=svm_f1,KMeans_f1=kmm_f1)
		print(outputText)

	def getEDA(self): 
		return 0
	def getAnalysis(self):
		return 0
