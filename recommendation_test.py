# User must create function called my_predict_rating(training_data, user_id, game_id)
# Where 'training_data' is a pandas dataframe with columns "user_id", "game_id", and "rating"
# 		'user_id' is the ID of the particular user which you are predicting a rating
#		'game_id' is the ID of the particular game for which you are predicting a rating
#		and the function returns the predicted rating which that particular user would give for that particular game




import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
from math import *

def my_predict_rating(training_data, user_id, game_id):
   # ADD YOUR CUSTOM CODE HERE
   return 7.25
   # rnd.randint(0, 9)

# Extracts some of the ratings from the provided Pandas DataFrame to create one
# set of training data, and one set of test data.
def extract_actual_ratings(data):
   training_data = data.copy()
   users = training_data['user_id'].drop_duplicates()

   my_list = list()
   test_size = 0
   train_size = 0
   for idx, user in users.iteritems():
   	user_ratings = training_data[training_data['user_id'] == user]
   	n_rated = user_ratings.size
   	test_list = list()
   	train_list = list()
   	for row in user_ratings.iterrows():
   		# This is a likelihood function that Matt has found works well
   		likelihood = sqrt(n_rated) / 1000
   		if n_rated > 1 and rnd.random() < likelihood:
   			my_list.append('test')
   			test_size += 1
   		else:
   			my_list.append('train')
   			train_size +=1

   my_list_df = pd.Series(my_list, name = 'train_or_test')
   
   training_mod = training_data
   training_mod['train_or_test'] = my_list_df

   test_data = training_mod[training_mod['train_or_test'] == 'test']
   train_data = training_mod[training_mod['train_or_test'] == 'train']

   test_data.drop('train_or_test', axis = 1, inplace=True)
   train_data.drop('train_or_test',axis = 1, inplace=True)

   print('Test Size:')
   print(test_size)

   print('Train Size:')
   print(train_size)

   return (train_data, test_data)

# data        Pandas DataFrame containing training data.
# inputs      Pandas DataFrame containing pairs of user IDs and game IDs. These
#               datapoints will be fed into the provided prediction function.
#               Should have two columns: user_id, game_id.
# predict_fn  Prediction function under test. Should be of the following form:
#               predict_fn(training_data, user_id, game_id) - returns double
#
# Returns a Pandas DataFrame containing the actual ratings that were predicted
# by the function under test. Will have three columns: user_id, game_id,
# actual_rating.
def predict_ratings(data, inputs, predict_fn):
   predict = lambda row: predict_fn(data, row['user_id'], row['game_id'])
   results = inputs.copy()
   results['rating'] = results.apply(predict, axis=1)
   return results

# Computes various error metrics after a test has been run: Root Mean Square,
# Median Absolute Deviation, and Mean Absolute Deviation
#
# expected  Pandas Series containing expected values.
# actuals   Pandas Series containing actual values.
#
# Returns (root_mean_sq, median_abs_dev, mean_abs_dev)
def compute_error_metrics(expected, actuals):
   delta = expected - actuals
   root_mean_sq = (((delta) ** 2).mean() ** 0.5)

   # TODO: properly calculate median and mean absolute deviations
   abs_delta = delta.abs()
   median_abs_dev = abs_delta.median()
   mean_abs_dev = abs_delta.mean()

   return (root_mean_sq, median_abs_dev, mean_abs_dev)

# data        Original data set.
# predict_fn  Prediction function under test. Should be of the following form:
#               predict_fn(training_data, user_id, game_id) - returns double
def test_recommendation_engine(data, predict_fn):
   # Select a certain number of ratings from the original dataset, and move them
   # into a dataset of "expected" ratings
   (training_data, actuals) = extract_actual_ratings(data)

   # The user IDs and game IDs to be passed into the model.
   inputs = actuals.copy()[['user_id', 'game_id']]

   # Ask the model to predict a rating for each input pair.
   expected = predict_ratings(training_data, inputs, my_predict_rating)

   # Compute and display error metrics
   (root_mean_sq, median_abs_dev, mean_abs_dev) = compute_error_metrics(
      expected['rating'], actuals['rating'])

   print('Root Mean Squared Error:   %.6f' % root_mean_sq)
   print('Median Absolute Deviation: %.6f' % median_abs_dev)
   print('Mean Absolute Deviation:   %.6f' % mean_abs_dev)

   # Plot scatterplot of test results combined with metrics
   scatter_data = pd.DataFrame()
   scatter_data['expected'] = expected['rating']
   scatter_data['actual'] = actuals['rating']

   scatter_data.plot(x='expected', y='actual', kind='scatter', alpha=0.9)
   plt.show()

def main():
   data = pd.read_csv('boardgame-elite-users.csv')
   data.rename(columns={'Compiled from boardgamegeek.com by Matt Borthwick': 'user_id', 'gameID':'game_id'}, inplace=True)
   test_recommendation_engine(data, my_predict_rating)

if __name__ == '__main__':
   main()
