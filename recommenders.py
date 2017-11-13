import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('boardgame-elite-users.csv')
data.rename(columns={'Compiled from boardgamegeek.com by Matt Borthwick': 'userID'}, inplace=True)

titles = pd.read_csv('boardgame-titles.csv')
titles.rename(columns={'boardgamegeek.com game ID': 'gameID'}, inplace=True)

data = data.merge(titles,on='gameID')


# Prints a histogram of the ratings of the given user
def user_histogram(user):

    user_num = data.loc[data['userID'] == user]

    fig, axes = plt.subplots(nrows=1, ncols=1)

    user_num['rating'].plot.hist(bins=10)
    axes.set_title('Ratings for User %d'%user)
    axes.set_xlabel('Rating')
    axes.set_ylabel('Count')

    plt.show()


# Prints a histogram of the ratings of the given game
def game_histogram(game):

    game_num = data.loc[data['gameID'] == game]

    fig, axes = plt.subplots(nrows=1, ncols=1)

    game_num['rating'].plot.hist(bins=10)
    axes.set_title('Ratings for Game %d'%game)
    axes.set_xlabel('Rating')
    axes.set_ylabel('Count')

    plt.show()

# Prints out a useful overall look at the data, including number of
# users per game and number of games per user
def overall_look():
    gameIDs = data.iloc[:,1]

    game_hist_data = gameIDs.value_counts()

    print('Number of Games:', game_hist_data.size)


    userIDs = data.iloc[:,0]

    user_hist_data = userIDs.value_counts()

    print('Number of Users:', user_hist_data.size)



    fig, axes = plt.subplots(nrows=2, ncols=1)

    title = 'Number of Users per Game'
    game_hist_data.plot(kind='bar',ax=axes[0],label=title)
    axes[0].set_title(title)
    axes[0].set_xlabel('Game ID')
    axes[0].set_ylabel('Number of Users')

    title = 'Number of Games per User'
    user_hist_data.plot(kind='bar',ax=axes[1],label=title)
    axes[1].set_title(title)
    axes[1].set_xlabel('User ID (subset of 1%d users)')
    axes[1].set_ylabel('Number of Games')

    fig.subplots_adjust(hspace=0.8)

    plt.show()

# Lists the games rating by a given user
def list_games(user):
    userdata = data[data['userID'] == user]
    games = userdata['gameID']
    sorted_games = games.sort_values()
    sorted_games.reset_index(drop=True, inplace=True)
    return sorted_games

# Lists the users who rated a given game
def list_users(game):
    userdata = data[data['gameID'] == game]
    users = userdata['userID']
    sorted_users = users.sort_values()
    sorted_users.reset_index(drop=True, inplace=True)
    return sorted_users

# Gets the rating from the specified user of the specified game
# This returns NaN if a rating for the requested user and game
# combo does not exist
def get_rating(user, game):
    only_game = data['gameID'] == game
    only_user = data['userID'] == user
    only_game_data = data[only_game]
    if user in only_game_data.userID.values:
        row = data[only_game & only_user]
        rating = row['rating'].values[0]
    else:
        rating = np.nan
    return rating


# Pivots data to make columns the users, for creating correlation matrix
def pivot_user(datain):
    matrix = datain.pivot(index='title', columns='userID',values='rating')
    return matrix


# Returns a pandas series with all of the
# specified type (userID, gameID, title)
# where "my_type" is a string
def list_all_type(data, my_type):
    IDs = data[my_type]
    all_types = IDs.drop_duplicates()
    all_types.reset_index(drop=True, inplace=True)
    return all_types





threshold = 8

game = 66690
thisgame = data['gameID'] == game
likegame = data['rating'] > threshold
loves_game = data[thisgame & likegame]
##print(loves_game.head())

user = 3256
thisuser = data['userID'] == user
likeuser = data['rating'] > threshold
user_loves = data[thisuser & likeuser]
sorted_user_loves = user_loves.sort_values('rating',ascending=False)
##print(sorted_user_loves.head(n=20))


matrix = pivot_user(data)

liked_games = data.loc[data['rating'] > threshold]
matrix_liked_games = pivot_user(liked_games)
liked_correlation = matrix_liked_games.corr()

correlation = matrix.corr()


unpacked_correlation = correlation.unstack()

array = [[]]
my_range = range(0, len(unpacked_correlation))
for i in my_range:
    if unpacked_correlation.index[i][0] != unpacked_correlation.index[i][1]:
        array.append([unpacked_correlation.index[i][0], 
            unpacked_correlation.index[i][1],
            unpacked_correlation.values[i]])

corr_df = pd.DataFrame(array)
corr_df.columns = ['user1','user2','corr_coef']


most_corr = corr_df[corr_df['corr_coef'] > 0.8]

all_users = list_all_type(data, 'userID')
haters = all_users[~all_users.isin(most_corr['user1'])]
print(haters)

user_histogram(26532)

