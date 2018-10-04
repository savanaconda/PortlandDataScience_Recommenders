# This file processes data about board game ratings from boardgamegeek.com and recommends board games to
# users based on their preferences. Implements user-based collaborative filtering to group similar users
# based on their ratings and swap recommendations for those users.
#
# Limitations to this algorithm are that it does not handle users which have very few ratings or do not group
# well with other users, and it also can only recommend certain games to users rather than giving a probability that
# a user will like a given game

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Reads data from file and renames header columns
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
def overall_look(plot='on'):
    gameIDs = data.iloc[:,1]

    game_hist_data = gameIDs.value_counts()

    # print('Number of Games:', game_hist_data.size)


    userIDs = data.iloc[:,0]

    user_hist_data = userIDs.value_counts()

    # print('Number of Users:', user_hist_data.size)


    if plot == 'on':
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

    return game_hist_data.size, user_hist_data.size

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

# threshold rating to consider that a user likes the game
threshold = 8

# Returns data from users which love the given game (rate above given threshold)
def users_love_game(game, threshold):
    thisgame = data['gameID'] == game
    likegame = data['rating'] > threshold
    loves_game = data[thisgame & likegame]
    loves_game = loves_game.sort_values('rating',ascending=False)
    loves_game = loves_game.drop(['gameID', 'title'], axis = 1)
    loves_game = loves_game.reset_index(drop=True)
    return loves_game

# Returns data on games which given user loves (rate above given threshold)
def games_user_loves(user, threshold):
    thisuser = data['userID'] == user
    likeuser = data['rating'] > threshold
    user_loves = data[thisuser & likeuser]
    user_loves = user_loves.sort_values('rating',ascending=False)
    user_loves = user_loves.drop(['userID'], axis = 1)
    user_loves = user_loves.reset_index(drop=True)
    return user_loves


# Returns data from users which hate the given game (rate below given threshold)
def users_hate_game(game, threshold):
    thisgame = data['gameID'] == game
    hategame = data['rating'] < threshold
    hates_game = data[thisgame & likegame]
    hates_game = hates_game.sort_values('rating',ascending=True)
    return hates_game

# Returns data on games which given user hates (rate below given threshold)
def games_user_hates(user, threshold):
    thisuser = data['userID'] == user
    hateuser = data['rating'] < threshold
    user_hates = data[thisuser & likeuser]
    user_hates = user_hates.sort_values('rating',ascending=True)
    return user_hates


# Gets correlation values between users, in table with user 1, user 2
# and correlation coefficient between them
def correlate_users(rating_threshold, plot=False):
    # rearranges data so that columns are users
    # limits the data to only the games that users like

    liked_games = data.loc[data['rating'] > rating_threshold]
    matrix = pivot_user(liked_games)
    correlation = matrix.corr()

    # test = matrix['272']
    # print(matrix[matrix.userID == 272])

    # if plot == True:
    #     matrix.plot(x=matrix['272'], y=matrix['388'], style = 'o')
    #     plt.show()
    # print(matrix.head())

    # Rearrange data into tables with correlation coefficients between two users
    unpacked_correlation = correlation.unstack()
    array = [[]]
    my_range = range(0, len(unpacked_correlation))
    for i in my_range:
        if unpacked_correlation.index[i][0] != unpacked_correlation.index[i][1]:
            array.append([unpacked_correlation.index[i][0], 
                unpacked_correlation.index[i][1],
                unpacked_correlation.values[i]])
    corr_df = pd.DataFrame(array)
    corr_df.columns = ['user1','otheruser','corr_coef']
    # Calculate the number of users in data
    all_unique_users = corr_df['user1'].value_counts()
    user_count = len(all_unique_users)

    return corr_df


# Gets the most correlated users, above a certain threshold
def most_correlated(corr_df, corrcoef_threshold):
    # Get only user correlations that are highly correlated
    # (correlation coefficient greater than threshold)
    most_corr = corr_df[corr_df['corr_coef'] > corrcoef_threshold]
    # Calculate the number of users in data
    unique_users = most_corr['user1'].value_counts()
    user_count = len(unique_users)

    # Removes users that are perfectly correlated (they will not have
    # different games to exchange)
    most_corr_not1 = most_corr[most_corr['corr_coef'] < 1.0]
    # Calculate the number of users in data
    unique_users = most_corr['user1'].value_counts()
    user_count = len(unique_users)

    return most_corr_not1

# Given the correlation table and a user, finds n_users number of users most
# correlated to the given user and their correlation coefficients
def most_correlated_users(corr_df, user, n_users):
    user_values = corr_df[corr_df['user1'] == user]
    # removes ones with correlation of 1.0 because that means
    # they are the same and there are no games to exchange
    user_values = user_values[user_values['corr_coef'] < 1.0]
    
    max_corr = user_values['corr_coef'].max()
    row_of_max = user_values[user_values['corr_coef'] == max_corr]

    sorted_user_values = user_values.sort_values('corr_coef',ascending=False)

    output_mostcorr = sorted_user_values[['otheruser','corr_coef']][0:n_users]
    output_mostcorr.reset_index(drop=True, inplace=True)

    return output_mostcorr


corr_values = correlate_users(7, plot=True)
otherusers = most_correlated_users(corr_values, 272, 5)
# print(otherusers)


# Next step is to identify the games that the otherusers like which the original user has not rated
# Take the average of the top 5 users's ratings to get a value

threshold = 8
for idx, user in enumerate(otherusers):
    # GET THIS WORKING FOR THE CORRECT LOOPING (ROW NOT COLUMN)
    useridx_loves = games_user_loves(otherusers['otheruser'][idx],threshold)
    # print(idx)
    if idx == 0:
        topgames = useridx_loves
    else:
        topgames = topgames.merge(useridx_loves, on=['gameID','title'], how='outer')
        # topgames.rename(index=str, columns={"rating_x":"rating_user2","rating_y":"rating_user3"},inplace=True)
# print(topgames)
# need to figure out how to average and skip NaNs





# Training and Testing -> takes a percentage of data as the
# train data set to establish the algorithm and see if it
# works on the test data set
userdata = pivot_user(data)
rand = np.random.uniform(size = userdata.shape[0])
userdata['rand'] = pd.Series(rand, index = userdata.index)

train = userdata[userdata['rand'] < 0.6]
test = userdata[userdata['rand'] >= 0.6]









# all_users = list_all_type(data, 'userID')
# haters = all_users[~all_users.isin(most_corr['user1'])]
# # print(haters)

