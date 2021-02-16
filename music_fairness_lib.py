import pandas as pd
import scipy
import random
import numpy as np
from sklearn import metrics
import os.path

def get_df_events():
    # initialize data
    item_threshold = 1  # used to filter out user/artist pairs that have been
    # listened to less than the threshold number of times
    popular_artist_fraction = 0.2  # top cutoff for what we consider popular artists, in this case the top 20%

    user_events_file = os.path.join("/home/jimi/music_fairness/LFM-1b-surprise-and-implicit-analysis/ML_Base_Project/data/user_events.txt")
    low_user_file = "/home/jimi/music_fairness/LFM-1b-surprise-and-implicit-analysis/ML_Base_Project/data/low_main_users.txt"
    medium_user_file = "/home/jimi/music_fairness/LFM-1b-surprise-and-implicit-analysis/ML_Base_Project/data/medium_main_users.txt"
    high_user_file = "/home/jimi/music_fairness/LFM-1b-surprise-and-implicit-analysis/ML_Base_Project/data/high_main_users.txt"

    # read in user events file
    cols = ['user', 'artist', 'album', 'track', 'timestamp']
    df_events = pd.read_csv(user_events_file, sep='\t', names=cols)
    print('No. of user events: ' + str(len(df_events)))
    df_events.head()  # check it is all read in properly

    # create unique user-artist matrix
    df_events = df_events.groupby(['user', 'artist']).size().reset_index(name='count')
    print('No. user-artist pairs: ' + str(len(df_events)))
    # each row contains a unique user-artist pair, along with how many times the
    # user has listened to the artist
    return df_events


def scale_df_events(df_events_in):
    """
    Creates a new Dataframe converting df_events.count (num of interactions) to .count (listening count scaled from 1 - 1000)
    :param df_events_in: the pandas Dataframe created by loading in user_events_file
    :return: new Dataframe
    """
    scaled_df_events = pd.DataFrame()
    for user_id, group in df_events_in.groupby('user'):
        # print(group)
        min_listens = group['count'].min()
        max_listens = group['count'].max()
        std = (group['count'] - min_listens) / (max_listens - min_listens)
        scaled_listens = std * 999 + 1
        to_replace = group.copy()
        to_replace['count'] = scaled_listens
        # print(to_replace)
        scaled_df_events = scaled_df_events.append(to_replace)
    return scaled_df_events


def user_events_file_to_lil_matrix():
    # Artist to User matrix where artist_user_matrix[a, u] = num of times user u listened to artist a

    # 352805, 3000 (total artists, users)
    rows, cols = 352805, 3000
    artist_user_matrix = scipy.sparse.lil_matrix((rows, cols), dtype=int)

    # user	artist	album	track	timestamp

    user_dict = {}  # simplify user id to 1, 2, 3 ...
    artist_dict = {}

    # populate with user_events_file
    with open("/home/jimi/music_fairness/LFM-1b-surprise-and-implicit-analysis/ML_Base_Project/data/user_events.txt", 'r') as fp:
        line = fp.readline()
        loop_count = 0
        while line:
            # get data from line
            line = fp.readline()
            parts = line.split("\t")

            # end case
            try:
                user_id = int(parts[0])
                artist_id = int(parts[1])
            except ValueError:
                print("end of file " + line)
                break

            # use user_dict to shorten user_id
            if user_id not in user_dict:
                # this user_id has not bee seen
                user_dict[user_id] = len(user_dict)
            user_idx = user_dict[user_id]

            # use track_dict to shorten track_id
            if artist_id not in artist_dict:
                # this user_id has not bee seen
                artist_dict[artist_id] = len(artist_dict)
            artist_idx = artist_dict[artist_id]

            # increment count of user to track
            artist_user_matrix[artist_idx, user_idx] += 1

            # progress marker
            loop_count = loop_count + 1
            if loop_count % 10000000 == 0:
                print(str(loop_count) + "/ 28718087")  # / num of lines in file

    print(len(user_dict))
    print(len(artist_dict))

    # helpful dicts for converting artist and user count back to their ids
    user_count_to_id_dict = {v: k for k, v in user_dict.items()}
    artist_count_to_id_dict = {v: k for k, v in artist_dict.items()}

    return artist_user_matrix, user_dict, artist_dict, user_count_to_id_dict, artist_count_to_id_dict


## FUNCTIONS FROM from jmsteinw's blog post ##


def make_train(ratings, pct_test=0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings,
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix.

    parameters:

    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix.

    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the
    training set for later comparison to the test set, which contains all of the original ratings.

    returns:

    training_set - The altered version of the original data with a certain percentage of the user-item pairs
    that originally had interaction set back to zero.

    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order
    compares with the actual interactions.

    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy()  # Make a copy of the original set to be the test set.
    test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix
    training_set = ratings.copy()  # Make a copy of the original data we can alter as our training set.
    nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of user,item index into list
    random.seed(0)  # Set the random seed to zero for reproducibility
    num_samples = int(
        np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples)  # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples]  # Get the user row indices
    item_inds = [index[1] for index in samples]  # Get the item column indices
    training_set[user_inds, item_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds))  # Output the unique list of user rows that were altered


# Usage
# a_to_u_train, a_to_u_test, altered_users = make_train(csr_sparse_matrix, pct_test = 0.2)

def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics.

    parameters:

    - predictions: your prediction output

    - test: the actual target result you are comparing to

    returns:

    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    # shuffle list of predictions (shuffle function)
    # shuffling dissassociates link to artist - > roc of .5 (random)
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered.

    parameters:

    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model

    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one.

    altered_users - The indices of the users where at least one user/item pair was altered from make_train function

    test_set - The test set constucted earlier from make_train function

    returns:

    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''

    store_auc = []  # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = []  # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis=0)).reshape(-1)  # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users:  # Iterate through each user that had an item altered
        training_row = training_set[user, :].toarray().reshape(-1)  # Get the training set row
        zero_inds = np.where(training_row == 0)  # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user, :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training
        pop = pop_items[zero_inds]  # Get the item popularity for our chosen items
        curr_auc_score = auc_score(pred, actual)
        store_auc.append(curr_auc_score)  # Calculate AUC for the given user and store
        curr_pop_score = auc_score(pop, actual)
        popularity_auc.append(curr_pop_score)  # Calculate AUC using most popular and score
        # print(user, "\t", curr_auc_score , "\t", curr_pop_score)
    # End users iteration

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))
    # Return the mean AUC rounded to three decimal places for both test and popularity benchmark
