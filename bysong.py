import reader
import usermatrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import spdiags
from scipy.sparse import vstack

def top_songs(song_num, item_user, uniq_songs_numerated):
    song_vector = item_user[song_num, :]
    cosine_item = song_vector.dot(item_user.T)
    cosine_item[:, song_num] = 0
    row, col = cosine_item.nonzero()
    data_cos = cosine_item.toarray()[0][col]
    similarity_dict = dict(zip(col, data_cos))
    top_items = dict(sorted(similarity_dict.items(), key = lambda x: x[1], reverse = True)[:10])
    return top_items.keys()


def bysong(songid):
    df_plcount = reader.read_train_triplets()
    df_unique = reader.read_unique_tracks()

    if len(df_plcount[df_plcount.song_id == songid]) < 1:
        print("Incorrect user_id")
        exit(1)

    #numerate songs from 0 to len(song) for matrix creatioin
    uniq_songs = df_plcount['song_id'].value_counts().index
    uniq_songs_numerated = pd.Series(index = uniq_songs, data = range(len(uniq_songs)))

    #numerate users from 0 to len(users) for matrix creatioin
    uniq_users = df_plcount['user_id'].value_counts().index
    uniq_users_numerated = pd.Series(index = uniq_users, data = range(len(uniq_users)))

    del uniq_songs
    del uniq_users
    df_plcount['song_num'] = df_plcount['song_id'].map(uniq_songs_numerated)
    df_plcount['user_num'] = df_plcount['user_id'].map(uniq_users_numerated)

    song_num = uniq_songs_numerated.loc[songid]
    item_user = usermatrix.user_matrix(df_plcount, uniq_songs_numerated, uniq_users_numerated).T
    similar_songs = top_songs(song_num, item_user, uniq_songs_numerated)
    df_unique.drop_duplicates('song_id', inplace=True)
    print("SONG:")
    print(df_unique[df_unique.song_id == songid][['artist', 'title']])
    
    print("\nSIMILAR:")
    songs_id = uniq_songs_numerated.loc[uniq_songs_numerated.isin(similar_songs)]
    pr = df_unique[df_unique.song_id.isin(songs_id.index)][['artist', 'title']]
    pr.index = range(len(pr))
    print(pr)