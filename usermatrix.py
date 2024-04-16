from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from scipy.sparse import spdiags
import pandas as pd
import numpy as np

#This method converts a list of (user, item, rating, time) to a sparse matrix
def user_feachers_matrix(user_num, song_num, play_count, len_uusers, len_usongs):
    return csr_matrix((play_count, (user_num, song_num)),
                             shape=(len_uusers, len_usongs))

def user_matrix(df_plcount, uniq_songs_numerated, uniq_users_numerated):

    len_usongs = len(uniq_songs_numerated)
    len_uusers = len(uniq_users_numerated)
    # normalization -> all - mean
    avarage_byuser = df_plcount[['user_num', 'play_count']].groupby('user_num').mean()
    avarage_byuser.reset_index('user_num', inplace = True)
    tmp = pd.merge(df_plcount, avarage_byuser, how = 'left', on='user_num')
    tmp['play_count_x'] = tmp['play_count_x'] - tmp['play_count_y']
    tmp.rename(columns = {'play_count_x':'play_count'}, inplace = True)

    # Create sparse matrices user_songs
    user_song_mtx = user_feachers_matrix(np.array(df_plcount['user_num']), np.array(df_plcount['song_num']), np.array(df_plcount['play_count']), len_uusers, len_usongs)

    #normolize matrix for easy cos simularity calculate
    user_song_mtx_norm = normalize(user_song_mtx.tocsr()).tocsr()
    return user_song_mtx_norm