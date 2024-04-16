import pandas as pd
import numpy as np
import reader
import usermatrix

def track_predict(user_num, sparse_matrix, similar_users_count): 
    #cosin similarity
    user_vector = sparse_matrix[user_num, :]
    cosin = user_vector.dot(sparse_matrix.T)

    #extract N the most similar

    row, col = cosin.nonzero()
    data_cos = cosin.toarray()[0][col]
    similarity_dict = dict(zip(col, data_cos))
    top_sim_users = dict(sorted(similarity_dict.items(), key = lambda x: x[1], reverse = True)[1:similar_users_count + 1])
    
    simils_items = sparse_matrix[list(top_sim_users.keys()), :]
    
    #multiple user_similarity coeff to every matrix row
    # selected_cos = cosin.tocsr()[0, list(top_sim_users.keys())]
    # simils_items = simils_items.astype(np.float32)
    # for i in range(similar_users_count):
    #    simils_items[i, :] = simils_items[i, :] * selected_cos[0, i]
    
    #summary by useres to see top trackes
    final_predicted_raitings = simils_items.sum(axis = 0)
    
    #drop columns which user listend in train
    user, users_listend = sparse_matrix[user_num, :].nonzero()

    row, col = final_predicted_raitings.nonzero()
    new_col = list(set(col) - set(users_listend))
    #extrack song numbers
    data_raiting = np.array(final_predicted_raitings[0, new_col])[0].tolist()
    dict_raiting = dict(zip(list(new_col), data_raiting))
    sorted_songs = dict(sorted(dict_raiting.items(), key = lambda x: x[1], reverse = True)[:10])
    return sorted_songs.keys()



def byuser(userid):
    df_plcount = reader.read_train_triplets()
    df_unique = reader.read_unique_tracks()

    if len(df_plcount[df_plcount.user_id == userid]) < 1:
        print("Incorrect user_id")
        exit(1)

    #numerate songs from 0 to len(song) for matrix creatioin
    uniq_songs = df_plcount['song_id'].value_counts().index
    uniq_songs_numerated = pd.Series(index = uniq_songs, data = range(len(uniq_songs)))

    #numerate users from 0 to len(users) for matrix creatioin
    uniq_users = df_plcount['user_id'].value_counts().index
    uniq_users_numerated = pd.Series(index = uniq_users, data = range(len(uniq_users)))

    df_plcount['song_num'] = df_plcount['song_id'].map(uniq_songs_numerated)
    df_plcount['user_num'] = df_plcount['user_id'].map(uniq_users_numerated)
    user_song_mtx_norm = usermatrix.user_matrix(df_plcount, uniq_songs_numerated, uniq_users_numerated)

    user_num = uniq_users_numerated.loc[userid]
    songs = track_predict(user_num, user_song_mtx_norm, 5)
    songs_ids = uniq_songs_numerated.loc[uniq_songs_numerated.isin(songs)].index.tolist()
    df_unique.drop_duplicates('song_id', inplace=True)
    songs = df_unique[df_unique.song_id.isin(songs_ids)][['artist', 'title']]
    songs.index = range(len(songs))
    print(songs)





