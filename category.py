import reader
import nltk
import gensim
import pandas as pd
from nltk.data import find
from nltk.stem import PorterStemmer
import collections

def category(tag):
    if tag not in ['love', 'war', 'happiness', 'loneliness', 'money']:
        print("Use tag from list: love war happiness loneliness money")
        exit(1)

    df_words = reader.read_mxm()
    df_plcount = reader.read_train_triplets()
    df_unique = reader.read_unique_tracks()

    songid_raiting = df_plcount[['play_count', "song_id"]].groupby("song_id").sum()
    unique_cleaned = df_unique.drop_duplicates(['song_id'])
    song_raiting_named = pd.merge(songid_raiting, unique_cleaned, on='song_id', how = 'inner')

    #word2vect
    ps = PorterStemmer()

    #find synonyms
    word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    top = model.most_similar(positive=tag, topn = 10)
    top_stem = set(map(lambda x: ps.stem(x[0]), top))
    top_stem = list(set(top_stem).intersection(set(df_words.columns)))

    #remove duplications in columns due to stemmings column names
    duplicated = [item for item, count in collections.Counter(df_words[top_stem].columns).items() if count > 1]
    for d in duplicated:
        df_words['tmp'] = df_words[d].sum(axis = 1)
        df_words.drop(d , axis = 1, inplace = True)
        df_words.rename(columns = {'tmp' : d}, inplace=True)

    threshold = 15
    #select collections
    s = df_words[top_stem].sum(axis = 1)
    selected_songs = df_words[(s > threshold)]['track_id']
    tag_col = pd.merge(song_raiting_named, selected_songs, how = 'inner', on = 'track_id').sort_values('play_count', ascending=False)[:50]
    tag_col.index = range(len(tag_col))
    print(tag_col[['artist', 'title']])
