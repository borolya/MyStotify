import pandas as pd
import zipfile
import numpy as np
import os
from nltk.stem import PorterStemmer

def read_unique_tracks():
    f = "data/cache/unique.pkl"
    if os.path.exists(f):
        df_unique = pd.read_pickle(f)
    else:
        df_unique = pd.read_csv("data/p02_unique_tracks.txt", sep = "<SEP>", header = None, engine='python')
        df_unique.columns = ["track_id", "song_id", "artist", "title"]
        df_unique.to_pickle("data/cache/unique.pkl")
    return df_unique

def read_train_triplets():
    f = "data/cache/play_count.pkl"
    if os.path.exists(f):
        df_plcount = pd.read_pickle("data/cache/play_count.pkl")
    else:
        with zipfile.ZipFile("data/p02_train_triplets.txt.zip", 'r') as zip_ref:
            zip_ref.extractall("data/cache/")
            df_plcount = pd.read_csv("data/cache/train_triplets.txt", sep = "\t", header = None, engine='python')
            df_plcount.columns = ["user_id", "song_id", "play_count"]
        df_plcount.to_pickle("data/cache/play_count.pkl")
    return df_plcount

def read_tagtraum():
    f = "data/cache/tagtraum.pkl"
    if os.path.exists(f):
        df_gnr = pd.read_pickle("data/cache/tagtraum.pkl")
    else:
        with open("data/p02_msd_tagtraum_cd2.cls", "r") as tagtraum_fd:
            content = list()
            lines = tagtraum_fd.readlines()
            useful_text = list(filter(lambda row: row[0] != '#', lines))
            for i, line in enumerate(useful_text):
                words = line[:-1].split("\t")
                if len(words) == 2:
                    words.append("-")
                content.append(words)

        df_gnr = pd.DataFrame(content, columns = ["track_id", "major_gnr", "minor_gnr"])
        df_gnr.to_pickle("data/cache/tagtraum.pkl")
    return df_gnr

def read_mxm():
    f = "data/cache/mxm.pkl"
    if os.path.exists(f):
        df_words = pd.read_pickle("data/cache/mxm.pkl")
    else:
        with zipfile.ZipFile('data/p02_mxm_dataset_train.txt.zip', 'r') as zip_ref:
            zip_ref.extractall("data/cache/")
            
            track_id_list = list()
            mxm_track_id_list = list()
            top_words_list = list()
            song_words_list = list()
            with open("data/cache/mxm_dataset_train.txt", "r") as mxm_fd:
                lines = mxm_fd.readlines()
                useful_text = list(filter(lambda row: row[0] != '#', lines))
                words_line = "".join(map(lambda x: x[1:-1], filter(lambda row: row[0] == '%', useful_text)))
                top_words_list = words_line.split(',')
                
                #apply stemming for words
                ps = PorterStemmer()
                top_words_list = list(map(lambda x: ps.stem(x), top_words_list))

                songs = list(filter(lambda row: row[0]!='%', useful_text))
                for i, song in enumerate(songs):
                    clean_song = song[:-1]
                    comma1 = clean_song.find(',')
                    comma2 = clean_song.find(',', comma1 + 1)       
                    res = '{' + clean_song[comma2+1:] + '}'
                    raw = eval(res)
                    song_words_list.append(raw)
                    track_id_list.append(clean_song[:comma1])
                    mxm_track_id_list.append(clean_song[comma1+1: comma2])
                    

            x = [] # list that will hold data for Data Frame
            num_words = len(top_words_list)
            for d in song_words_list:
                # initializing numpy vector
                temp = np.zeros(num_words, dtype=np.float64)
                for key, val in d.items():
                    key -= 1
                    temp[key] = val
                x.append(temp) 
            df_words = pd.DataFrame(np.array(x), columns = top_words_list)
            df_words['track_id'] = track_id_list
            df_words['mxm_track_id'] = mxm_track_id_list

            #clean dupl
        df_words.to_pickle("data/cache/mxm.pkl")
    return df_words

