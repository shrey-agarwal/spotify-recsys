import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import operator
import re
import pickle
from threading import Thread
from collections import defaultdict
import emoji
import copy


class MyThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self):
        Thread.join(self)
        return self._return
        

        
stopwords = ['is','a','at','an','the','of','it','and','as','that','all', 'my']
choices = pickle.load(open('annoy-dim-256-tree-100-mincount-10/playlist_names.pkl', 'rb'))
#choices = pickle.load(open('data/playlist_names.pkl', 'rb'))

def normalize_name(name):
    """
    Normalize playlist names by removing some stopwords.
    
    Params
    ------
    name : str
        Playlist name to normalize
    """
    global stopwords
    
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    name = name.lower()
    querywords = name.split()
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    resultwords1  = [word for word in querywords if word.lower() not in ['music', 'songs', 'playlist', 'play', 'song', 'list', 'hits', 'hit']]
    resultwords = resultwords1 if resultwords1 else resultwords
    name = ' '.join(resultwords)
    return name



dict_choices = defaultdict( list )
dict_choices1 = defaultdict( list )
for i,w in enumerate(choices):
    dict_choices[w].append(i)

for w in dict_choices:    
    dict_choices1[normalize_name(emoji.replaceEmoji(w))].extend(dict_choices[w])
    
dict_choices = dict_choices1




def get_topN_playlists_helper(playlists, pid_list, topN=10):
    N = topN
    end = len(choices)
    top_songs = {}
    keys = dict_choices.keys()
    for i, playlist in enumerate(playlists):
        #print('Extracting playlist: {}'.format(pid_list[i]))
        d = {}
        playlist = normalize_name(emoji.replaceEmoji(playlist))
        topN1 = process.extract(playlist, keys, limit=2*N, scorer=fuzz.token_set_ratio)
        topN2 = process.extract(playlist, keys, limit=2*N, scorer=fuzz.ratio)
        for x in topN1:
            if x[0] in d:
                d[x[0]] = (d[x[0]][0] + 0.4*x[1], d[x[0]][1] + 1)
            else:
                d[x[0]] = (0.4*x[1], 1)
        
        for x in topN2:
            if x[0] in d:
                d[x[0]] = (d[x[0]][0] + 0.6*x[1], d[x[0]][1] + 1)
            else:
                d[x[0]] = (0.6*x[1], 1)
        topN = sorted(d.items(), key=lambda x: x[1][0], reverse=True)[:N]
        topN = [(x[0], x[1][0]) for x in topN]
        temp_list = []
        for name,score in topN:
            temp_list.extend(dict_choices[name][:10])
            if len(temp_list) < 10:
                continue
            else:
                break
        top_songs[pid_list[i]] = copy.deepcopy(temp_list[:10])
        
    return top_songs
    
def get_topN_playlists(playlists, topN=10, no_of_threads=8):
    ps_ids = {}
    print("Length of playlists = ",len(playlists))
    batch_size = len(playlists)//no_of_threads
    start_ind = 0
    end_ind = batch_size
    t = []
    for i in range(0, no_of_threads):
        if i == (no_of_threads-1):
            t.append(MyThread(target=get_topN_playlists_helper, args=(playlists[start_ind:len(playlists)], range(start_ind, len(playlists)), topN)))
        else:
            t.append(MyThread(target=get_topN_playlists_helper, args=(playlists[start_ind:end_ind], range(start_ind, end_ind), topN)))
        start_ind += batch_size
        end_ind += batch_size

    for i in range(0, no_of_threads):
        t[i].start()
 
    for i in range(0, no_of_threads):
        ps_ids = {**t[i].join(), **ps_ids}
    
    return ps_ids
    
        
if __name__ == '__main__':
    #playlists = ['spanish playlist', 'Groovin', 'uplift', 'WUBZ','spanish playlist', 'Groovin', 'uplift', 'WUBZ']
    playlists = ['country hits', 'pop', 'songs', 'rock', 'trap music','spanish']
    ps_ids = get_topN_playlists(playlists[:], 10, 1)
    for id in ps_ids:
        for i in ps_ids[id]: 
            print(id, choices[i], i, end=', ')
        print()
        