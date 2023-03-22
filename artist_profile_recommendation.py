# -*- coding: utf-8 -*-
""" Artist profile recommendation.ipynb
    Creating an artist profile recommender based on artist's collaboration patterns. """

import sys
import ndjson
import json
import numpy as np 
import pandas as pd 
import re

""" New Section"""

with open('track_contributors.ndjson') as f:
    data = ndjson.load(f)

list_of_tracks = []  
list_of_artist_names = []
list_of_artist_roles = []
list_of_artist_profiles = []

k=0
for x in data:
    k=k+1
    for artist in x['track_contributors']:
        list_of_tracks.append("track_{}".format(k))
        list_of_artist_names.append(artist["name"])
        list_of_artist_roles.append(artist["roles"])
        list_of_artist_profiles.append(artist["has_artist_profile"])

""" Defining functions"""

def column_has_null(df, col):
    if df[col].isnull().values.any() == True:
        missings = df[col].isnull().sum()
        print ('Column has {} missing values'.format(missings))
    else:
        print ('There is no missing values in column')

def drop_null(df, col):
    # I refer to missing data in general as null, NaN, or NA values.
    #NAN means not a number
    #NA is generally interpreted as missing, does not exist
    # NULL is an empty object
    # Pandas Data Frames are based on R DataFrames. In R NA and NUll are separate things
    # pandas built on numpy which has no NA or null, but has NaN. Consequently pandas also uses NaN values
    # DatFRame.values return numpy representation of the DataFrame, the axes labels will be removed
    if df[col].isnull().values.any() == True:
        df = df[pd.notnull(df[col])]
        return (df)

def find(df, col, name):
    result = df.loc[df[col].str.lower().isin(name)]
    return(result)

def find_substring(df, col, substring, case):
    result = df[df[col].str.contains(substring, na=False, case=case)]
    return(result)

def delete_duplicated_rows(df):
    dup = df[df.duplicated()].shape[0]
    df = df.drop_duplicates(subset = None, keep ='first', inplace = False)
    return (df)

def remove_rows_with_value(df, col, value):
    index = df.loc[df[col].str.lower().isin(value)].index
    df.drop(index, inplace = True)
    return (df)

def remove_rows_by_index(df, index):
    df=df.drop(index)
    return(df)

def character_normalization(df, col):
    from unidecode import unidecode
    df[col] = df[col].apply(lambda row: row if pd.isnull(row) else unidecode(row))
    return (df)

def count(df, col):
    name_counts = df[col].value_counts(dropna = True, sort = True)
    df_name_counts = pd.DataFrame(name_counts)
    df_name_counts = df_name_counts.reset_index()
    df_name_counts.columns = [col, 'counts']
    return(df_name_counts)

def whitespace(df, col):
    df[col] = df[col].str.strip()
    df[col] = df[col].str.replace(r'/^(\s){1,}$/', '')
    return(df)

def csv(df):
    df_csv = df.to_csv(header=None, index=False).strip('\n').split('\n')
    return(df_csv)
    
#df[col] = df[col].apply(lambda row: sep.join(x for x in row))
#df[col] = df[col].apply(lambda row: row.split(sep))
    
def split_attached_words(df, col):
    #artist.apply(lambda row: ','.join(x for x in sorted(row.roles)), axis=1)
    df[col] = df[col].apply(lambda row: row if pd.isnull(row) else  ' '.join(re.findall('[A-Z][^A-Z]*', row)))
    return(df)

import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

artist = pd.DataFrame(
    {'name': list_of_artist_names, 
     'has_profile': list_of_artist_profiles,
     'track': list_of_tracks,
     'roles': list_of_artist_roles,
    })
artist.shape

""" Data cleaning
    Cleaning data in column roles
"""

column_has_null(artist, 'roles')

artist = artist.explode('roles').reset_index(drop = True)

artist['roles'] = artist['roles'].str.title()

find_substring(artist, 'roles', '[@_!#$%^&*()<>?/\|}{~:]', False)['roles'].value_counts()

""" If column row contains values separated by any of following special characters [@_!#$%^&*()<>?/\|}{~:], I want to split it and create new rows for each role unless role not intend to be splitted: Vocal & Instrumental Ensemble or Alto (Boy)."""

index = list(artist['roles'][artist['roles'].str.contains('\&.{1,}Ensemble$', na = False, case = False)].index)
artist.loc[index, 'roles'] = artist.loc[index, 'roles'].str.replace('\&','and')

artist = artist.assign(roles = artist['roles'].str.split('/|&')).explode('roles').reset_index(drop = True)
artist['roles'] = artist['roles'].str.replace('(\(|\))','')

"""I want to remove whitespace characters at the beginning and end of string and substitute multiple whitespace with single whitespace.
"""

artist = whitespace(artist,'roles')

"""In the code below I check values in column roles and their frequency. It also helps to find outliers: wrong (different) spelling of the words, and try to correct them if necessary."""

roles_count = count(artist, 'roles')
list_of_roles = list(roles_count['roles'])
roles_count.head(10)

"""In next bit of code I normalize values in column roles."""

artist = character_normalization(artist, 'roles')

"""Remove dublicated words such as Soloist Soloist
"""

artist['roles'] = artist['roles'].str.replace(r'(\b\w+\b)(\s+\1)+', r'\1')

# Solo or *solo
#find_substring(artist, 'roles', r'[Ss]olo\b(\s{0,})', False)
artist['roles'] = artist['roles'].str.replace(r'[Ss]olo\b(\s{0,})', '')

count(artist[artist['roles'].str.contains("Boy", na=False, case=False)], 'roles')

""" Cleaning data in column names"""

len(artist['name'].unique())

"""
Following bit of code intend to find and drop from dataframe rows with values that are not name of artist or title of musical collective and hence they are not usefull for our analysis

1.  rows with variations of abbreviations of anonymous, unknown or not applicable such as: 'anonymous', 'n/a', 'not applicable', 'unknown', 'anonymus', 'anonyme', 'anonim', 'anonimous' or same as roles such as Orchertra, Solo, Soloist, Ensemble or traditional with role composer.
2.  rows with variations or abbreviations of various artists such as: 'various', 'various artist', 'varios', 'v/a', 'v.a'
3.  rows that instead names contain titles such as: soundtracks, scores for movies, TV and musicals, music from film , original  cast"""

# 1.
df_1 = find(artist, 'name', ['anonymous','n/a','not applicable','unknown','anonymus', 'anonyme',  'anonim','anonimous', 'soloists', 'orchestra', 'choir','chorus', 'ensemble', 'solo', 'traditional','traditionnel', 'bible'])
df_2 = find_substring(artist, 'name', '|'.join(['anonymous','n/a','not applicable','unknown']), False)
count(pd.concat([df_1,df_2]), 'name').head(10)

# 2.
(count(find_substring(artist, 'name', '|'.join(['various', 'various artist', 'varios',r'(\W)v(\W{0,})a(\W)']), False), 'name')).head(10)

# 3.
(count(find_substring(artist, 'name', '|'.join(['soundtrack', 'music from', r'(\W)ost(\W)',r'(\W)cast(\W)', r'(\W)casts(\W)', r'(\W)score(\W)',r'(\W)applause(\W)']), False), 'name')).head(10)

artist = remove_rows_with_value(artist, 'name',['anonymous','n/a','not applicable','unknown','anonymus', 'anonyme',  'anonim', 'anonimous','soloists', 'orchestra', 'choir','chorus', 'ensemble', 'solo','traditional','traditionele','traditionelle','traditionnel', 'bible'])

list_to_remove = (['n/a', 'not applicable', 'unknown', 'anonymous', 'soundtrack', 'music from', r'(\W)ost(\W)', r'(\W)cast(\W)', r'(\W)casts(\W)', r'(\W)score(\W)','various', 'various artist', 'varios', r'(\W)v(\W{0,})a(\W)'])

index = list(find_substring(artist, 'name', '|'.join(list_to_remove), False).index)
del list_to_remove

artist = remove_rows_by_index(artist, index)
del index

""" Next I check if name includes title such as Mr , Mrs, Ms, Dr, Sir, Sr., Madam and if dataframe already include variant of name without title, I delete it for the sake of data consistency."""

regex = r'^(Mr|Mrs|Ms|Dr)\W{1,}'
index = list(find_substring(artist, 'name', regex , True).index)
without_title = artist.loc[index, 'name'].str.replace(regex, '').str.strip()
artist_name_lower = [i.lower().strip() for i in list(artist['name'])]
remove_title = [i for i in without_title if i.lower() in artist_name_lower]
index_remove_title = list(set(artist.loc[(artist.index.isin(index)) & (artist['name'].str.replace(regex, '').str.strip()).isin(remove_title)]['name'].index))

remove_title_df = artist.loc[index_remove_title]

remove_title_df = (remove_title_df.assign(name = remove_title_df['name'].str.replace(regex, '').str.strip()).reset_index(drop = True))


artist = artist.loc[~artist.index.isin(index_remove_title)]
artist = pd.concat([artist, remove_title_df]).reset_index(drop = True)

del [regex, index, without_title, artist_name_lower, remove_title, index_remove_title, remove_title_df]

regex = r'(\s{0,})((?i)sir|(?i)madam|(?i)sr\.)(\s{1,})'
artist['name'] = artist['name'].str.replace(regex, '')

""" Following bit of code intend to find collaboration between artists that are indicated with 'feat.','featuring', 'feat', 'w/' or 'with') and if possible split those artist"""

count(find_substring(artist, 'name', '|'.join(['(\W)feat(\W)', '(\W)featuring(\W)','(\W)w\/(\W)', '(\W)with(\W)', '(\W)ft(\W)']), False), 'name').head(10)

artist = (artist.assign(name = artist['name'].str.split(' (?i)feat | (?i)with | (?i)feat.')).explode('name').reset_index(drop = True))

""" In code below I try to find bands, ensembles and compound artists: artists who are generally listed together or as a band or ensemble are not considered compound artists and must be listed together such as: Piano Duo Genova & Dimitrov or ASKO | Schönberg Conducor (director) and his/her musical collective Orchestra & Chorus, Choir & Orkecter, Chor & Orkester.
I assume that they generally could have role Ensemble, hence I fill missing values in column roles with value Ensemble """

compound_index = (list(artist[(artist['name'].str.contains(r'(\W{1,})[@_!#$%^&*()<>?/\|}{~:](\W{1,})', na=False, case=False))& (((artist['has_profile'] == True) & (artist['roles'].isnull()))| ((artist['has_profile'] == False) & (artist['roles'].notnull()))| ((artist['has_profile'] == True) & (artist['roles'].notnull())))].index))

compound_index = compound_index + list(find_substring(artist, 'name', '|'.join(['(\W)his(\W)', '(\W)his(\W)','(\W)her(\W)', '(\W)her(\W)','(\W)their(\W)', '(\W)sein(\W)', '(\W)seine(\W)']), False).index)

regex_chor_orchester = (r'(Choir|Chorus|Chor|Choeurs|Orkester|Orchestra|Orchester|Orkestra|Orchestre|Band|Opera|Ensemble|Radio|TV)(\s{0,})(\&|and|und)(\s{0,})(Choir|Chorus|Chor|Choeurs|Orkester|Orchestra|Orchester|Orkestra|Orchestre|Band|Opera|Ensemble|Radio|TV)')

compound_index = compound_index + list(find_substring(artist, 'name', regex_chor_orchester , False).index)

compound_index = list(set(compound_index))

artist.loc[(artist.index.isin(compound_index)) & (artist['roles'].isna()), 'roles'] = 'Ensemble'

"""In following bit of code I find artists that are listed together but are not compound artists, bands or ensembles and they

1.   are listed separated with ',' or '&'
2.   are listed separated with multiple '|'
3.   are listed separated with single '|'
4.   are listed separated with multiple '/'
5.   are listed separated with single '/'
"""

# 1. ',' or '&'
questionable_compound_artist = find_substring(artist.loc[~artist.index.isin(compound_index)], 'name', '\&', False)['name'].str.split('\&').explode().str.split(',').explode()

questionable_compound_artist_index = list(questionable_compound_artist.index)

artist_name_lower = [i.lower().strip() for i in list(artist['name'])]

not_compound_artist = list(set([i for i in questionable_compound_artist if i.lower() in artist_name_lower]))

not_compound = find_substring(artist.loc[questionable_compound_artist_index], 'name', '|'.join(not_compound_artist), False)

not_compound_index = list(set(not_compound.index))

not_compound_artists_df = artist.loc[not_compound_index]

not_compound_artists_df = (not_compound_artists_df.assign(name = not_compound_artists_df['name'].str.split('&')).explode('name').reset_index(drop = True))
not_compound_artists_df = (not_compound_artists_df.assign(name = not_compound_artists_df['name'].str.split(',')).explode('name').reset_index(drop = True))

artist = artist.loc[~artist.index.isin(not_compound_index)]
artist = pd.concat([artist, not_compound_artists_df]).reset_index(drop = True)

# delete aadditional variables
del [questionable_compound_artist, questionable_compound_artist_index, artist_name_lower, not_compound_artist, not_compound, not_compound_index, not_compound_artists_df]

# 2. multiple '|'
not_compound_index = list(find_substring(artist, 'name', '\|(.+?)\|+?', False).index)

not_compound_artists_df = artist.loc[not_compound_index]

not_compound_artists_df = (not_compound_artists_df.assign(name = not_compound_artists_df['name'].str.split('|')).explode('name').reset_index(drop = True))

artist = artist.loc[~artist.index.isin(not_compound_index)]
artist = pd.concat([artist, not_compound_artists_df]).reset_index(drop = True)

# delete aadditional variables
del [not_compound_index, not_compound_artists_df]

# 3. single '|'
find_substring(artist, 'name', '\|', False)

index = list(find_substring(artist, 'name', '\|', False).index)

questionable_compound_artist = find_substring(artist.loc[(artist.index.isin(index)) & (artist['roles'].isna())], 'name', '\|', False)['name'].str.split('\|').explode()

questionable_compound_artist_index = list(questionable_compound_artist.index)
artist_name_lower = [i.lower().strip() for i in list(artist['name'])]

not_compound_artist = list(set([i for i in questionable_compound_artist if i.lower() in artist_name_lower]))

not_compound = find_substring(artist.loc[questionable_compound_artist_index], 'name', '|'.join(not_compound_artist), False)

not_compound_index = list(set(not_compound.index))

not_compound_artists_df = artist.loc[not_compound_index]

not_compound_artists_df = (not_compound_artists_df.assign(name = not_compound_artists_df['name'].str.split('|')).explode('name').reset_index(drop = True))

artist = artist.loc[~artist.index.isin(not_compound_index)]
artist = pd.concat([artist, not_compound_artists_df]).reset_index(drop = True)

index = list(find_substring(artist, 'name', '\|', False).index)

artist.loc[index, 'name'] = artist['name'].str.replace(r'\|', ' ')

del [index, questionable_compound_artist, questionable_compound_artist_index, artist_name_lower, not_compound_artist,not_compound, not_compound_index, not_compound_artists_df]

# 4. multiple '/' only if those are not in brackets
index_1 = list(find_substring(artist, 'name', '\/(.+?)\/+?', False).index)
index_2 = list(find_substring(artist, 'name', '\((.+?)\/(.+?)\/+?(.+?)\)', False).index)

not_compound_index = [x for x in index_1 if (x not in index_2)]
not_compound_artists_df_1 = artist.loc[not_compound_index]
not_compound_artists_df_2 = artist.loc[index_2]

not_compound_artists_df_1 = (not_compound_artists_df_1.assign(name = not_compound_artists_df_1['name'].str.split('/')).explode('name').reset_index(drop = True))
not_compound_artists_df_2 = (not_compound_artists_df_2.assign(name = not_compound_artists_df_2['name'].str.split('(')).explode('name').reset_index(drop = True))
not_compound_artists_df_2 = (not_compound_artists_df_2.assign(name = not_compound_artists_df_2['name'].str.split(',')).explode('name').reset_index(drop = True))
# not_compound_artists_df_2 = (not_compound_artists_df_2.assign(name = not_compound_artists_df_2['name'].\
#                                                            str.split('/')))
not_compound_artists_df_2['roles'] = not_compound_artists_df_2['name'].str.findall('\/(\w+)').explode()

not_compound_artists_df_2['name'] = not_compound_artists_df_2['name'].str.replace('\/(\w+)(\W{0,})', '')
artist = artist.loc[~artist.index.isin(index_1)]
artist = pd.concat([artist, not_compound_artists_df_1, not_compound_artists_df_2]).reset_index(drop = True)

del [index_1,index_2, not_compound_index, not_compound_artists_df_1, not_compound_artists_df_2]

#5. single '/'
index = list(find_substring(artist, 'name', '[^0-9]\/[^0-9]', False).index)

not_compound_artist = find_substring(artist.loc[(artist.index.isin(index)) & (artist['roles'].isna()) & (artist['has_profile']==False)], 'name', '[^0-9]\/[^0-9]', False)

not_compound_index = list(not_compound_artist.index)

not_compound_artists_df = artist.loc[not_compound_index]

not_compound_artists_df = (not_compound_artists_df.assign(name = not_compound_artists_df['name'].str.split('/')).explode('name').reset_index(drop = True))

artist = artist.loc[~artist.index.isin(not_compound_index)]
artist = pd.concat([artist, not_compound_artists_df]).reset_index(drop = True)

index = list(find_substring(artist, 'name', '\/', False).index)

artist.loc[index, 'name'] = artist['name'].str.replace(r'\/', ' ')

del [index, not_compound_artist,  not_compound_index, not_compound_artists_df]

""" In following bit of code I attempt to find titles and names formatted name as Last, First and chage order First Last if those names are present in dataset in both variants """

index_reversed = list(find_substring(artist, 'name', '\w+\s{0,},\s{0,}\w+', False).index)
reversed_title = artist.loc[index_reversed, 'name']
reversed_title = [i.split(',') for i in list(reversed_title)]

reversed_title = list(map(lambda x: ' '.join(str(e) for e in x[::-1]), reversed_title))
reversed_title
artist_name_lower = [i.title().replace(" ","") for i in list(artist['name'])]
reverse_title = [i.title().replace(" ","") for i in reversed_title if i.title().replace(" ","") in artist_name_lower]

index_reverse_title = list(set(artist.loc[(artist.index.isin(index_reversed)) & pd.Series([' '.join(i.split(',')[::-1]).title().replace(" ","") for i in list(artist['name'])]).isin(reverse_title)]['name'].index))

reverse_title_df = artist.loc[index_reverse_title]

reverse_title_df = (reverse_title_df.assign(name = pd.Series([' '.join(i.split(',')[::-1]).title().replace(" ","") for i in list(artist['name'])])).reset_index(drop = True))

reverse_title_df = split_attached_words(reverse_title_df, 'name')
artist = artist.loc[~artist.index.isin(index_reverse_title)]
artist = pd.concat([artist, reverse_title_df]).reset_index(drop = True)

del [index_reversed, reversed_title, artist_name_lower, reverse_title, index_reverse_title, reverse_title_df]

""" Following bit of code find out if there are records in name column with brackets [], (), {} and if there are what type of information they contain:


1. include birth and death dates of type:  (1888-1945), (d1888-1945), (d.1888-1945), (1888), (?-1888), (1888-?) ect. 
2. include roles 
3. rest
"""

names_with_brackets = find_substring(artist, 'name', r'\((.*?)\)', False).index
within_brackets = find_substring(artist, 'name', r'\((.*?)\)', False)['name']
print("There are {} records with brackets".format(len(within_brackets)))

regex = '\((.*?)\)'
list_od_values_within_brackets = within_brackets.str.findall(regex).explode()
list_od_values_within_brackets

#1. 
regex = r'\(\d+-\d+\)|\((\w{0,})(\W{0,})\d+-\d+\)|\((\w{0,})(\W{0,})\d+(\W{0,})\)|\(\d+(?i)th(\s{0,})(?i)\w+\)|\((\w{0,})(\W{0,})\d+(\s{0,})(A(\W{0,})D(\W{0,})|C(\W{0,})D(\W{0,}))\)'
#find_substring(artist, 'name', regex, False)
artist['name'] = artist['name'].str.replace(regex, '')
del regex

regex = '\((.*?)\)'
artist['name'] = artist['name'].str.replace(regex, '')
del regex

""" Now I want to remove whitespace characters at the beginning and end of string.

In following bit of code for this part I use string method title() in order to unite capitalizaation and adress problem of spelling differences."""

artist['name'] = artist['name'].str.title().str.replace('[-\'@_!#$%^&*()<>?/\|}{~:]', ' ')
artist = whitespace(artist, 'name')
artist = split_attached_words(artist, 'name')

artist = whitespace(artist, 'roles')
artist['roles'] = artist['roles'].str.title().str.replace('[-\'@_!#$%^&*()<>?/\|}{~:]', ' ')
artist = split_attached_words(artist, 'roles')


artist[artist['roles'].str.contains("Solo", na=False, case=False)]['roles'].value_counts()

""" And lastly I divide roles per 5 groups: Conductor, Ensemble, Composer, Soloist and Others"""

regex_other = (r'Author|Actor|Librettist|Consultant|^Art|Personnel|Reconstruction|Copy|Lyricist|Arranger|\
Contributor|Coordinator|Additional|Mixer|Engineer|Arrangements|Casting|Assistant$|photographer|\
Word|Interviewer|Design|Programm|Compiler|Editor|Masterer|Writer|Computer|Production|\
^Misc.|Digital|Narrat|Contractor|Record|Speaker|Producer|Realization|Transcriber|Produced|Effects\
Interjections|Musical|^Ton')

artist.loc[list(find_substring(artist, 'roles', regex_other, False).index), 'roles'] = 'Others'

regex_ensemble = r'Orchestra$|Choir$|Chorus$|quartet$|ensemble|trio|band$|Quartet|duo|duet'
artist.loc[list(find_substring(artist, 'roles', regex_ensemble, False).index), 'roles'] = 'Ensemble'

regex_composer = r'Adapter|Orchestration|Orchesrtator'
artist.loc[list(find_substring(artist, 'roles', regex_composer, False).index), 'roles'] = 'Composer'

regex_conductor = r'Choir|Director|Chorus|Conductor'
artist.loc[list(find_substring(artist, 'roles', regex_conductor, False).index), 'roles'] = 'Conductor'

regex_soloist = r'^(?!Conductor|Ensemble|Composer|Others)'
artist.loc[list(find_substring(artist, 'roles', regex_soloist, False).index), 'roles'] = 'Soloist'

set(artist['roles'])

# delete dublicated rows
artist = delete_duplicated_rows(artist).reset_index()

roles_count = count(artist, 'roles')
list_of_roles = list(roles_count['roles'])
roles_count.head(10)

column_has_null(artist, 'roles')

column_has_null(artist,'name')
#artist = drop_null(artist, 'name')

""" Graph

Given dataset contains a set of releases, where each release is tagged with a list of artists that participated in creating it.

This makes it much simpler to count the number of collaborations between artists, and creating the graph.

My idea is that using this approach it can provide relevant recommendations for a problem of the creation of new artist profiles.

For the sake of analysis I drop all rows with roles Others.
"""

artist = artist.loc[artist['roles']!='Others']

""" In the following bit of code I create graph, where nodes are artist and egdes are connection between artists. These edges have no direction since I assumed symmetric relationship between artists in absence of additional information."""

from collections import defaultdict
from itertools import combinations

df_grouped = artist.groupby('track')['name'].apply(list).reset_index()

d = defaultdict(int)

for idx, row in df_grouped.iterrows():
    for comb in combinations(row['name'], 2):
        d[frozenset(comb)] += 1

d = {tuple(sorted(k)): v for k, v in d.items()}

df_out = pd.DataFrame(list(d.items())).rename(columns={0: 'node', 1: 'collaborations'})

df_out = (df_out.join(df_out['node'].apply(pd.Series)).drop('node', 1).rename(columns={0: 'node 1', 1: 'node 2'}))

# The edges must be 2-tuple (u,v), hence I drop 1-tuple edges
empty_edges = df_out.loc[df_out['node 2'].isna()]
df_out = df_out.drop(empty_edges.index).reset_index(drop=True)

# By using sum() in group by I chosse true if sum >0 and False if sum ==0
df = artist.groupby(['name'])['has_profile'].apply(list).reset_index()
df['has_profile'] = df['has_profile'].apply(lambda x : True if len(set(x))==2 else x[0])

df_out.sort_values(by = 'collaborations', ascending=False)

# Create an empty graph
import networkx as nx
import itertools
import matplotlib.pyplot as plt
G = nx.Graph()

from random import sample
node_names = list(df['name'])                                     
edges =  list(zip(df_out['node 1'], df_out['node 2']))

G = nx.Graph()
G.add_nodes_from(node_names)
G.add_edges_from(edges)

"""In following bit of code i add attribute 'has_profile'  to nodes"""

has_profile_dict = dict(df.values.tolist())

nx.set_node_attributes(G, has_profile_dict, 'has_profile')

"""Graph has more than 1 component"""

print(nx.is_connected(G))

print(nx.info(G))

""" And now I want to find which nodes are most important in my network with most common centrality measures degree centrality """

degree_dict = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree()')

import operator as ope

ds = [degree_dict, has_profile_dict]
d = {}
for i in  degree_dict.keys():
    d[i] = tuple(d[i] for d in ds)

sorted_degree = sorted(d.items(), key = ope.itemgetter(1),  reverse = True)
#sorted_degree

k=1
for i in sorted_degree:
    if i[1][1]==False:
        print(i)
    k=k+1
    if k==10:
        break

""" Conclusion. I presented an analysis of musical influence based on collaboration between artists in a graph structure. From my result, I can see that among 10 top recommended profiles:  2 are ensembles, 1 conductor and rest are soloists, whereas top 10 recommended artists with profiles are composers. It is quite unexpected result to me, since data set contains numerous essemblies that tend to collaborate between each other (choir and orchester f.e), various soloists, conductors. """

find_substring(artist, 'name', 'John  Mauceri', False)

sorted_degree[:10]

""" Further work can be done in increasing quality of data, for example I would remove name of places, update role based on information in name, try to solve issue of different spelling, such as J.S.Bach, J Bach and Johann Sebastian Bach and so on.

Next I could explore relationship using different centrality measures or experiment with weights associated with each artist. 

Since data set is quite large it is hard to vizualize network. Probably as further work, I would draw it so that I could see  its shape,is it centralized or decentralized, dense or sparse, cyclical or linear. """