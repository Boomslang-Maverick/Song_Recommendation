#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data=pd.read_csv("C:\\Users\\Admin\\Desktop\\BDS2020\\data.csv")
data


# In[3]:


year_data=pd.read_csv("C:\\Users\\Admin\\Desktop\\BDS2020\\data_by_year.csv")
year_data


# In[4]:


genre_data=pd.read_csv("C:\\Users\\Admin\\Desktop\\BDS2020\\data_by_genres.csv")
genre_data


# In[5]:


artist_data=pd.read_csv("C:\\Users\\Admin\\Desktop\\BDS2020\\data_by_artist.csv")
artist_data


# In[6]:


data.isnull().sum()


# In[7]:


genre_data.isnull().sum()


# In[8]:


year_data.isnull().sum()


# In[9]:


artist_data.isnull().sum()


# In[10]:


pip install yellowbrick


# In[11]:


from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

features = np.array(feature_names)


visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)     
visualizer.show()


# In[12]:


sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features)
fig.show()


# In[13]:


top10_genres = genre_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()


# In[14]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist


# In[15]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)
from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()


# In[16]:


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels


from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()


# In[17]:


pip install spotipy


# In[18]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import time 
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from sklearn.pipeline import Pipeline
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from math import pi, ceil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[19]:


song_df=pd.read_csv("C:\\Users\\Admin\\Desktop\\BDS2020\\cleaned_data_spotify.csv")
song_df.drop('Unnamed: 0',axis=1,inplace=True)
song_df


# In[20]:


client_id = '7573e000a6a54da2acaef0edd94a3236'
client_secret ='632231db7c1244c88614a03b231691fa'


client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[21]:


song_df.isnull().sum()


# In[22]:


ids = []
def getTrackIDs(playlist_id):
    
    try:
        playlist = spotify.playlist(playlist_id)
    
        for item in playlist['tracks']['items']:
            track = item['track']
            if track is not None:
                ids.append(track['id'])
        return ids
    except:
        print("Entered a invalid a link")
        pass


# In[ ]:


def getTrackFeatures(id_s):
    
    
   
    
    meta = spotify.track(id_s)
        # meta
    name = meta['name']

    artists = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    duration_ms = meta['duration_ms']
    popularity = meta['popularity']
    explicit=meta['explicit']
    features = spotify.audio_features(tracks=id_s)
    tempo=features[0]['tempo']
    valence=features[0]['valence']
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    mode=features[0]['mode']
    key=features[0]['key']
    import datetime

    track = [valence, acousticness,artists,danceability,duration_ms,energy,explicit,id_s,instrumentalness,loudness,key,liveness,mode,name,popularity,release_date,speechiness,tempo]
    return track


tracks=[]
def get_playlist_df():

    ids = getTrackIDs(input("Enter a link to your playlist "))
    for i in range(len(ids)):
        time.sleep(.5)
        track = getTrackFeatures(ids[i])
        tracks.append(track)  
    playlist_df = pd.DataFrame(tracks, columns =  ["valence","acousticness","artists","danceability","duration_ms","energy","explicit","id","instrumentalness","loudness","key","liveness","mode","name","popularity","release_date","speechiness","tempo"])
    playlist_df['year'] = pd.DatetimeIndex(playlist_df['release_date']).year
    year =playlist_df['year']
    playlist_df.drop(labels=['year'], axis=1, inplace = True)
    playlist_df.insert(2, 'year', year)    
    return playlist_df    
     playlist_df=get_playlist_df()   


# In[23]:


playlist_df=pd.read_csv("C:\\Users\\Admin\\Desktop\\BDS2020\\playlist.csv")


# In[24]:


playlist_df


# In[25]:


columns_cluster = ['valence', 'acousticness','instrumentalness','energy']


# In[26]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss = StandardScaler()

songs_scaled = ss.fit_transform(song_df[columns_cluster])


playlist_scaled = ss.fit_transform(playlist_df[columns_cluster])
columns_to_cluster_scaled = [ 'valence_scaled','acousticness_scaled',"instrunentalness_scaled",'energy_scaled']

df_songs_scaled = pd.DataFrame(songs_scaled, columns=columns_to_cluster_scaled)
df_playlist_scaled = pd.DataFrame(playlist_scaled, columns=columns_to_cluster_scaled)
df_song_scaled_f=pd.concat([df_songs_scaled,song_df[['name','artists']]],axis=1)
df_playlist_scaled_f=pd.concat([df_playlist_scaled,playlist_df[['name','artists']]],axis=1)
df_songs_scaled.isnull().sum()


# In[27]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
n_clusters = range(2,12)
ssd = []
sc = []


for n in n_clusters:
    km = KMeans(n_clusters=n, max_iter=500, n_init=10, init='k-means++', random_state=42)
    km.fit(songs_scaled)
    preds = km.predict(songs_scaled) 
    centers = km.cluster_centers_ 
    ssd.append(km.inertia_) 
    score = silhouette_score(songs_scaled, preds, metric='euclidean')
    sc.append(score)
    print("Number of Clusters = {}, Silhouette Score = {}".format(n, score))


# In[28]:


plt.plot(n_clusters, sc, marker='.', markersize=12, color='red')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score behavior over the number of clusters')
plt.show()


# In[29]:


for n, s in zip(n_clusters, ssd):
    print('Number of Clusters = {}, Sum of Squared Distances = {}'.format(n, s))


# In[30]:


plt.plot(n_clusters, ssd, marker='.', markersize=12)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal K')
plt.show()


# In[31]:


k=6

model = KMeans(n_clusters=k, random_state=42).fit(songs_scaled)
pred = model.predict(songs_scaled)
print('10 first clusters: ', model.labels_[:10])


# In[32]:


model_playlist=KMeans(n_clusters=k, random_state=42).fit(playlist_scaled)
pred_playlist = model.predict(playlist_scaled)


# In[33]:


df_songs_scaled['cluster'] = model.labels_

df_songs_scaled['cluster'].value_counts().plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Amount of songs')
plt.title('Amount of songs per cluster')
plt.show()


# In[34]:


c0 = df_songs_scaled[df_songs_scaled['cluster']==0]
c1 = df_songs_scaled[df_songs_scaled['cluster']==1]
c2 = df_songs_scaled[df_songs_scaled['cluster']==2]
c3 = df_songs_scaled[df_songs_scaled['cluster']==3]
c4 = df_songs_scaled[df_songs_scaled['cluster']==4]
c5 = df_songs_scaled[df_songs_scaled['cluster']==5]
c0


# In[35]:


c0.drop(['cluster'] ,axis=1,inplace=True)
c0=c0.melt(var_name='groups', value_name='vals')
c1.drop(['cluster'] ,axis=1,inplace=True)
c1=c1.melt(var_name='groups', value_name='vals')
c2.drop('cluster' ,axis=1,inplace=True)
c2=c2.melt(var_name='groups', value_name='vals')
c3.drop('cluster' ,axis=1,inplace=True)
c3=c3.melt(var_name='groups', value_name='vals')
c4.drop('cluster' ,axis=1,inplace=True)
c4=c4.melt(var_name='groups', value_name='vals')
c5.drop('cluster' ,axis=1,inplace=True)
c5=c5.melt(var_name='groups', value_name='vals')


# In[36]:


f, axes = plt.subplots(6, 1,figsize=(15,25))

ax= sns.violinplot( data=c0 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])
ax = sns.violinplot( data=c2 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[2])
ax = sns.violinplot( data=c3 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c4 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[4])
ax = sns.violinplot( data=c5 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[5])

plt.show()


# In[37]:


display(df_songs_scaled['cluster'].value_counts())
minor_cluster = df_songs_scaled['cluster'].value_counts().tail(1)
print("Amount of songs in the smallest cluster: ", int(minor_cluster.values))


# In[38]:


df_songs_joined = pd.concat([song_df,df_songs_scaled], axis=1).set_index('cluster')

for cluster in range(k):
    display(df_songs_joined.loc[cluster, ['artists','name']].sample(frac=1).head(10))


# In[39]:


pca = PCA(n_components=3, random_state=0)
songs_pca = pca.fit_transform(songs_scaled)
pca.explained_variance_ratio_.sum()


# In[40]:


df_pca = pd.DataFrame(songs_pca, columns=['C1', 'C2', 'C3'])
df_pca['cluster'] = model.labels_
df_pca.head()


# In[41]:


sampled_clusters_pca = pd.DataFrame()

for c in df_pca.cluster.unique():
    df_cluster_sampled_pca = df_pca[df_pca.cluster == c].sample(n=int(minor_cluster), random_state=20)
    sampled_clusters_pca = pd.concat([sampled_clusters_pca,df_cluster_sampled_pca], axis=0)
sampled_clusters_pca.cluster.value_counts()


# In[42]:


def clusters_view_using_pca(data_graph):
    sns.scatterplot(x='C1', y='C2', hue='cluster', data=data_graph, legend="full", palette='Paired')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Clusters view using PCA')
    plt.show()
clusters_view_using_pca(sampled_clusters_pca)


# In[43]:


fig = plt.figure()
fig.suptitle('Clusters view with 3 dimensions using PCA')
ax = Axes3D(fig)

ax.scatter(df_pca['C1'], df_pca['C2'], df_pca['C3'],
           c=df_pca['cluster'], cmap='Paired')

ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
plt.show()


# In[56]:


tsne = TSNE(n_components=2, perplexity=50, random_state=42)
songs_tsne = tsne.fit_transform(songs_scaled)
df_tsne = pd.DataFrame(songs_tsne, columns=['C1', 'C2'])
df_tsne['cluster'] = model.labels_
df_tsne.head()


# In[45]:


sampled_clusters_tsne = pd.DataFrame()

for c in df_tsne.cluster.unique():
    df_cluster_sampled_tsne = df_tsne[df_tsne.cluster == c].sample(n=int(minor_cluster), random_state=42)
    sampled_clusters_tsne = pd.concat([sampled_clusters_tsne,df_cluster_sampled_tsne], axis=0)
sampled_clusters_tsne.cluster.value_counts()


# In[46]:


sns.scatterplot(x='C1', y='C2', hue='cluster', data=sampled_clusters_tsne, legend="full", palette='Paired')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Clusters view using t-SNE')
plt.show()


# In[47]:


df_playlist_scaled['cluster'] = model_playlist.labels_

df_playlist_scaled['cluster'].value_counts().plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Amount of songs')
plt.title('Amount of songs per cluster')
plt.show()


# In[48]:


songs_pca = pca.fit_transform(playlist_scaled)
pca.explained_variance_ratio_.sum()
df_playlist_pca = pd.DataFrame(songs_pca, columns=['C1', 'C2', 'C3'])
df_playlist_pca['cluster'] = model_playlist.labels_
df_playlist_pca


# In[49]:


sns.scatterplot(x='C1', y='C2', hue='cluster', data=df_playlist_pca, legend="full", palette='Paired')


# In[50]:


pip install plotly


# In[51]:


df_playlist_scaled_f['playlist_track'] = 1
# drop unnecessary features
# label each label track as 0 in the playlist column
df_song_scaled_f['playlist_track'] = 0
# create main dataframe using playlist songs and all label's songs, to be fed into model
main_df = pd.concat([df_playlist_scaled_f, df_song_scaled_f], axis=0).reset_index(drop=True)
# convert main dataframe to array
main_array = np.array(main_df.drop(['name', 'artists','playlist_track'], axis=1))
k_means = KMeans(random_state=1, n_clusters=6) # default n_clusters = 8
# fit to data
k_means.fit(main_array)
# predict which clusters each song belongs too
predicted_clusters = k_means.fit_predict(main_array)
# each instance was assigned to one of the clusters
print(predicted_clusters)


# In[52]:


pred_series = pd.Series(predicted_clusters)
main_df_w_pred = pd.concat([main_df, pred_series], axis=1)
main_df_w_pred.rename(columns={0:'cluster'},inplace=True)
main_df_w_pred.cluster.unique()


# In[53]:


# look at the clusters most similar to user's playlist
relevant_clusters = main_df_w_pred['cluster'][main_df_w_pred.playlist_track==1]

# drop the user playlist songs from df, so they aren't recommended again
pred_df = main_df_w_pred.drop(main_df_w_pred[main_df_w_pred.playlist_track==1].index)
pred_df


# In[54]:


relevant_clusters


# In[57]:


recs=[]
for k in relevant_clusters:
    recs +=pred_df[['name', 'artists']][pred_df.cluster==k].sample(1).values.tolist()
    pred_df.drop(pred_df[['name', 'artists']][pred_df.cluster==k].sample(1).index,inplace=True)

print(' similar to your playlist:\n')
for song in recs:
    print(song[0]+', by '+ song[1])


# In[ ]:




