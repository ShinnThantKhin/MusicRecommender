import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("Set2")
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity

def search_song(song_name, data):
    data_song = data.query('name == @song_name')
    if data_song.shape[0] == 0:
        found_flag = False
        found_song = None
        # raise Exception('The song does not exist in the dataset!')
    else:
        found_flag = True
        found_song = data_song[['name', 'artists', 'release_date']].to_numpy()
         # print(f"Great! This song is in the dataset: \n {dat_song[['name', 'artists', 'release_date']].to_numpy()}")
    return found_flag, found_song

def get_feature_vector(song_name, year, data, features_list):
    print(data.head())
    # select dat with the song name and year
    data_song = data.query('name == @song_name and year == @year')
    song_repeated = 0
    if len(data_song) == 0:
        raise Exception('The song does not exist in the dataset or the year is wrong! \
                        \n Use search function first if you are not sure.')
    if len(data_song) > 1:
        song_repeated = data_song.shape[0]
        print(f'Warning: Multiple ({song_repeated}) songs with the same name and artist, the first one is selected!')
        data_song = data_song.head(1)
    feature_vector = data_song[features_list].values
    return feature_vector, song_repeated

# define a function to get the most similar songs
def show_similar_songs(song_name, year, data, features_list, top_n=10, plot_type='wordcloud', font_path=None):
    feature_vector, song_repeated = get_feature_vector(song_name, year, data, features_list)
    feature_for_recommendation = data[features_list].values
    # calculate the cosine similarity
    similarities = cosine_similarity(feature_for_recommendation, feature_vector).flatten()

    # get the index of the top_n similar songs not including itself
    if song_repeated == 0:
        related_song_indices = similarities.argsort()[-(top_n+1):][::-1][1:]
    else:
        related_song_indices = similarities.argsort()[-(top_n+1+song_repeated):][::-1][1+song_repeated:]
        
    # get the name, artist, and year of the most similar songs
    similar_songs = data.iloc[related_song_indices][['name', 'artists', 'year']]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    if plot_type == 'wordcloud':
        # make a word cloud of the most similar songs and year, use the simalirity score as the size of the words
        similar_songs['name+year'] = similar_songs['name'] + ' (' + similar_songs['year'].astype(str) + ')'
        # create a dictionary of song and their similarity
        song_similarity = dict(zip(similar_songs['name+year'], similarities[related_song_indices]))
        # sort the dictionary by value
        song_similarity = sorted(song_similarity.items(), key=lambda x: x[1], reverse=True)
        # # create a mask for the word cloud
        # mask = np.array(Image.open("spotify-logo.png"))
        # create a word cloud
        wordcloud = WordCloud(width=1200, height=600, max_words=50, 
                            background_color='white', colormap='Set2',font_path=font_path).generate_from_frequencies(dict(song_similarity))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{top_n} most similar songs to: {song_name} ({year})', fontsize=16)
        plt.tight_layout(pad=0)
    
    elif plot_type == 'bar':
        # plot the text of the most similar songs and year in order, like a stacked bar chart
        similar_songs['name+year'] = similar_songs['name'] + ' (' + similar_songs['year'].astype(str) + ')'
        # create a dictionary of song and their similarity
        song_similarity = dict(zip(similar_songs['name+year'], similarities[related_song_indices]))
        # sort the dictionary by value
        song_similarity = sorted(song_similarity.items(), key=lambda x: x[1], reverse=True)
        # plot the text of the most similar songs and year in order, like a stacked bar chart
        plt.barh(range(len(song_similarity)), [val[1] for val in song_similarity], 
                 align='center', color=sns.color_palette('pastel', len(song_similarity)))
        plt.yticks(range(len(song_similarity)), [val[0] for val in song_similarity])
        plt.gca().invert_yaxis()
        plt.title(f'{top_n} most similar songs to: {song_name} ({year})', fontsize=16)
        min_similarity = min(similarities[related_song_indices])
        max_similarity = max(similarities[related_song_indices])
        # add song name on the top of each bar
        for i, v in enumerate([val[0] for val in song_similarity]):
            plt.text(min_similarity*0.955, i, v, color='black', fontsize=8)
        # plt.xlabel('Similarity', fontsize=15)
        # plt.ylabel('Song', fontsize=15)
        plt.xlim(min_similarity*0.95, max_similarity)
        # not show figure frame and ticks
        plt.box(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        
    else:
        raise Exception('Plot type must be either wordcloud or bar!')
    
    return fig

# load data
data = pd.read_csv('data/data_for_recommender.csv')

song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

all_features = song_features_normalized + song_features_not_normalized + ['decade', 'popularity']

st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def main():
    st.title("Music Recommender")
    st.markdown("Welcome to music recommender! \
                \n You can search for a song and get recommendations based on the song you searched for. Enjoy!")

    # add a search box for searching the song by giving capital letters and year
    st.markdown("### Ready to get recommendations based on song?")
    song_name = st.text_input('Enter the name of the song')
    if song_name != '':
        song_name = song_name.upper()
    st.markdown( "*** If you are not sure if the song is in the database or not sure, Please click the button below to search for the song! ")
    if st.button('Search for my song'):
        found_flag, found_song = search_song(song_name, data)
        if found_flag:
            st.markdown("Perfect, this song is in the dataset:")
            st.markdown(found_song)
        else:
            st.markdown("Sorry, this song is not in the dataset. Please try another song!")

    year = st.text_input('Enter the year of the song (e.g. 2019)')
    if year != '':
        year = int(year)

    # add selectbox for selecting the features
    st.markdown("### Select Features")
    features = st.multiselect('Select the features you care about', all_features)

    # add a slider for selecting the number of recommendations
    st.markdown("### Number of Recommendations")
    num_recommendations = st.slider('Select the number of recommendations', 5, 20, 10)

    if st.button('Get Recommendations'):
        if song_name == '':
            st.markdown("Please enter the name of the song!")
        elif year == '':
            st.markdown("Please enter the year of the song!")
        else:
            
            # show the most similar songs in wordcloud
            fig_cloud = show_similar_songs(song_name, year, data, features, num_recommendations, plot_type='wordcloud')
            st.markdown(f"### Great! Here are your recommendation for \
                        \n#### {song_name} ({year})!")
            st.pyplot(fig_cloud)

            # show the most similar songs in bar chart
            fig_bar = show_similar_songs(song_name, year, data, features, top_n=5, plot_type='bar')
            st.markdown("### Get a closer look at the top 5 recommendations for you!")
            st.pyplot(fig_bar)

if __name__ == "__main__":
    main()
