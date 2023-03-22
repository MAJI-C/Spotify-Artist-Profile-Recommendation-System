# Spotify Artist Profile Recommendation System

This project aims to build a recommendation system for Spotify artist profiles based on their collaboration patterns. The idea behind the recommendation system is that if two artists collaborate frequently, then they are likely to share similar audiences and musical styles, and therefore, recommending one artist profile to a user who is a fan of the other artist is likely to be successful.

##### Data Collection

The data used in this project is collected from the Spotify Web API. The data collected is in JSON format and contains information about track contributors, such as artist name, role, and whether they have a Spotify artist profile. The data is then converted to a Pandas DataFrame for further processing and analysis.

##### Data Cleaning and Preparation

The collected data is cleaned and prepared for analysis. The cleaning and preparation process includes handling missing values, removing duplicates, normalizing character encoding, splitting roles, and removing special characters ect.

##### Data Analysis

The cleaned data is analyzed to identify the most frequently collaborating artists and the most common roles in collaborations. The analysis also includes exploring the distribution of the number of tracks per artist and the number of collaborators per track.

##### Recommendation System

The recommendation system is based on a collaborative filtering approach, where we identify similar artists based on their collaboration patterns. The approach involves building a co-occurrence matrix, where each cell represents the number of times two artists have collaborated. The matrix is then used to calculate similarity scores between artists using the cosine similarity measure. The similarity scores are used to generate recommendations for each artist profile.

##### Output

The project does not output any files or data. It is a proof-of-concept project that analyzes the data and provides recommendations for further analysis.

##### Future Improvements

1. Include more data sources, such as social media profiles and tour information, to improve the recommendation system's accuracy.
2. Explore other recommendation algorithms, such as content-based filtering and hybrid filtering, to determine which approach provides the best results.
3. Build a user interface to enable users to search for artist profiles and receive recommendations based on their preferences.

### Built With

- Python 3.x
- numpy
- pandas
- ndjson
- re
