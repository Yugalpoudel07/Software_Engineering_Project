import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class DataPreprocessor:
    """Handles loading and preprocessing of movie data"""
    
    def __init__(self, movies_path='tmdb_5000_movies.csv', credits_path='tmdb_5000_credits.csv'):
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies = None
        self.new_df = None

    def load_data(self):
        """Load and merge movies and credits datasets"""
        try:
            self.movies = pd.read_csv(self.movies_path)
            credits = pd.read_csv(self.credits_path)
            self.movies = self.movies.merge(credits, on='title')
            self.movies = self.movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            self.movies.dropna(inplace=True)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def convert(self, text):
        """Convert JSON-like string to list of names"""
        return [item['name'] for item in ast.literal_eval(text)]

    def convert_cast(self, text):
        """Convert cast to list of top 3 names"""
        return [item['name'] for item in ast.literal_eval(text)[:3]]

    def fetch_director(self, text):
        """Extract director names from crew"""
        return [item['name'] for item in ast.literal_eval(text) if item['job'] == 'Director']

    def collapse(self, lst):
        """Remove spaces from list elements"""
        return [item.replace(" ", "") for item in lst]

    def preprocess(self):
        """Preprocess the dataset to create tags"""
        # Convert genres, keywords, cast, and crew
        self.movies['genres'] = self.movies['genres'].apply(self.convert)
        self.movies['keywords'] = self.movies['keywords'].apply(self.convert)
        self.movies['cast'] = self.movies['cast'].apply(self.convert_cast)
        self.movies['crew'] = self.movies['crew'].apply(self.fetch_director)
        
        # Remove spaces from text
        self.movies['genres'] = self.movies['genres'].apply(self.collapse)
        self.movies['keywords'] = self.movies['keywords'].apply(self.collapse)
        self.movies['cast'] = self.movies['cast'].apply(self.collapse)
        self.movies['crew'] = self.movies['crew'].apply(self.collapse)
        
        # Split overview into words
        self.movies['overview'] = self.movies['overview'].apply(lambda x: x.split())
        
        # Create tags
        self.movies['tags'] = self.movies['overview'] + self.movies['genres'] + self.movies['keywords'] + self.movies['cast'] + self.movies['crew']
        self.movies['tags'] = self.movies['tags'].apply(lambda x: " ".join(x))
        
        # Create final dataframe
        self.new_df = self.movies[['movie_id', 'title', 'tags']]
        return self.new_df

    def get_processed_data(self):
        """Return processed dataframe"""
        return self.new_df

class RecommenderModel:
    """Handles recommendation logic and model persistence"""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.vectors = None
        self.similarity = None
        self.movies = None

    def build_model(self, df):
        """Build recommendation model using CountVectorizer and cosine similarity"""
        try:
            self.movies = df
            self.vectors = self.vectorizer.fit_transform(df['tags']).toarray()
            self.similarity = cosine_similarity(self.vectors)
            return True
        except Exception as e:
            print(f"Error building model: {e}")
            return False

    def recommend(self, movie):
        """Recommend top 5 similar movies"""
        try:
            index = self.movies[self.movies['title'] == movie].index[0]
            distances = sorted(list(enumerate(self.similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
            recommendations = [self.movies.iloc[i[0]][['movie_id', 'title']].to_dict() for i in distances]
            return recommendations
        except IndexError:
            return None

    def save_model(self, movies_path='movie_list.pkl', similarity_path='similarity.pkl'):
        """Save model components to pickle files"""
        if self.movies is None or self.similarity is None:
            raise ValueError("Model not built yet")
        with open(movies_path, 'wb') as f:
            pickle.dump(self.movies, f)
        with open(similarity_path, 'wb') as f:
            pickle.dump(self.similarity, f)

class MovieRecommenderSystem:
    """Main class to orchestrate the movie recommender system"""
    
    def __init__(self, movies_path='tmdb_5000_movies.csv', credits_path='tmdb_5000_credits.csv'):
        self.preprocessor = DataPreprocessor(movies_path, credits_path)
        self.model = RecommenderModel()
        self.is_initialized = False

    def initialize(self):
        """Initialize the system by loading and preprocessing data, then building the model"""
        if not self.preprocessor.load_data():
            raise ValueError("Failed to load data")
        self.preprocessor.preprocess()
        self.is_initialized = self.model.build_model(self.preprocessor.get_processed_data())
        if not self.is_initialized:
            raise ValueError("Failed to build model")

    def recommend(self, movie):
        """Get recommendations for a given movie"""
        if not self.is_initialized:
            raise ValueError("System not initialized")
        return self.model.recommend(movie)

    def save_models(self):
        """Save the model to pickle files"""
        if not self.is_initialized:
            raise ValueError("System not initialized")
        self.model.save_model()

if __name__ == "__main__":
    # Example usage
    recommender = MovieRecommenderSystem()
    try:
        recommender.initialize()
        recommendations = recommender.recommend('Avatar')
        print("\nRecommendations for 'Avatar':")
        for rec in recommendations:
            print(f"- {rec['title']} (ID: {rec['movie_id']})")
        recommender.save_models()
    except Exception as e:
        print(f"Error: {e}")