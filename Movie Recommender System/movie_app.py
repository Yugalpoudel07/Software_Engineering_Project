import streamlit as st
import pandas as pd
import pickle
import requests

class Authenticator:
    """Handles user authentication"""
    
    def __init__(self):
        # Hardcoded credentials for demo (use secure database in production)
        self.credentials = {
            'user1': 'password123',
            'admin': 'adminpass'
        }
        
    def verify_credentials(self, username, password):
        """Verify username and password"""
        return username in self.credentials and self.credentials[username] == password

class MovieRecommenderApp:
    """Streamlit app for movie recommender system"""
    
    def __init__(self):
        self.auth = Authenticator()
        self.movies = None
        self.similarity = None
        
    def load_models(self):
        """Load precomputed models from pickle files"""
        try:
            with open('movie_list.pkl', 'rb') as f:
                self.movies = pickle.load(f)
            with open('similarity.pkl', 'rb') as f:
                self.similarity = pickle.load(f)
            return True
        except FileNotFoundError as e:
            st.error(f"Error: {e}. Ensure movie_list.pkl and similarity.pkl are in the same directory.")
            return False

    def fetch_poster(self, movie_id):
        """Fetch movie poster from TMDB API"""
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
            data = requests.get(url).json()
            poster_path = data['poster_path']
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        except Exception:
            return "https://via.placeholder.com/500x750?text=No+Poster"

    def recommend(self, movie):
        """Recommend top 5 similar movies with posters"""
        try:
            index = self.movies[self.movies['title'] == movie].index[0]
            distances = sorted(list(enumerate(self.similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
            recommendations = []
            for i in distances:
                movie_data = self.movies.iloc[i[0]]
                recommendations.append({
                    'title': movie_data['title'],
                    'movie_id': movie_data['movie_id'],
                    'poster': self.fetch_poster(movie_data['movie_id'])
                })
            return recommendations
        except IndexError:
            return None

    def login_page(self):
        """Render login page"""
        st.title("Movie Recommender System - Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if self.auth.verify_credentials(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    def main_page(self):
        """Render main recommendation page"""
        st.title("Movie Recommender System")
        st.write(f"Welcome, {st.session_state['username']}!")
        
        if not self.load_models():
            return
            
        st.header("Find Similar Movies")
        movie_list = self.movies['title'].values
        selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)
        
        if st.button("Show Recommendations"):
            recommendations = self.recommend(selected_movie)
            if recommendations:
                st.success(f"Recommendations for '{selected_movie}':")
                cols = st.columns(5)
                for i, rec in enumerate(recommendations):
                    with cols[i]:
                        st.text(rec['title'])
                        st.image(rec['poster'], width=150)  # Updated to use 'width' parameter
            else:
                st.error(f"Movie '{selected_movie}' not found in the dataset")

def main():
    """Main application logic"""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
    
    app = MovieRecommenderApp()
    
    if not st.session_state['logged_in']:
        app.login_page()
    else:
        app.main_page()

if __name__ == "__main__":
    main()