# streamlit_app.py
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define categories
categories = {
    1: 'religion',
    2: 'age',
    3: 'ethnicity',
    4: 'gender',
    5: 'other_cyberbullying',
    6: 'not_cyberbullying'
}

# Streamlit app
st.title('Cyberbullying Detection')

# Input text from user
input_text = st.text_area('Enter the message:', '')

# Preprocess the input and make prediction
if st.button('Predict'):
    if input_text:
        # Transform the input text using the vectorizer
        input_vector = vectorizer.transform([input_text])
        
        # Debugging: Print the input vector shape
        st.write(f'Input vector shape: {input_vector.shape}')
        
        # Make prediction using the loaded model
        prediction = model.predict(input_vector)
        
        # Debugging: Print the prediction result
        st.write(f'Raw prediction result: {prediction}')
        
        # Map prediction to category
        predicted_category = categories.get(prediction[0], 'Unknown category')
        
        st.write(f'The message is classified as: **{predicted_category}**')
    else:
        st.write('Please enter a message to classify.')
