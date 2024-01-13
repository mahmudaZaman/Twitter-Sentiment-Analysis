import streamlit as st
import pickle
import time
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub

model = tf.keras.models.load_model(
       ('/Users/shuchi/Documents/work/personal/python/sentiment_analysis/out/tweet_model.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

st.title('Twitter Sentiment Analysis')

tweet = st.text_input('Enter your tweet')

submit = st.button('Predict')

if submit:
    prediction = model.predict([tweet])
    if prediction[0] > 0.5:
        st.write("Disaster Tweet")
    else:
        st.write("Not Disaster Tweet")
    print(prediction[0])
    st.write(prediction[0])
