import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from src.models.train_model import run_train_pipeline


def streamlit_run():
    model = tf.keras.models.load_model(
           ('/Users/shuchi/Documents/work/personal/Twitter-Sentiment-Analysis/out/tweet_model.h5'),
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

def model_run():
    run_train_pipeline()


if __name__ == '__main__':
    mode = os.getenv("mode", "streamlit")
    print("mode", mode)
    if mode == "model":
        model_run()
    else:
        streamlit_run()
