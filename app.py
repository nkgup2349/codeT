import streamlit as st

import pandas as pd
import numpy as np
import time
import joblib

pipe_lr = joblib.load(open("model/result.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:

        prediction = predict_emotions(raw_text)
        prob = get_prediction_proba(raw_text)

        st.success("Prediction")
        emoji_icon = emotions_emoji_dict[prediction]
        st.write("{}:{}".format(prediction, emoji_icon))

        probability = np.max(prob) * 100  # still a NumPy float array
        probability = float(probability)  # now it's a normal Python float

        progress_bar = st.progress(0)
        for percent_complete in range(int(probability) + 1):
            time.sleep(0.015)
            progress_bar.progress(percent_complete)

        st.write(f"With Probability over: {probability:.2f}%")

        st.subheader("Confusion Matrix")
        st.image("model/output.png", caption="Confusion Matrix (on Test Data)")





if __name__ == '__main__':
    main()
