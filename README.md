# 🧠 Text Emotion Detection

A machine learning project that classifies input text into emotional categories like *Happy, Sad, Angry, Fear, Surprise*, etc., using the **EmotionDataset** and a web interface built with Streamlit.

---

## 📂 Dataset Used

- **EmotionDataset**  
  A labeled dataset with examples of short texts (like tweets) mapped to emotions.
  - Columns: `Emotion`, `Text`
  - Source: Public NLP dataset often used for emotion classification tasks.

---

## ⚙️ Approach Summary

- **Text Preprocessing**:
  - Removed user handles and stopwords using `neattext`.
- **Feature Extraction**:
  - Used `CountVectorizer` to convert text into a bag-of-words representation.
- **Model Training**:
  - Trained a **Logistic Regression** classifier using a scikit-learn `Pipeline`.
  - Also experimented with SVM and Random Forest (not used in final model).
- **Evaluation**:
  - Accuracy score calculated on the test set.
  - Confusion matrix plotted using `seaborn` and displayed in the frontend.
- **Web UI**:
  - Built using **Streamlit**.
  - User can input text and get predicted emotion + probability progress bar.

---

## 📦 Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

##🚀 Run the App
```bash
streamlit run app.py
```


## Output Example

``` Input
 "I'm very excited for the concert!"
```

```Output
 Emotion: joy 😂
```

With probabiliy over: Progress bar showing model certainty (e.g., 92.6%)
