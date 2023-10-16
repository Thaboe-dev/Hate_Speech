import streamlit as st
import numpy as np
import pickle 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from googletrans import Translator

# Load the pretrained model and tokenizer
model = load_model(r"C:\Users\LENOVO\Documents\NLP\hate_speech_model.h5")  
tokenizer = pickle.load(open(r"C:\Users\LENOVO\Documents\NLP\hate_speech_tokenizer.pkl", 'rb'))


# Define the class labels mapping
label_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}

# Create a Translator object
translator = Translator()

def translate_to_english(text, source_language):
    try:
        translation = translator.translate(text, src=source_language, dest='en')
        translated_text = translation.text if translation is not None else "Translation failed"
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        translated_text = "Translation failed"

    return translated_text

def predict_class(model, tokenizer, text):
    # Tokenize and pad the input text
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=50, padding='post', truncating='post')

    # Make predictions
    predictions = model.predict(text_pad)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map predicted class to the corresponding label
    class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    predicted_label = class_labels[predicted_class]

    return predicted_label

# Streamlit App
st.title("Hate Speech Detection App")

# Multilingual Support
selected_language = st.selectbox("Select Language", ["Shona","Ndebele","English", "French", "Spanish", "German"])

# Input text
text_input = st.text_area("Enter text for hate speech detection:")

# Translate text to English
if text_input.strip() and selected_language != "English":
    text_input = translate_to_english(text_input, selected_language)

# Button to make predictions
if st.button("Predict"):
    # Get the prediction
    prediction = predict_class(model,tokenizer,text_input)

    # Display the prediction
    st.write(text_input)
    st.success(f"Prediction: {prediction}")

