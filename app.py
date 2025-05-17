
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("your_model.h5")
    return model

model = load_model()
class_names = ['Healthy', 'Yellow_Leaf_Disease', 'Bud_Rot', 'Fruit_Rot', 'Stem_Borer', 'Koleroga']

remedies = {
    "Healthy": {
        "en": "Plant is healthy.",
        "kn": "ಸಸ್ಯವು ಆರೋಗ್ಯವಾಗಿರುತ್ತದೆ.",
        "hi": "पौधा स्वस्थ है।"
    },
    "Yellow_Leaf_Disease": {
        "en": "Apply balanced fertilizers rich in magnesium.",
        "kn": "ಮ್ಯಾಗ್ನೀಶಿಯಮ್ ಸಮೃದ್ಧ ಸಮತೋಲಿತ ರಸಗೊಬ್ಬರ ಹಾಕಿ.",
        "hi": "मैग्नीशियम युक्त संतुलित उर्वरक का प्रयोग करें।"
    },
    "Bud_Rot": {
        "en": "Remove infected parts and spray Bordeaux mixture.",
        "kn": "ಸಂಕ್ರಮಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದು ಹಾಕಿ ಮತ್ತು ಬೋರ್ಡೊ ಮಿಶ್ರಣ ಸಿಂಪಡಿಸಿ.",
        "hi": "संक्रमित भागों को हटाएं और बोर्डो मिश्रण छिड़कें।"
    },
    "Fruit_Rot": {
        "en": "Ensure proper drainage and use fungicide.",
        "kn": "ಸರಿಯಾದ ನೀರು ನಿಕಾಸಿಗೆ ಖಾತರಿಪಡಿಸಿ ಮತ್ತು ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ.",
        "hi": "सही जल निकासी सुनिश्चित करें और फफूंदनाशी का उपयोग करें।"
    },
    "Stem_Borer": {
        "en": "Use insecticides like Chlorpyrifos at the base.",
        "kn": "ಕಣಿವೆ ಬಳಿ ಕ್ಲೋರಪೈರಿಫೋಸ್ ಹತ್ತಿರದ ಕೀಟನಾಶಕವನ್ನು ಬಳಸಿ.",
        "hi": "नीचे क्लोरपायरीफॉस जैसे कीटनाशक का उपयोग करें।"
    },
    "Koleroga": {
        "en": "Spray 1% Bordeaux mixture during monsoon.",
        "kn": "ಮಳೆಯ ಸಮಯದಲ್ಲಿ 1% ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಸಿಂಪಡಿಸಿ.",
        "hi": "मानसून के दौरान 1% बोर्डो मिश्रण छिड़कें।"
    }
}

st.title("Krishi Doctor – Crop Disease Detection")
lang = st.selectbox("Choose Language", ["English", "Kannada", "Hindi"])
lang_map = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

uploaded_file = st.file_uploader("Upload an image of the leaf or fruit", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]

    st.subheader(f"Predicted: {predicted_class.replace('_', ' ')}")
    st.text(remedies[predicted_class][lang_map[lang]])
