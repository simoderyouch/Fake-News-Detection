import streamlit as st
from transformers import pipeline
import requests
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

############ SETTING UP THE PAGE LAYOUT AND TITLE ############

# `st.set_page_config` is used to display the default layout width, the title of the app, and the emoticon in the browser tab.

st.set_page_config(layout="centered", page_title="FND Fake News")

############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.
c1, c2 = st.columns([0.32, 2])

# The snowflake logo will be displayed in the first column, on the left.
with c1:
    st.caption("")
    st.title("📑")

# The heading will be on the right.
with c2:
    st.caption("")
    st.title("FND Detection")

# We need to set up session state via st.session_state so that app interactions don't reset the app.
if "valid_inputs_received" not in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ SIDEBAR CONTENT ############

st.sidebar.subheader("Model Options")
st.sidebar.write("")
SELECTED_MODEL = st.sidebar.selectbox("Choose a model", ("XLM-RoBERTa", "LSTM", "Logistic Regression", "SVM", "Random Forest"))

LANGUAGE = st.sidebar.selectbox("Language", ("en", "ar"), format_func=lambda x: "English" if x == "en" else "Arabic")

# Model selection

if SELECTED_MODEL:
    st.session_state.valid_inputs_received = False

MODEL_INFO = {
    "XLM-RoBERTa": """
    This model is trained on over 400,000 Arabic news articles and 70,000 English news articles from different media sources.
    It's based on the 'xlm-roberta-base' architecture and can process text up to 512 tokens.
    """,
    "LSTM": """
    Bidirectional LSTM model trained with pre-trained word embeddings:
    - For English: Google's Word2Vec (GoogleNews-vectors-negative300.bin)
    - For Arabic: FastText embeddings (cc.ar.300.bin)
    """,
    "Logistic Regression": """
    Traditional machine learning model using TF-IDF features.
    Good baseline with interpretable results.
    """,
    "SVM": """
    Support Vector Machine classifier using TF-IDF features.
    Effective for text classification tasks with high-dimensional data.
    """,
    "Random Forest": """
    Ensemble learning method using multiple decision trees.
    Robust against overfitting with good generalization.
    """,
    None: "NO MODEL SELECTED",
}


model_info_container = st.sidebar.container(border=True)
model_info_container.markdown(MODEL_INFO[SELECTED_MODEL])


copyright_container = st.sidebar.container(border=True)
copyright_container.markdown("Copyright ©️ 2024 [Mohamed Ed Deryouch](https://huggingface.co/edderyouch)")


############ TABBED NAVIGATION ############


MainTab, InfoTab = st.tabs(["Main", "Info"])

############ API VERIFICATION FUNCTIONS ############
FACT_CHECK_API_KEY = "your key please"

def verify_with_news_api(query):
    if not FACT_CHECK_API_KEY:
        return None
    
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        claims = response.json().get("claims", [])
        return any(claim.get("claimReview", []) for claim in claims)
    return False

############ MODEL FUNCTION ############
@st.cache_resource
def load_tokenizer(language):
    # Load the appropriate tokenizer for the selected language
    if language == "en":
        # Load English tokenizer
        tokenizer_path = "./Models/English/tokenizer.pkl"
    else:
        # Load Arabic tokenizer
        tokenizer_path = "./Models/Arabic/tokenizer.pkl"
    
    try:
        return joblib.load(tokenizer_path)
    except:
        # Fallback to creating a new tokenizer if saved one isn't available
        return Tokenizer()

@st.cache_resource
def load_lstm_model(language):
    model_path = f"./Models/{'English' if language == 'en' else 'Arabic'}/lstm.h5"
    return load_model(model_path)

@st.cache_resource
def load_transformer_model(language):
    model_path = f"./Models/{'English' if language == 'en' else 'Arabic'}/prerainmodel"
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

@st.cache_resource
def load_ml_model(model_name, language):
    model_path = f"./Models/{'English' if language == 'en' else 'Arabic'}/{model_name}_model.pkl"
    return joblib.load(model_path)

@st.cache_resource
def load_vectorizer(language):
    vectorizer_path = f"./Models/{'English' if language == 'en' else 'Arabic'}/tfidf_vectorizer.pkl"
    try:
        return joblib.load(vectorizer_path)
    except:
        # Fallback to creating a new vectorizer if saved one isn't available
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer()

def MODEL_RESULT(model_name, news, language):
    if model_name == "XLM-RoBERTa":
        classifier = load_transformer_model(language)
        result = classifier(news)
        
        if result[0]["label"] == "LABEL_1":
            return "REAL NEWS"
        else:
            return "FAKE NEWS"
    
    elif model_name == "LSTM":
        model = load_lstm_model(language)
        tokenizer = load_tokenizer(language)
        
        # Process text for LSTM
        sequence = tokenizer.texts_to_sequences([news])
        padded = pad_sequences(sequence, maxlen=300)
        prediction = model.predict(padded)[0][0]
        
        return "REAL NEWS" if prediction > 0.5 else "FAKE NEWS"
    
    else:  # Traditional ML models
        model = load_ml_model(model_name, language)
        vectorizer = load_vectorizer(language)
        
        # For ML models, we need to vectorize the text
        try:
            # Try to use the pre-trained vectorizer
            X = vectorizer.transform([news])
        except:
            # Fallback if the vectorizer hasn't been fit
            vectorizer.fit([news])
            X = vectorizer.transform([news])
        
        prediction = model.predict(X)[0]
        return "REAL NEWS" if prediction == 1 else "FAKE NEWS"

############ MAIN TAB CONTENT ############
with MainTab:
    st.write("")
    st.markdown("Classify news as real or fake using the selected model.")
    st.write("")
    
    container = st.container(border=True)
    container.write(f"Selected model: {SELECTED_MODEL}")
    container.write(f"Selected language: {'English' if LANGUAGE == 'en' else 'Arabic'}")

    with st.form(key="main_form"):
        # Predefined news examples
        pre_defined_news = {
            "en": "SCIENTISTS DISCOVER TALKING ELEPHANTS: Researchers at Harvard University claim to have taught elephants to speak English.",
            "ar": "هبطت مركبة ناسا الجوالة 'بيرسيفيرانس' بنجاح على سطح المريخ"
        }
        
        news = st.text_area(
            "Enter news to classify",
            pre_defined_news[LANGUAGE],
            height=200,
            help="Please provide the news that you need to verify for its truthfulness.",
            key="news",
        )

        submit_button = st.form_submit_button(label="Analyze News")

    if submit_button and not news.strip():
        st.warning("📑 Please enter some news text to analyze")
        st.stop()
    elif submit_button or st.session_state.valid_inputs_received:
        if submit_button:
            st.session_state.valid_inputs_received = True
        
        # Get model prediction - using the appropriate language model directly
        with st.spinner("Analyzing..."):
            prediction = MODEL_RESULT(SELECTED_MODEL, news, LANGUAGE)
            
            # Optional: Verify with external API
            query = " ".join(news.split()[:5])  # Use first 5 words as query
            api_result = verify_with_news_api(query)

        # Display results
        st.subheader("Result")
        if prediction == "REAL NEWS":
            st.success("✅ Model Prediction: REAL NEWS")
        else:
            st.error("❌ Model Prediction: FAKE NEWS")
            
        # Display additional verification if available
        if api_result is not None:
            st.subheader("External Verification")
            if api_result:
                st.info("ℹ️ This claim has been fact-checked by external sources")
            else:
                st.info("ℹ️ No external fact-checks found for this claim")

############ INFO TAB CONTENT ############
with InfoTab:
    st.header("About This Project")
    st.markdown("""
    ## Multilingual Fake News Detection System
    
    This application uses machine learning and deep learning models to detect fake news in both English and Arabic.
    
    ### Models Available:
    
    1. **XLM-RoBERTa**: A transformer-based model that supports cross-lingual transfer learning
    2. **LSTM**: Deep learning model using word embeddings
    3. **Traditional ML Models**: Logistic Regression, SVM, and Random Forest
    
    ### How It Works:
    
    1. Enter news text in English or Arabic
    2. Select your preferred model
    3. The system will analyze the text and classify it as real or fake
    4. Each language uses its own dedicated model
    
    ### Dataset:
    
    The models were trained on datasets containing:
    - Over 70,000 English news articles
    - Over 400,000 Arabic news articles
    
    ### Accuracy:
    
    Model performance varies by language, with accuracy ranging from 85% to 95% depending on the model and language.
    """)


