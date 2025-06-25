import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, VitsModel
import torch
import tempfile
import soundfile as sf
import time
from groq import Groq
from langsmith import Client
from langsmith.run_helpers import traceable
from langsmith.run_trees import RunTree
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import logging
import uuid

# Configure logging for LangSmith debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langsmith")

# Load environment variables
load_dotenv()

# ========== LangSmith Configuration ==========
def configure_langsmith(tracing_enabled=True, langsmith_api_key=None):
    """Configure LangSmith environment variables."""
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key or os.getenv("LANGSMITH_API_KEY", "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")
    os.environ["LANGSMITH_PROJECT"] = "multi-ia-app"
    os.environ["LANGSMITH_TRACING"] = "true" if tracing_enabled else "false"
    logger.info(f"LangSmith tracing {'enabled' if tracing_enabled else 'disabled'}")

# Initialize LangSmith client with error handling
def initialize_langsmith_client():
    try:
        client = Client()
        logger.info("LangSmith client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith client: {str(e)}")
        st.error(f"Erreur lors de l'initialisation de LangSmith : {str(e)}")
        return None

# Validate API key (only Groq)
def validate_api_key(groq_api_key):
    """Validate Groq API key by attempting to initialize client."""
    try:
        groq_client = Groq(api_key=groq_api_key)
        # Test a simple request to validate Groq API key
        groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        logger.info("Groq API key validated successfully")
        return True, "Clé API Groq valide"
    except Exception as e:
        logger.error(f"Invalid Groq API key: {str(e)}")
        return False, f"Clé API Groq invalide : {str(e)}"

# --------- UTILS for Lottie ---------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --------- LOTTIE ANIMATIONS ---------
main_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
loading_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_p8bfn5to.json")
about_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Multi-IA 🌐✨",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ---------- MODERN CSS (Pastel/Glassmorphism Palette) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');
    * { font-family: 'Quicksand', 'Poppins', sans-serif; }
    body, .main {
        background: linear-gradient(135deg, #fafdff 0%, #dbeafe 60%, #e0e7ff 100%) !important;
        min-height: 100vh;
    }
    .stApp {
        background: linear-gradient(135deg, #fafdff 0%, #dbeafe 60%, #e0e7ff 100%) !important;
    }
    .section-card, .visual-card, .metric-card {
        background: rgba(255,255,255,0.92);
        border-radius: 32px;
        box-shadow: 0 8px 40px 0 #c7d2fe33, 0 1.5px 8px #e0e7ff33;
        padding: 2.2rem 2rem;
        margin-bottom: 2rem;
        transition: box-shadow 0.3s, transform 0.2s;
        backdrop-filter: blur(8px);
    }
    .section-card:hover, .visual-card:hover, .metric-card:hover {
        box-shadow: 0 16px 48px 0 #c7d2fe44, 0 2px 12px #e0e7ff55;
        transform: translateY(-2px) scale(1.01);
    }
    .section-title, .section-title-visual {
        color: #6366f1;
        font-size: 2.3em;
        font-weight: 700;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 8px #e0e7ff44;
        letter-spacing: 1px;
    }
    .badge, .ia-badge {
        display: inline-block;
        background: linear-gradient(90deg, #c7d2fe 0%, #e0e7ff 100%);
        color: #6366f1;
        border-radius: 16px;
        padding: 0.4em 1.2em;
        font-size: 1.05em;
        font-weight: 700;
        margin: 0.2em 0.3em;
        box-shadow: 0 2px 8px #c7d2fe22;
        letter-spacing: 1px;
    }
    .stButton>button, .glow-btn {
        background: linear-gradient(90deg, #c7d2fe 0%, #e0e7ff 100%);
        color: #6366f1;
        border-radius: 32px;
        padding: 1em 2.5em;
        font-size: 1.18em;
        font-weight: 700;
        box-shadow: 0 4px 24px #c7d2fe33, 0 1.5px 8px #e0e7ff33;
        margin-top: 0.5em;
        margin-bottom: 1.2em;
        transition: box-shadow 0.2s, transform 0.2s, background 0.2s;
        cursor: pointer;
    }
    .stButton>button:hover, .glow-btn:hover {
        box-shadow: 0 8px 32px #c7d2fe55, 0 2px 12px #e0e7ff55;
        background: linear-gradient(90deg, #e0e7ff 0%, #c7d2fe 100%);
        color: #3730a3;
        transform: translateY(-2px) scale(1.04);
    }
    .about-avatar {
        border-radius: 50%;
        border: 4px solid #c7d2fe;
        box-shadow: 0 4px 16px #c7d2fe33;
        margin-bottom: 1em;
    }
    .about-contact-btn {
        background: linear-gradient(90deg, #c7d2fe 0%, #e0e7ff 100%);
        color: #6366f1;
        border-radius: 22px;
        padding: 0.6em 1.7em;
        border: none;
        font-weight: 700;
        margin: 0.5em 0.5em 0.5em 0;
        font-size: 1.13em;
        box-shadow: 0 2px 8px #c7d2fe22;
        transition: background 0.2s, color 0.2s;
    }
    .about-contact-btn:hover {
        background: linear-gradient(90deg, #e0e7ff 0%, #c7d2fe 100%);
        color: #3730a3;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background: transparent !important;
        color: #3730a3 !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px #c7d2fe33;
    }
    .stSelectbox > div > div {
        background: transparent !important;
        color: #3730a3 !important;
    }
    .stSelectbox label {
        color: #6366f1 !important;
    }
    .stSelectbox div[role="option"] {
        color: #3730a3 !important;
        background: #f3f4f6 !important;
    }
    .stSelectbox span {
        color: #3730a3 !important;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        box-shadow: 2px 0 18px #c7d2fe22;
        border-radius: 22px;
        backdrop-filter: blur(6px);
    }
    .stProgress > div > div { background-color: #c7d2fe; }
    .stAlert { border-radius: 14px; }
    .section-sep {
        border: none;
        border-top: 2px solid #e0e7ff;
        margin: 2.5em 0 2em 0;
        width: 80%;
    }
    .card-fade {
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s forwards;
        animation-delay: 0.2s;
    }
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: none;
        }
    }
    .css-1d391kg, .css-1d391kg .sidebar-content {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%) !important;
        border-radius: 22px !important;
        box-shadow: 0 4px 24px #c7d2fe22 !important;
    }
    .stSelectbox [data-baseweb="select"] {
        color: #3730a3 !important;
        background: #fafdff !important;
    }
    .stSelectbox [data-baseweb="select"] * {
        color: #3730a3 !important;
        background: #fafdff !important;
    }
    .stSelectbox [data-baseweb="select"] div[role="option"] {
        color: #3730a3 !important;
        background: #f3f4f6 !important;
    }
    .stSelectbox [data-baseweb="select"] span {
        color: #3730a3 !important;
    }
    .stSelectbox label {
        color: #6366f1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR (Lottie + Option Menu) ----------
with st.sidebar:
    st_lottie(main_animation, height=120, key="sidebar_animation")
    selected = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Génération de texte", "Text-to-Speech", "Traduction", "À propos", "Paramètres"],
        icons=['house', 'robot', 'volume-up', 'globe', 'info-circle', 'gear'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background": "linear-gradient(135deg, #fafdff 0%, #dbeafe 60%, #e0e7ff 100%)",
                "box-shadow": "0 4px 24px #c7d2fe22",
                "border-radius": "22px"
            },
            "icon": {"color": "#6366f1", "font-size": "22px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "left",
                "margin": "0px",
                "padding": "12px 18px",
                "color": "#6366f1",
                "background": "transparent",
                "border-radius": "14px",
                "transition": "background 0.2s, color 0.2s",
                "--hover-color": "#e0e7ff",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #e0e7ff 0%, #c7d2fe 100%)",
                "color": "#6366f1",
                "box-shadow": "0 2px 12px #c7d2fe44",
            },
        }
    )

# ---------- API Key Management ----------
def save_api_key(groq_api_key, langsmith_api_key, tracing_enabled=True):
    config_dir = Path(".config")
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    
    config = {
        "groq_api_key": groq_api_key,
        "langsmith_api_key": langsmith_api_key,
        "tracing_enabled": tracing_enabled
    }
    with open(config_file, "w") as f:
        json.dump(config, f)
    configure_langsmith(tracing_enabled, langsmith_api_key)

def load_api_key():
    config_file = Path(".config/config.json")
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
            return config.get("groq_api_key"), config.get("langsmith_api_key", "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"), config.get("tracing_enabled", True)
    return None, "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", True

# ---------- API Key Setup Interface ----------
def show_api_key_setup():
    st.markdown("""
    <style>
        .api-setup {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 1.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            animation: fadeInUp 0.8s ease;
        }
        .api-setup h1 {
            color: #1f3c88;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .api-setup p {
            color: #4a5568;
            text-align: center;
            margin-bottom: 2rem;
        }
        .api-input {
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 0.75rem;
            width: 100%;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        .api-input:focus {
            border-color: #4facfe;
            box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
        }
        .api-button {
            background: linear-gradient(to right, #4facfe, #00f2fe);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.75rem;
            width: 100%;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .api-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .api-info {
            background: #ebf8ff;
            border-radius: 0.75rem;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 0.9rem;
            color: #2b6cb0;
        }
        .api-section {
            background: #f8fafc;
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .api-section h3 {
            color: #2c5282;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="api-setup">
        <h1>🔑 Configuration de l'API Groq</h1>
        <p>Pour utiliser toutes les fonctionnalités de Multi-IA, veuillez configurer votre clé API Groq.</p>
    </div>
    """, unsafe_allow_html=True)

    # Groq API Section
    st.markdown("""
    <div class="api-section">
        <h3>🤖 API Groq</h3>
    </div>
    """, unsafe_allow_html=True)
    
    groq_api_key = st.text_input(
        "Clé API Groq",
        type="password",
        help="Entrez votre clé API Groq pour activer la génération de texte avancée. Obtenez-la sur https://console.groq.com."
    )

    # LangSmith API Section (préremplie)
    st.markdown("""
    <div class="api-section">
        <h3>📊 API LangSmith</h3>
        <p>La clé LangSmith par défaut est utilisée pour le monitoring. Vous pouvez la modifier si nécessaire.</p>
    </div>
    """, unsafe_allow_html=True)
    
    langsmith_api_key = st.text_input(
        "Clé API LangSmith",
        type="password",
        value="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",
        help="Clé API LangSmith pour le monitoring. La clé par défaut est préremplie."
    )

    # Tracing toggle
    tracing_enabled = st.checkbox("Activer le traçage LangSmith", value=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("💾 Sauvegarder", use_container_width=True):
            if groq_api_key:
                # Valider la clé Groq avant de sauvegarder
                is_valid, message = validate_api_key(groq_api_key)
                if is_valid:
                    save_api_key(groq_api_key, langsmith_api_key, tracing_enabled)
                    st.success("Clé API Groq sauvegardée avec succès ! Traçage " + ("activé" if tracing_enabled else "désactivé"))
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Veuillez entrer une clé API Groq")
    
    with col2:
        st.markdown("""
        <div class="api-info">
            <p>🔍 Comment obtenir votre clé API Groq :</p>
            <ol>
                <li>Créez un compte sur <a href="https://console.groq.com" target="_blank">console.groq.com</a></li>
                <li>Accédez à la section "API Keys"</li>
                <li>Générez une nouvelle clé API</li>
            </ol>
            <p>🔍 La clé LangSmith par défaut est utilisée pour le monitoring. Pour une clé personnalisée :</p>
            <ol>
                <li>Créez un compte sur <a href="https://smith.langchain.com" target="_blank">smith.langchain.com</a></li>
                <li>Accédez à vos paramètres</li>
                <li>Générez une nouvelle clé API</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ---------- Model Loading with Caching ----------
@st.cache_resource
def load_tts_model(language_code="eng"):
    try:
        model = VitsModel.from_pretrained(f"facebook/mms-tts-{language_code}")
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{language_code}")
        logger.info(f"TTS model loaded for language: {language_code}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load TTS model: {str(e)}")
        return None, None

@st.cache_resource
def load_models():
    gen = pipeline("text-generation", model="gpt2")
    tts_model, tts_tokenizer = load_tts_model()
    logger.info("Text generation and TTS models loaded")
    return gen, tts_model, tts_tokenizer

text_gen, tts_model, tts_tokenizer = load_models()

# Initialize global clients
groq_client = None
langsmith_client = None

def initialize_clients(groq_api_key, langsmith_api_key):
    global groq_client, langsmith_client
    try:
        groq_client = Groq(api_key=groq_api_key)
        langsmith_client = initialize_langsmith_client()
        logger.info("Groq and LangSmith clients initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize clients: {str(e)}")
        st.error(f"Erreur lors de l'initialisation des clients : {str(e)}")
        return False

# ---------- Test Tracing Function ----------
@traceable(run_type="chain", name="test_tracing", tags=["test"])
def test_tracing_function(input_text):
    """Test function to verify LangSmith tracing."""
    logger.info(f"Testing tracing with input: {input_text}")
    return f"Traçage testé : {input_text}"

# ---------- Enhanced Functions ----------
@traceable(run_type="chain", name="groq_text_generation", tags=["text-generation", "groq"])
def generate_text_with_groq(prompt, length, temperature=1.0):
    try:
        if not groq_client:
            logger.error("Groq client not initialized")
            st.error("Client Groq non initialisé. Veuillez vérifier la configuration.")
            return None
            
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=length
        )
        result = response.choices[0].message.content
        logger.info(f"Text generation completed for prompt: {prompt[:50]}...")
        return result
    except Exception as e:
        logger.error(f"Groq error: {str(e)}")
        st.error(f"Erreur Groq : {str(e)}")
        return None

# Liste des paires de langues supportées
SUPPORTED_LANGUAGE_PAIRS = {
    ("en", "fr"), ("fr", "en"), ("en", "es"), ("es", "en"),
    ("de", "en"), ("en", "de"), ("it", "en"), ("en", "it"),
    ("nl", "en"), ("en", "nl"), ("pt", "en"), ("en", "pt")
}

@traceable(run_type="chain", name="translation", tags=["translation", "helsinki-nlp"])
def translate(text, src_lang, tgt_lang):
    global langsmith_client
    if not langsmith_client:
        logger.error("LangSmith client not initialized")
        st.error("Le client LangSmith n'est pas initialisé. Veuillez configurer vos clés API.")
        return None
    
    if (src_lang, tgt_lang) not in SUPPORTED_LANGUAGE_PAIRS:
        logger.error(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
        st.error(f"La paire de langues {src_lang} -> {tgt_lang} n'est pas supportée.")
        return None
    
    try:
        import sentencepiece
    except ImportError:
        logger.error("sentencepiece is not installed")
        st.error("La bibliothèque 'sentencepiece' est requise pour la traduction. Installez-la avec `pip install sentencepiece`.")
        return None

    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        translator = pipeline("translation", model=model_name)
        result = translator(text)[0]["translation_text"]
        logger.info(f"Translation completed: {src_lang} -> {tgt_lang}")
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        st.error(f"Erreur de traduction : {str(e)}. Vérifiez que le modèle {model_name} est disponible.")
        return None

@traceable(run_type="chain", name="text_to_speech", tags=["tts", "mms-tts"])
def text_to_speech(text, model, tokenizer):
    global langsmith_client
    if not langsmith_client:
        logger.error("LangSmith client not initialized")
        st.error("Le client LangSmith n'est pas initialisé. Veuillez configurer vos clés API.")
        return None
        
    if model is None or tokenizer is None:
        logger.error("TTS model or tokenizer is None")
        st.error("Modèle TTS non chargé correctement.")
        return None
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
        sample_rate = model.config.sampling_rate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, waveform, sample_rate)
            audio_file = open(fp.name, "rb")
            logger.info("Text-to-speech generation completed")
            return audio_file.read()
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        st.error(f"Erreur TTS : {str(e)}")
        return None

# ========== Modern LangSmith Monitoring Badge ==========
def display_langsmith_badge(tracing_enabled):
    status = "ACTIF" if tracing_enabled else "INACTIF"
    color = "#00b894" if tracing_enabled else "#dc2626"
    st.markdown(
        f'''
        <style>
        .ls-badge {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            animation: fadeIn 1.2s;
        }}
        .ls-dot {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            margin-right: 10px;
            box-shadow: 0 0 10px #00f2fe, 0 0 20px #4facfe;
            animation: pulse 1.5s infinite;
        }}
        .ls-text {{
            font-weight: bold;
            color: #1f3c88;
            font-size: 1.1rem;
            letter-spacing: 1px;
            text-shadow: 0 2px 8px #e0f7fa;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 10px #00f2fe, 0 0 20px #4facfe; }}
            50% {{ box-shadow: 0 0 20px #00f2fe, 0 0 40px #4facfe; }}
            100% {{ box-shadow: 0 0 10px #00f2fe, 0 0 20px #4facfe; }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
        <div class="ls-badge">
            <div class="ls-dot"></div>
            <span class="ls-text">LangSmith Monitoring : <span style="color:{color}">{status}</span></span>
        </div>
        ''', unsafe_allow_html=True
    )

# ---------- Main Application ----------
def main():
    global groq_client, langsmith_client
    
    # Load API keys and tracing setting
    groq_api_key, langsmith_api_key, tracing_enabled = load_api_key()
    
    # Vérifier si la clé API Groq est présente et valide
    if not groq_api_key:
        st.warning("Une clé API Groq est requise pour utiliser l'application.")
        show_api_key_setup()
        return
    
    # Valider la clé Groq
    is_valid, message = validate_api_key(groq_api_key)
    if not is_valid:
        st.error(message)
        show_api_key_setup()
        return
    
    configure_langsmith(tracing_enabled, langsmith_api_key)
    
    # Display LangSmith status badge
    display_langsmith_badge(tracing_enabled)

    # Initialize clients with the saved API keys
    if not initialize_clients(groq_api_key, langsmith_api_key):
        st.error("Impossible d'initialiser les clients API. Veuillez vérifier votre clé API Groq.")
        return

    # Set up LangChain environment
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "multi-ia-app"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    # ---------- Main Interface ----------
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="title-animation" style="color: white; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                ✨ Multi-IA : Génération · Voix · Traduction ✨
            </h1>
            <p style="color: white; font-size: 1.2rem; opacity: 0.9;">
                Bienvenue dans un univers intelligent où le texte prend vie ! 🧠🔍💬
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- Dynamic Content ----------
    if selected == "Accueil":
        st.markdown("""
            <div style='
                text-align: center; 
                padding: 2.5rem 1rem 2rem 1rem; 
                background: linear-gradient(135deg, #f8fafc 0%, #e4e8eb 100%);
                border-radius: 22px; 
                margin-bottom: 36px;
                box-shadow: 0 6px 32px 0 rgba(76, 110, 245, 0.08);
            '>
                <h1 class='section-title' style='font-size:2.8em; margin-bottom:0.2em;'>✨ Multi-IA : Génération · Voix · Traduction ✨</h1>
                <p style='color: #2c3e50; font-size: 1.35em; font-weight: 400; margin-bottom:0.8em;'>
                    <i>Bienvenue dans un univers intelligent où le texte prend vie ! 🧠🔊🌍</i>
                </p>
                <hr style='border: none; border-top: 1.5px solid #e4e8eb; width: 60%; margin: 1.5em auto 1.5em auto;'/>
                <p style='color: #4b79a1; font-size: 1.1em; max-width: 700px; margin: auto;'>
                    Cette application met la puissance du machine learning et du NLP au service de la créativité et de la productivité : <b>génère, vocalise, traduis</b> en quelques clics.<br>
                    <span style='color:#283e51;'>Pensée pour les passionnés d'IA, accessible à tous.</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("""
            <div class='section-card' style='margin-bottom:1.5em;'>
                <h3 style='color:#4b79a1; font-size:1.3em;'>🎯 Mission</h3>
                <p style='font-size:1.08em;'>
                    Offrir une plateforme IA tout-en-un pour explorer la génération de texte, la synthèse vocale et la traduction automatique, avec une expérience utilisateur moderne et agréable.
                </p>
            </div>
            <div class='section-card'>
                <h3 style='color:#4b79a1; font-size:1.2em;'>🔬 Technologies</h3>
                <span class='badge'>Groq</span>
                <span class='badge'>Meta MMS-TTS</span>
                <span class='badge'>Helsinki-NLP</span>
                <span class='badge'>LangSmith</span>
                <span class='badge'>Streamlit</span>
                <span class='badge'>Python</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st_lottie(main_animation, height=220, key="main_animation")
            st.markdown("""
            <div style='margin-top: 18px; background: linear-gradient(135deg, #4b79a1 0%, #283e51 100%); color: white; padding: 18px; border-radius: 12px; text-align: center; box-shadow: 0 2px 10px rgba(76,110,245,0.10);'>
                <h2 style='margin:0;'>+1000</h2>
                <p style='margin:0;'>Utilisateurs IA</p>
            </div>
            """, unsafe_allow_html=True)

    elif selected == "Génération de texte":
        st.markdown("""
            <div style='background: rgba(255,255,255,0.75); border-radius: 38px; box-shadow: 0 12px 48px 0 #a18cd144, 0 2px 12px #fbc2eb55; padding: 2.8rem 2.2rem 2.2rem 2.2rem; margin-bottom: 2.5rem; position: relative; max-width: 900px; margin-left:auto; margin-right:auto;'>
                <div style='display:flex; align-items:center; justify-content:center; margin-bottom:1.2em;'>
                    <img src="https://lottie.host/6e7e2e7b-6e7e-4e7e-8e7e-6e7e2e7b6e7e/ai.json" alt="AI" style="height:48px; margin-right:16px;"/>
                    <h1 style='font-size:2.3em; color:#a18cd1; font-weight:800; letter-spacing:1px; margin:0;'>Générateur de texte IA</h1>
                </div>
                <span class='badge'>Groq · Génération</span>
                <hr class='section-sep' style='margin:1.2em 0 2em 0;'/>
            </div>
            <style>
            .modern-input {background:rgba(255,255,255,0.92); border:2px solid #a18cd1; border-radius:20px; padding:0.85rem; font-size:1.08em; transition:all 0.3s;}
            .modern-input:focus {border-color:#fbc2eb; box-shadow:0 0 0 3px #a18cd133;}
            </style>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            prompt = st.text_area("Prompt", placeholder="Ex: Raconte-moi une histoire sur l'IA du futur...", key="modern-prompt", height=100)
        with col2:
            length = st.slider("Longueur du texte", 20, 200, 80, key="slider-longueur")
            temperature = st.slider("Température", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="slider-temperature")
        gen_btn = st.button("✨ Générer le texte", key="modern-gen-btn")
        if gen_btn:
            with st.spinner("Génération en cours..."):
                st_lottie(loading_animation, height=80, key="loading-textgen")
                texte = generate_text_with_groq(prompt, length, temperature)
            if texte:
                st.markdown(f"""
                    <div class='fade-in'> style='background: linear-gradient(120deg, #fafdff 60%, #fbc2eb 100); border-radius: 28px; box-shadow: 0 4px 24px #a18cd122; padding: 2rem 1.5rem; margin-top:1.5em; max-width:767px; margin-left:auto; margin-right:;'>
                        <span class='ia-badge'>🤖 Réponse IA</span>
                        <div style='font-size:1.18em; color:#222; margin: 1em 0 0.5em 0;'>{texte}</div>
                        <button class='copy-btn' onclick="navigator.clipboard.writeText(`{texte}`)">Copier</button>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Erreur lors de la génération du texte.")

    elif selected == "Text-to-Speech":
        with st.container():
            st.subheader("🎤 Synthèse vocale", anchor="synthese-vocale")
            st.markdown("""
            <div class="block">
                <p>Convertis n'importe quel texte en voix naturelle avec notre modèle TTS avancé.</p>
            </div>
            """, unsafe_allow_html=True)
            
            realtime_option = st.checkbox("Mode progressif", value=True)
            
            text_input = st.text_area("Texte à vocaliser :")
            language = st.selectbox("Langue", ["eng", "fra", "spa", "deu", "ita"], 
                                  format_func=lambda x: {
                                      "eng": "Anglais", "fra": "Français", 
                                      "spa": "Espagnol", "deu": "Allemand", 
                                      "ita": "Italien"
                                  }[x])
            
            if st.button("🔊 Écouter"):
                with st.spinner("Génération audio en cours..."):
                    model, tokenizer = load_tts_model(language)
                    if model:
                        if realtime_option:
                            text_placeholder = st.empty()
                            audio_placeholder = st.empty()
                            
                            words = text_input.split()
                            displayed_text = ""
                            
                            audio = text_to_speech(text_input, model, tokenizer)
                            
                            st.markdown(
                                f"<div class='generated-text'>{text_input}</div>", 
                                unsafe_allow_html=True
                            )
                            st.audio(audio, format="audio/wav")
                            else:
                                st.error("Erreur dans la synthèse vocale.")
                        else:
                            audio = text_to_speech(text_input, model, tokenizer)
                            if audio:
                                st.markdown(
                                    f"<div class='generated-text'>{text_input}</div>", 
                                    unsafe_allow_html=True
                                )
                                st.audio(audio, format="audio/wav")
                            else:
                                st.error("Erreur dans la synthèse vocale.")
                    else:
                        st.error("Modèle TTS non chargé correctement.")

    elif selected == "Traduction":
        with st.container():
            st.subheader("🌍 Traduction automatique", anchor="traduction-automatique")
            st.markdown("""
            <div class="block">
                <p>Traduis instantanément entre plusieurs langues avec notre modèle de traduction avancé.</p>
            </div>
            """, unsafe_allow_html=True)
            
            langues = {
                "allemand (de)": "de",
                "espagnol (es)": "es",
                "français (fr)": "fr",
                "anglais (en)": "en",
                "italien (it)": "it",
                "néerlandais (nl)": "nl",
                "portugais (pt)": "pt",
            }
            
            col1, col2 = st.columns(2)
            with col1:
                src = st.selectbox("Langue source :", list(langues.keys()), index=0)
            with col2:
                tgt = st.selectbox("Langue cible :", list(langues.keys()), index=1)
            
            if langues[src] == langues[tgt]:
                st.error("La langue source et la langue cible doivent être différentes.")
            elif (langues[src], langues[tgt]) not in SUPPORTED_LANGUAGE_PAIRS:
                st.warning(f"La paire de langues {src} -> {tgt} n'est pas supportée.")
            else:
                texte_input = st.text_area("Texte à traduire :", 
                                        placeholder="Exemple : Wie schön ist das Wetter heute ?")
                
                if st.button("📤 Traduire"):
                    with st.spinner("Traduction en cours..."):
                        result = translate(texte_input, langues[src], langues[tgt])
                        if result:
                            st.markdown(f"<div class='block'><p><strong>{src.capitalize()}</strong> : {texte_input}</p></div>", 
                                    unsafe_allow_html=True)
                            st.markdown(f"<div class='block'><p><strong>{tgt.capitalize()}</strong> : <span class='translated-text'>{result}</span></p></div>", 
                                    unsafe_allow_html=True)
                        else:
                            st.error("Erreur lors de la traduction ou modèle non disponible.")

    elif selected == "À propos":
        st.markdown("""
            <div style='background: linear-gradient(120deg, #4b79a1 0%, #283e51 100%); padding: 2rem; border-radius: 18px; color: white; margin-bottom: 2rem; text-align: center;'>
                <h1 class='section-title' style='color:white;'>À propos</h1>
                <p>Découvrez le créateur, le projet et les technologies utilisées</p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st_lottie(about_animation, height=220, key="about_animation")
            st.image(
                "https://avatars.githubusercontent.com/u/TheBeyonder237",
                width=180,
                caption="Ngoue David",
                output_format="auto",
                use_container_width=False,
                channels="RGB"
            )
            st.markdown("""
                <div style='text-align:center; margin-top:1em;'>
                    <button class='about-contact-btn' onclick="window.open('mailto:ngouedavidrogeryannick@gmail.com')">📧 Email</button>
                    <button class='about-contact-btn' onclick="window.open('https://github.com/TheBeyonder237')">🌐 GitHub</button>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='section-card'>
                <h2 class='section-title'>Qui suis-je ?</h2>
                <p>
                    Je suis un passionné de l'intelligence artificielle et de la donnée.<br>
                    Actuellement en Master 2 en IA et Big Data, je travaille sur des solutions innovantes dans le domaine de l'Intelligence Artificielle appliquée à la finance et à la santé.
                </p>
                <h3 style='color:#4b79a1;'>Compétences</h3>
                <span class='badge'>Python</span>
                <span class='badge'>Machine Learning</span>
                <span class='badge'>Deep Learning</span>
                <span class='badge'>NLP</span>
                <span class='badge'>Data Science</span>
                <span class='badge'>Cloud Computing</span>
                <span class='badge'>Streamlit</span>
                <span class='badge'>Scikit-learn</span>
                <span class='badge'>XGBoost</span>
                <span class='badge'>Pandas</span>
                <span class='badge'>Plotly</span>
                <span class='badge'>SQL</span>
                <h3 style='color:#4b79a1; margin-top:1.5em;'>Projets récents</h3>
                <ul>
                    <li><b>💳 Credit Card Expenditure Predictor</b> : Application de prédiction de dépenses de carte de crédit.</li>
                    <li><b>🫀 HeartGuard AI</b> : Prédiction de risques cardiaques par IA.</li>
                    <li><b>🔍 Multi-IA</b> : Plateforme multi-modèles pour la génération de texte, synthèse vocale et traduction.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; color: #666; padding: 20px;'>
            Développé avec ❤️ par Ngoue David
            </div>
        """, unsafe_allow_html=True)

    elif selected == "Paramètres":
        st.markdown("""
            <div class='section-card'>
                <h1 class='section-title'>⚙️ Paramètres</h1>
                <p>Configurez les options de l'application, y compris le traçage LangSmith.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # API Key settings
        st.markdown("### Configuration de la clé API Groq")
        if st.button("Modifier la clé API Groq"):
            show_api_key_setup()
        
        # Tracing settings
        st.markdown("### Configuration du traçage")
        new_tracing_enabled = st.checkbox("Activer le traçage LangSmith", value=tracing_enabled)
        if new_tracing_enabled != tracing_enabled:
            groq_api_key, langsmith_api_key, _ = load_api_key()
            save_api_key(groq_api_key, langsmith_api_key, new_tracing_enabled)
            st.success(f"Traçage LangSmith {'activé' if new_tracing_enabled else 'désactivé'}")
            st.rerun()

        # Test tracing
        st.markdown("### Tester le traçage")
        test_input = st.text_input("Entrez un texte pour tester le traçage", key="test-tracing-input")
        if st.button("Tester"):
            result = test_tracing_function(test_input)
            st.write(result)
            st.info("Vérifiez votre projet LangSmith (multi-ia-app) pour voir la trace !")

if __name__ == "__main__":
    main()