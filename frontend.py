import streamlit as st
import requests

st.set_page_config(page_title="NER System", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

examples = [
    "Barack Obama visited London for the G8 Summit meeting.",
    "Apple Inc released the new iPhone in San Francisco.",
    "The European Union and United Nations issued a joint climate report.",
    "Manchester United defeated Arsenal at Old Trafford Stadium.",
    "Angela Merkel met Emmanuel Macron in Berlin.",
    "NASA launched the Artemis mission from Cape Canaveral in Florida."
]

if "text_input" not in st.session_state:
    st.session_state.text_input = examples[0]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F7F1E8 0%, #EFE2D0 100%);
}

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #3D2B1F;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6F5B47;
    margin-bottom: 28px;
}

[data-testid="stMetric"] {
    background-color: #FFFDF8;
    border: 1px solid #E2CDAF;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(92, 64, 51, 0.08);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #3D2B1F !important;
}

.stTextArea textarea {
    background-color: #FFFDF8 !important;
    color: #2B1D14 !important;
    border: 2px solid #D1B894 !important;
    border-radius: 16px !important;
    font-size: 16px !important;
}

.stButton button {
    background: linear-gradient(135deg, #B08D57, #8B6B43) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    height: 50px;
    font-size: 16px;
    font-weight: 700;
}

.stButton button:hover {
    background: linear-gradient(135deg, #8B6B43, #6B4E31) !important;
}

.card {
    background-color: #FFFDF8;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #E2CDAF;
    box-shadow: 0px 10px 30px rgba(92, 64, 51, 0.10);
}

.entity {
    background-color: #E8D8C3;
    color: #3D2B1F;
    padding: 10px 16px;
    margin: 6px;
    border-radius: 20px;
    display: inline-block;
    font-weight: 700;
    border: 1px solid #D1B894;
}

h1, h2, h3, h4, label, p, li, span {
    color: #3D2B1F !important;
}

.about-box {
    background-color: #FFFDF8;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #B08D57;
    box-shadow: 0px 8px 22px rgba(92, 64, 51, 0.08);
    color: #3D2B1F;
}
            
            /* Dropdown */
div[data-baseweb="select"] > div {
    background-color: #FFFDF8 !important;
    border: 2px solid #D1B894 !important;
    border-radius: 12px !important;
    color: #3D2B1F !important;
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: #3D2B1F !important;
    font-weight: 600;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #FFFDF8 !important;
    border: 1px solid #D1B894 !important;
}

/* Dropdown options */
li[role="option"] {
    background-color: #FFFDF8 !important;
    color: #3D2B1F !important;
}

/* Hover effect */
li[role="option"]:hover {
    background-color: #E8D8C3 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI-Powered Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Extract people, organizations, locations, and entities using Transformer-based NLP</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Model", "DistilBERT")
col2.metric("Dataset", "CoNLL-2003")
col3.metric("Task", "NER")

st.divider()

col_left, col_right = st.columns([1.35, 1])

with col_left:
    st.subheader("Type a Sentence")

    text = st.text_area(
        "Enter text for entity detection:",
        key="text_input",
        height=170
    )

    c1, c2 = st.columns(2)

    with c1:
        clear = st.button("Clear", use_container_width=True)

    with c2:
        analyze = st.button("Submit", use_container_width=True)

    if clear:
        st.session_state.clear()
        st.rerun()

    st.subheader("☰ Examples")

    selected_example = st.selectbox(
        "Choose a sample sentence:",
        examples
    )

    use_example = st.button("Use Selected Example", use_container_width=True)

    if use_example:
        st.session_state.clear()
        st.session_state.text_input = selected_example
        st.rerun()

with col_right:
    st.subheader("About This System")
    st.markdown("""
    <div class="about-box">
        This application uses a fine-tuned Transformer model for Named Entity Recognition.
        The backend is built with FastAPI and the frontend is built with Streamlit.
        <br><br>
        <b>Entity Types</b>
        <ul>
            <li>PERSON</li>
            <li>LOCATION</li>
            <li>ORGANIZATION</li>
            <li>MISC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if analyze:
    text = st.session_state.text_input

    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing text..."):
            response = requests.post(API_URL, json={"text": text})

        if response.status_code == 200:
            entities = response.json()["entities"]

            st.divider()
            st.subheader("Detected Entities")

            if entities:
                st.success(f"{len(entities)} entities detected")

                html = '<div class="card">'
                for entity in entities:
                    html += f'<span class="entity">{entity["word"]} → {entity["label"]}</span>'
                html += '</div>'

                st.markdown(html, unsafe_allow_html=True)

                st.subheader("Entity Table")
                st.table(entities)
            else:
                st.info("No named entities found.")
        else:
            st.error("Backend API error. Make sure FastAPI is running.")