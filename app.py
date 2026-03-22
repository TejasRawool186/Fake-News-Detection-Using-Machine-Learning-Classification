import streamlit as st
import joblib
import plotly.graph_objects as go
import numpy as np
import re
from newspaper import Article


from newspaper import Article

def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None
    
    
st.set_page_config(page_title="AI Fake News Detection", layout="wide")

# ================= THEME =================
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: radial-gradient(circle at top,#020617,#020617);
color:white;
}

.sidebar .sidebar-content{
background: #020617;
}

.ai-card{
background: linear-gradient(145deg,#020617,#0f172a);
padding:25px;
border-radius:16px;
box-shadow: 0 20px 40px rgba(0,0,0,0.6);
text-align:center;
transition:0.3s;
}

.ai-card:hover{
transform: translateY(-5px);
box-shadow: 0 30px 60px rgba(0,0,0,0.8);
}

.metric{
font-size:42px;
font-weight:700;
}

.sub{
color:#94a3b8;
font-size:14px;
}

.title{
font-size:56px;
font-weight:800;
text-align:center;
background: linear-gradient(90deg,#3B82F6,#8B5CF6,#06B6D4);
-webkit-background-clip:text;
color:transparent;
}

.subtitle{
text-align:center;
color:#94a3b8;
font-size:18px;
}

.side-card{
padding:16px;
border-radius:12px;
margin-bottom:14px;
color:white;
box-shadow: 0 6px 20px rgba(0,0,0,0.5);
}

.svm{background:linear-gradient(135deg,#22c55e,#16a34a);}
.lr{background:linear-gradient(135deg,#3b82f6,#2563eb);}
.nb{background:linear-gradient(135deg,#f59e0b,#d97706);}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='title'>Fake News AI Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Multi-Model AI Verification System</div>", unsafe_allow_html=True)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.markdown("## 📊 Model Performance")


st.sidebar.markdown("<div class='side-card lr'>Logistic Accuracy 98.3%</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='side-card nb'>Naive Bayes 93.7%</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='side-card svm'>SVM Accuracy 99.5%</div>", unsafe_allow_html=True)

# ================= LOAD MODELS =================
lr = joblib.load("logistic_regression_model.pkl")
nb = joblib.load("naive_bayes_model.pkl")
svm = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

## ================= INPUT =================

st.markdown("""
<style>
.source-select div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #374151 !important;
    min-height: 38px !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

col_title, col_source = st.columns([0.75, 0.25])

with col_title:
    st.markdown("### 📝 Enter News Article")

with col_source:
    source = st.selectbox(
        "",
        ["User Input", "Social Media", "News Website", "Blog", "Government", "Unknown"],
        key="source",
        label_visibility="collapsed"
    )


news = st.text_area("", height=220)

def get_url(news):
    url_pattern = r'(https?://\S+)'
    match = re.search(url_pattern, news)
    return match.group(0) if match else None


def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

# logic
def source_modifier(source):
    if source == "Social Media":
        return -7
    elif source == "Blog":
        return -4
    elif source == "News Website":
        return +2
    elif source == "Government":
        return +5
    else:
        return 0
    
#
if st.button("🚀 Analyze AI Prediction"):

    url = get_url(news)

    if url:
        st.info("🔗 URL detected — extracting article...")

        extracted = get_article_text(url)

        if extracted is None:
            st.error("❌ Unable to extract article text")
            st.stop()

        else :
             st.success("✅ Article fetched successfully...")

        news_text = extracted
    else:
        news_text = news
        
    modifier = source_modifier(source)

    vec = vectorizer.transform([news_text])

    lr_prob = lr.predict_proba(vec)[0]
    nb_prob = nb.predict_proba(vec)[0]

    try:
        svm_prob = svm.predict_proba(vec)[0]
    except:
        score = svm.decision_function(vec)[0]
        prob_real = 1 / (1 + np.exp(-score))
        svm_prob = [1 - prob_real, prob_real]
        
    lr_pred = lr.predict(vec)[0]
    nb_pred = nb.predict(vec)[0]
    svm_pred = svm.predict(vec)[0]

    lr_conf = lr_prob[1] * 100
    nb_conf = nb_prob[1] * 100
    svm_conf = svm_prob[1] * 100

    # adjust confidence
    modifier = source_modifier(source)

    lr_conf_adj = max(0, min(100, lr_conf + modifier))
    nb_conf_adj = max(0, min(100, nb_conf + modifier))
    svm_conf_adj = max(0, min(100, svm_conf + modifier))

    # ================= CARDS =================
    st.markdown("## 🤖 Model Intelligence")

    def card(title, pred, conf):
        color = "#22c55e" if pred==1 else "#ef4444"
        label = "REAL" if pred==1 else "FAKE"
        st.markdown(f"""
        <div class='ai-card'>
            <div class='sub'>{title}</div>
            <div class='metric' style='color:{color}'>{label}</div>
            <div class='sub'>Confidence {conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: card("Logistic Regression",lr_pred, lr_conf_adj)
    with c2: card("Naive Bayes",nb_pred, nb_conf_adj)
    with c3: card("SVM",svm_pred, svm_conf_adj)

    # ================= ENSEMBLE =================
    votes = [lr_pred,nb_pred,svm_pred]
    final = max(set(votes), key=votes.count)

    consensus = (max(votes.count(0),votes.count(1)) / 3)*100

    st.markdown("---")

    # ================= LAYOUT =================
    col1,col2 = st.columns([0.45,0.55])

    with col1:

        st.markdown("### 🧠 AI Consensus Strength")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=consensus,
            number={'suffix':"%",'font':{'size':36}},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"white"},
                'steps':[
                    {'range':[0,40],'color':"#ef4444"},
                    {'range':[40,70],'color':"#f59e0b"},
                    {'range':[70,100],'color':"#22c55e"},
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color':"white"},
            height=280
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.markdown("### 🎯 Final AI Verdict")

        if final==1:
            st.success("REAL NEWS — AI Ensemble Consensus")
        else:
            st.error("FAKE NEWS — AI Ensemble Consensus")

        st.info(f"{votes.count(1)} Models voted REAL, {votes.count(0)} voted FAKE")

# ================= FOOTER =================
st.markdown("---")
st.caption("AI Fake News Detection System • Research Grade Interface")
