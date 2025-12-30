## app.py    FAQ Generateur
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib
import pickle
import re
from pathlib import Path
from groq import Groq
import warnings
warnings.filterwarnings("ignore")

## Configuration page
st.set_page_config(page_title="FAQ Pro", page_icon="🚀", layout="wide")

## CSS
st.markdown("""
<style>
.footer-pro {
    #position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    background: #000000 !important;
    color: #ffffff !important;
    text-align: center;
    padding: 20px 0;
    font-size: 1.1em;
    font-weight: 500;
    z-index: 1000;
}
            
.footer-pro p {
    margin: 0;
    color: #ffffff !important;
}
            
</style>
""", unsafe_allow_html=True)

st.title(" GENERATEUR FAQ")
st.markdown("***Vos données → FAQ professionnelle automatiquement***")

## SIDEBAR - GUIDE THÈMES
st.sidebar.title("⚙️ **Paramètres**")
n_clusters = st.sidebar.slider(" **Thèmes max**", 3, 30, 10)
max_questions = st.sidebar.slider(" **Questions/thèmes**", 3, 7, 4)

## GUIDE THÈMES
st.sidebar.subheader(" **Guide Thèmes**")
use_guide_themes = st.sidebar.checkbox("Filtrer mes thèmes", value=False)
guide_themes_input = []
if use_guide_themes:
    guide_input = st.sidebar.text_area(
        "**Vos thèmes prioritaires** (1/ligne)",
        placeholder="""Paiements en ligne
Livraison & Retours
Support Client
Produits & Services""",
        height=150,
        help="Seulement ces thèmes apparaîtront dans la FAQ !"
    )
    guide_themes_input = [t.strip() for t in guide_input.strip().split('\n') if t.strip()]
    st.sidebar.success(f" **{len(guide_themes_input)} thèmes ciblés**")

# PARAMÈTRES SUPPLÉMENTAIRES
st.sidebar.subheader("**Style FAQ**")
faq_tone = st.sidebar.selectbox("*Ton FAQ*", ["Professionnel", "Amical", "Technique"])
max_response_length = st.sidebar.slider("Longueur réponses", 150, 800, 300)

st.sidebar.caption(" **Guide thèmes = FAQ 100% personnalisée**")

# API Key
api_key = st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    st.error(" Clé API manquante dans `.streamlit/secrets.toml`")
    st.stop()

## CHARGEMENT MODÈLES 
@st.cache_resource
def charger_modeles():
    try:
        if Path("webfaq_artifacts/sentence_transformer_model").exists():
            bert = SentenceTransformer("webfaq_artifacts/sentence_transformer_model")
            st.success(" TON BERT local chargé")
        else:
            bert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            st.info("ℹ BERT standard utilisé")
        
        kmeans_paths = ["kmeans_model.joblib", "webfaq_artifacts/kmeans_model.joblib"]
        kmeans = None
        for path in kmeans_paths:
            if Path(path).exists():
                kmeans = joblib.load(path)
                st.success(f" KMeans chargé: {path}")
                break
        
        if kmeans is None:
            st.warning(" Clustering automatique")
        
        return bert, kmeans
    except Exception as e:
        st.error(f" Erreur modèles: {e}")
        bert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return bert, None

# Fonctions existantes (nettoyage, pivot, noms thèmes) INCHANGÉES
def nettoyer_donnees(df):
    st.info(" Nettoyage...")
    df_clean = df.drop_duplicates(subset=['question', 'answer']).copy()
    df_clean = df_clean[
        (df_clean['question'].str.strip().str.len() > 10) &
        (df_clean['answer'].str.strip().str.len() > 20) &
        (df_clean['question'].notna()) &
        (df_clean['answer'].notna())
    ].copy()
    
    SAMPLE_SIZE = min(2000, len(df_clean))
    if SAMPLE_SIZE > 0:
        df_sample = df_clean.head(SAMPLE_SIZE)
        temp_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = temp_model.encode(df_sample['question'].astype(str).tolist())
        sim_matrix = cosine_similarity(embeddings)
        keep_indices = [df_sample.index[i] for i in range(len(df_sample)) 
                       if len(np.where(sim_matrix[i] > 0.95)[0]) == 1]
        df_dedup = df_sample.loc[keep_indices]
    else:
        df_dedup = df_clean
    
    def clean_text(text):
        text = str(text).strip()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\?\!\.,;:\'-]', '', text)
        return text[:500]
    
    df_dedup['question_clean'] = df_dedup['question'].apply(clean_text)
    df_dedup['answer_clean'] = df_dedup['answer'].apply(clean_text)
    return df_dedup.drop_duplicates(subset=['question_clean', 'answer_clean'])

def format_pivot(df_clean):
    pivot_rows = []
    for i, row in df_clean.iterrows():
        pivot_rows.append({
            "id": f"q_{i}",
            "question": row['question_clean'],
            "reponse": row['answer_clean'],
            "theme_id": -1
        })
    return pd.DataFrame(pivot_rows)

## FILTRAGE PAR GUIDE THÈMES
def filtrer_par_guide_themes(pivot_df, bert_model, guide_themes):
    if not guide_themes:
        return pivot_df
    
    st.info(f" **Filtrage sur {len(guide_themes)} thèmes spécifiques...**")
    
    theme_embeddings = bert_model.encode(guide_themes)
    question_embeddings = bert_model.encode(pivot_df["question"].tolist())
    
    similarities = cosine_similarity(question_embeddings, theme_embeddings)
    best_theme_idx = np.argmax(similarities, axis=1)
    best_scores = np.max(similarities, axis=1)
    
    filtered_rows = []
    for i, (theme_idx, score) in enumerate(zip(best_theme_idx, best_scores)):
        if score > 0.35:
            row = pivot_df.iloc[i].copy()
            row['theme_id'] = theme_idx
            row['nom_theme'] = guide_themes[theme_idx]
            filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

def generer_noms_themes(pivot_df, api_key):
    st.info(" **Noms IA des thèmes...**")
    client = Groq(api_key=api_key)
    
    def appel_ia(prompt, max_tokens=40):
        return client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=max_tokens
        ).choices[0].message.content.strip()
    
    noms_themes = {}
    themes_uniques = sorted(pivot_df["theme_id"].unique())
    
    for theme_id in themes_uniques:
        questions_theme = pivot_df[pivot_df["theme_id"] == theme_id]["question"].tolist()
        echantillon = questions_theme[:5]
        texte = "\n".join([f"• {q[:60]}..." for q in echantillon])
        
        prompt = f"""Expert FAQ français. À partir de ces questions, proposez UN SEUL titre de thème (3-7 mots maximum)
        et chaque thème doit être distinctif, bien formulé, et pertinent pour un utilisateur final. meme si l'utilisateur 
        a fixé un nombre de thèmes élevé mais que son fichier ne traite pas beaucoup de thèmes ressort uniquement les thèmes que traite
        son fichier d'entrée:

QUESTIONS:
{texte}

TITRE:"""
        
        try:
            titre = appel_ia(prompt)
            titre = re.sub(r'^\d+\.?\s*', '', titre).strip(' "').strip()
            noms_themes[theme_id] = titre[:50]
        except:
            noms_themes[theme_id] = f"Thème {theme_id}"
    
    return noms_themes

def generer_faq_pro(pivot_df, api_key, noms_themes, max_q=5):
    st.info("✨ **Synthèse FAQ professionnelle...**")
    client = Groq(api_key=api_key)
    
    def appel_ia(prompt, max_tokens=900):
        return client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=max_tokens
        ).choices[0].message.content.strip()
    
    faq_complet = "# FAQ Professionnelle Générée\n"
    
    top_themes = pivot_df["theme_id"].value_counts().head(12).index
    
    for theme_id in top_themes:
        donnees_theme = pivot_df[pivot_df["theme_id"] == theme_id]
        titre_theme = noms_themes.get(theme_id, f"Thème {theme_id}")
        
        donnees_theme['longueur_q'] = donnees_theme['question'].str.len()
        top_paires = donnees_theme.nlargest(10, 'longueur_q')[['question', 'reponse']]
        
        prompt = f"""Expert FAQ français. À partir de ces paires Q/R d'un même thème, 
        créez une FAQ parfaite avec des questions naturelles et des réponses précises, sans numéroté des questions ni les reponses.:

**THÈME : {titre_theme}**

PAIRES:
"""
        for _, row in top_paires.iterrows():
            prompt += f"Q: {row['question'][:120]}\nR: {row['reponse'][:160]}\n\n"
        
        prompt += f"""FORMAT Markdown:

## {titre_theme}

**Question 1?**

Réponse 1.

**Question 2?**

Réponse 2.

**RÈGLES:**
- {max_q} questions MAX
- Questions naturelles
- Français pro"""
        
        try:
            section = appel_ia(prompt)
            faq_complet += section + "\n\n" #+ "━"*80 + "\n\n"
        except:
            faq_complet += f"""## {titre_theme}

**Question exemple?**

Réponse exemple.

""" + "━"*80 + "\n\n"
    
    return faq_complet

## INTERFACE PRINCIPALE
if "resultats" not in st.session_state:
    st.session_state.resultats = None

fichier_csv = st.file_uploader("📁 **Uploadez votre CSV**", type=["csv"])

if fichier_csv:
    df = pd.read_csv(fichier_csv)
    st.success(f" ***{len(df):,} lignes*** analysées")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(" **Aperçu données**")
        st.dataframe(df[['question', 'answer']].head(10), use_container_width=True)
    
    with col2:
        st.metric(" Lignes", len(df))
    
    if st.button(" **GÉNÉRER FAQ PRO**", type="primary", use_container_width=True):
        with st.spinner(" **Pipeline IA complet...**"):
            try:
                bert_modele, kmeans_modele = charger_modeles()
                df_propre = nettoyer_donnees(df)
                pivot_df = format_pivot(df_propre)
                
                embeddings = bert_modele.encode(pivot_df["question"].tolist(), batch_size=32)
                if kmeans_modele is not None:
                    themes = kmeans_modele.predict(embeddings)
                else:
                    n_auto_clusters = min(n_clusters, len(pivot_df)//10 or 10)
                    kmeans_auto = KMeans(n_clusters=n_auto_clusters, random_state=42, n_init=10)
                    themes = kmeans_auto.fit_predict(embeddings)
                pivot_df["theme_id"] = themes
                
                # GUIDE THÈMES
                if use_guide_themes and guide_themes_input:
                    pivot_df = filtrer_par_guide_themes(pivot_df, bert_modele, guide_themes_input)
                
                noms_themes = generer_noms_themes(pivot_df, api_key)
                pivot_df["nom_theme"] = pivot_df["theme_id"].map(noms_themes)
                faq_finale = generer_faq_pro(pivot_df, api_key, noms_themes, max_questions)
                
                st.session_state.resultats = {
                    "pivot": pivot_df,
                    "faq": faq_finale,
                    "themes": noms_themes,
                    "guide_used": use_guide_themes
                }
                st.success(" **FAQ GÉNÉRÉE !**")
                st.balloons()
                
            except Exception as e:
                st.error(f" Erreur: {str(e)}")

# ========================================
# RÉSULTATS
# ========================================
if st.session_state.get("resultats"):
    resultats = st.session_state.resultats
    
    tabs = st.tabs(["📚 **FAQ Pro**", "🎯 **Thèmes**", "🔍 **Données**", "📊 **Stats**"])
    
    with tabs[0]:
        st.markdown(resultats["faq"])
        st.download_button(" **FAQ Markdown**", resultats["faq"], "FAQ.md")
    
    with tabs[1]:
        if resultats.get("guide_used"):
            st.success(f" **Guide thèmes activé** - {len(guide_themes_input)} thèmes filtrés")
        stats_themes = resultats["pivot"]['theme_id'].value_counts().head(12)
        st.subheader(" **Répartition thèmes**")
        st.bar_chart(stats_themes)

        theme_selectionne = st.selectbox(
            " **Explorer un thème :**", 
            resultats["pivot"]["nom_theme"].unique()
        )
        donnees_theme = resultats["pivot"][
            resultats["pivot"]["nom_theme"] == theme_selectionne
        ][["nom_theme", "question", "reponse"]].head(10)
        st.dataframe(donnees_theme, use_container_width=True)
    
    with tabs[2]:
        st.dataframe(resultats["pivot"][["question", "theme_id", "nom_theme"]].head(20))
    
    with tabs[3]:
        col1, col2, col3 = st.columns(3)
        col1.metric(" Paires", f"{len(resultats['pivot']):,}")
        col2.metric(" Thèmes", len(resultats["pivot"]['theme_id'].unique()))
        col3.metric(" Q/thème", f"{len(resultats['pivot'])/len(resultats['pivot']['theme_id'].unique()):.0f}")

# ========================================
# FOOTER BLANC SUR NOIR - ÉTENDU 100%
# ========================================
st.markdown("""
<div class="footer-pro">
    <p> Générateur Intelligent De FAQ.  Vos données → FAQ parfaite en 1 clic</p>
</div>
""", unsafe_allow_html=True)
