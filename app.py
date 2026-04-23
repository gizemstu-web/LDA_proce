import streamlit as st
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
import spacy
import pyLDAvis.gensim_models
import streamlit.components.v1 as components
import re
import PyPDF2  # PDF okuma kütüphanesi

# Sayfa yapılandırması
st.set_page_config(page_title="İtalyanca Metin Analizi | LDA", layout="wide")

@st.cache_resource
def load_spacy():
    """İtalyanca dil modelini yükler."""
    return spacy.load("it_core_news_sm")

nlp = load_spacy()

def preprocess_italian_text(text, stop_words):
    """Metni temizler, lemmatization uygular ve durak sözcükleri atar."""
    text = re.sub(r'\s+', ' ', str(text).lower())
    doc = nlp(text)
    # Edebi analizde anlam taşıyan İsim, Sıfat ve Fiilleri filtreleyelim
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.lemma_ not in stop_words]
    return tokens

def main():
    st.title("🇮🇹 İtalyanca Metinler İçin Konu Modellemesi (LDA)")
    st.markdown("""
    Bu uygulama, İtalyanca edebi metinler ve **Relazioni** gibi diplomatik belgeler üzerinde 
    **Latent Dirichlet Allocation** algoritmasını kullanarak gizli temaları keşfetmenizi sağlar.
    """)

    # --- SIDEBAR: PARAMETRE VE DOSYA AYARLARI ---
    st.sidebar.header("Model Parametreleri")
    
    # PDF desteği buraya eklendi
    uploaded_file = st.sidebar.file_uploader("Veri Setini Yükle (CSV, TXT veya PDF)", type=['csv', 'txt', 'pdf'])
    
    num_topics = st.sidebar.slider("Konu Sayısı (K)", min_value=2, max_value=20, value=5)
    passes = st.sidebar.slider("Eğitim Turu (Passes)", min_value=1, max_value=50, value=10)
    alpha = st.sidebar.selectbox("Alpha", ["symmetric", "asymmetric", "auto"])
    
    st.sidebar.divider()
    extra_stop_words = st.sidebar.text_area("Ek Durak Sözcükler (Virgülle ayırın)", placeholder="es: serenissimo, signoria, doge...")

    documents = []

    if uploaded_file is not None:
        # 1. SENARYO: CSV DOSYASI
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            column_name = st.selectbox("Analiz edilecek sütunu seçin:", df.columns)
            documents = df[column_name].dropna().tolist()
            
        # 2. SENARYO: PDF DOSYASI (Yeni eklenen kısım)
        elif uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    # PDF metnini paragraf boşluklarından böleriz
                    paragraphs = text.split('\n\n')
                    # Çok kısa satırları temizleyip listeye ekleriz
                    documents.extend([p.strip() for p in paragraphs if len(p.strip()) > 30])
            st.sidebar.success(f"PDF işlendi: {len(documents)} bölüm bulundu.")
            
        # 3. SENARYO: TXT DOSYASI
        else:
            documents = uploaded_file.read().decode("utf-8").split('\n')

        # --- MODELLEME BAŞLATMA ---
        if st.button("Modeli Eğit ve Görselleştir"):
            if len(documents) > 0:
                with st.spinner("İtalyanca metinler işleniyor..."):
                    
                    custom_stops = [s.strip() for s in extra_stop_words.split(',')] if extra_stop_words else []
                    processed_docs = [preprocess_italian_text(doc, custom_stops) for doc in documents]
                    
                    # Boş kalan dökümanları temizleyelim
                    processed_docs = [doc for doc in processed_docs if len(doc) > 0]
                    
                    dictionary = corpora.Dictionary(processed_docs)
                    corpus = [dictionary.doc2bow(text) for text in processed_docs]
                    
                    lda_model = LdaModel(
                        corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        passes=passes,
                        alpha=alpha,
                        random_state=42
                    )
                    
                    st.subheader("Tespit Edilen Temalar ve Anahtar Kelimeler")
                    cols = st.columns(3)
                    for idx, topic in lda_model.print_topics(-1):
                        cols[idx % 3].write(f"**Tema {idx+1}:**")
                        cols[idx % 3].caption(topic)

                    st.subheader("İnteraktif Konu Haritası")
                    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
                    html_obj = pyLDAvis.prepared_data_to_html(vis_data)
                    components.html(html_obj, height=800, scrolling=True)
            else:
                st.error("Dosyadan metin okunamadı, lütfen formatı kontrol edin.")

    else:
        st.info("Lütfen sol menüden bir dosya yükleyerek başlayın.")

if __name__ == "__main__":
    main()
