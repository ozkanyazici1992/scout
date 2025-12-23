import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. SAYFA VE TEMA AYARLARI (KIRMIZI KONSEPT)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Scout - Kırmızı",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile Kırmızı/Siyah Tema Entegrasyonu
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    /* Başlıklar */
    h1, h2, h3 {
        color: #ff4b4b !important; /* Streamlit Kırmızısı */
        font-family: 'Helvetica', sans-serif;
    }
    /* Buton Tasarımı */
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border-radius: 8px;
        border: none;
        height: 50px;
        width: 100%;
        font-weight: bold;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #b71c1c;
        color: white;
    }
    /* Kart Görünümü (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #d32f2f; /* Sol taraf kırmızı çizgi */
    }
    div[data-testid="stMetricLabel"] {
        color: #9e9e9e;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    /* Tablo Tasarımı */
    div[data-testid="stDataFrame"] {
        background-color: #1e1e1e;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME (GOOGLE DRIVE ENTEGRASYONU)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Google Drive Linkinden ID'yi alıp CSV indirme linkine çeviriyoruz
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    
    try:
        # URL'den okuyoruz
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Veri Google Drive'dan çekilemedi. İnternet bağlantınızı kontrol edin. Hata: {e}")
        return None, None

    # --- Veri Ön İşleme ---
    def normalize_name(text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.lower().strip()

    def work_rate_score(wr):
        if not isinstance(wr, str): return 1
        scores = {'Low': 1, 'Medium': 2, 'High': 3}
        parts = wr.split('/')
        if len(parts) == 2:
            return scores.get(parts[0].strip(), 1) + scores.get(parts[1].strip(), 1)
        return 2

    # İsimleri temizle
    if 'Name' in df.columns:
        df['Clean_Name'] = df['Name'].apply(normalize_name)
    
    # Work Rate skorla
    if 'Work Rate' in df.columns:
        df['Work_Rate_Score'] = df['Work Rate'].apply(work_rate_score)
    
    # Sayısal özellikler
    features = [
        'Overall', 'Potential', 'Value(£)', 'Wage(£)', 
        'Age', 'International Reputation', 'Skill Moves', 
        'Weak Foot', 'Special', 'Work_Rate_Score',
        'Height(cm.)', 'Weight(lbs.)'
    ]
    
    # Sütunlar var mı kontrol et, yoksa hata vermesin diye doldur
    available_features = [f for f in features if f in df.columns]
    
    # Eksik verileri doldur
    df[available_features] = df[available_features].fillna(df[available_features].median())
    
    return df, available_features

# Yükleme Göstergesi
with st.spinner('Veriler Google Drive üzerinden indiriliyor ve işleniyor...'):
    df, feature_cols = load_data()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. MANTIKSAL FONKSİYONLAR
# -----------------------------------------------------------------------------
def get_player_suggestions(df, search_term):
    """İsim düzeltme ve tahmin mekanizması"""
    clean_term = unicodedata.normalize('NFKD', search_term).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    # Tam Eşleşme
    matches = df[df['Clean_Name'].str.contains(clean_term, na=False)]
    if not matches.empty:
        return matches.sort_values(by='Overall', ascending=False).iloc[0], None
    
    # Fuzzy Match (Yazım Hatası)
    all_names = df['Clean_Name'].unique().tolist()
    close_matches = difflib.get_close_matches(clean_term, all_names, n=1, cutoff=0.6)
    
    if close_matches:
        found_name = close_matches[0]
        suggestion = df[df['Clean_Name'] == found_name].iloc[0]
        return suggestion, f"Bunu mu demek istediniz: **{suggestion['Name']}**?"
    
    return None, None

def calculate_similarity(df, target_player, features):
    """KNN Modeli ile benzerleri bulur (MEVKİ KİLİTLİ)"""
    target_pos = target_player['Position']
    
    # MEVKİ FİLTRESİ
    pool = df[df['Position'] == target_pos].copy()
    
    if len(pool) < 5:
        return None, "Yetersiz Veri"
    
    # Scale ve Model
    scaler = StandardScaler()
    scaled_pool = scaler.fit_transform(pool[features])
    
    k = min(len(pool), 11)
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(scaled_pool)
    
    # Hedef Vektör
    target_vector = scaler.transform(target_player[features].to_frame().T)
    distances, indices = knn.kneighbors(target_vector)
    
    recommendations = []
    # indices[0][1:] -> Kendisi hariç diğerleri
    for i, idx in enumerate(indices[0][1:]):
        neighbor = pool.iloc[idx]
        
        dist = distances[0][i+1]
        score = max(0, 100 - (dist * 5))
        
        # Yorum Mantığı
        comment = "-"
        if neighbor['Value(£)'] < target_player['Value(£)'] / 2: comment = "💰 Bütçe Dostu"
        elif neighbor['Overall'] > target_player['Overall']: comment = "🏆 Daha Güçlü"
        elif neighbor['Age'] < target_player['Age'] - 3: comment = "👶 Genç Yetenek"
        elif neighbor['Potential'] > target_player['Potential']: comment = "🚀 Yüksek Potansiyel"
        elif abs(neighbor['Overall'] - target_player['Overall']) < 2: comment = "⚖️ Dengi"

        recommendations.append({
            'Oyuncu': neighbor['Name'],
            'Mevki': neighbor['Position'],
            'Takım': neighbor['Club'],
            'Yaş': neighbor['Age'],
            'Güç': neighbor['Overall'],
            'Değer (£)': f"£{neighbor['Value(£)']:,}",
            'Benzerlik': f"%{score:.1f}",
            'Not': comment
        })
        
    return pd.DataFrame(recommendations), None

# -----------------------------------------------------------------------------
# 4. ARAYÜZ (UI) TASARIMI
# -----------------------------------------------------------------------------

# Başlık Bölümü
st.title("🦁 AI FOOTBALL SCOUT")
st.markdown("Yapay zeka destekli, mevkii hassasiyetli oyuncu öneri sistemi.")
st.divider()

# Arama Bölümü
col_search, col_btn = st.columns([4, 1])
with col_search:
    player_name = st.text_input("Futbolcu Adı Girin (Örn: Mbappe, Van Dijk, Ozil)", placeholder="Oyuncu adı yazıp Enter'a basın...")
with col_btn:
    st.write("") 
    st.write("") 
    search_clicked = st.button("ANALİZ ET")

# --- SONUÇ EKRANI ---
if search_clicked or player_name:
    if not player_name:
        st.warning("Lütfen bir isim girin.")
    else:
        target_player, suggestion_msg = get_player_suggestions(df, player_name)
        
        if target_player is None:
            st.error(f"❌ '{player_name}' veritabanında bulunamadı.")
        else:
            if suggestion_msg:
                st.info(f"⚠️ '{player_name}' bulunamadı. {suggestion_msg} analiz ediliyor.")
            
            # --- HEDEF OYUNCU KARTI ---
            st.subheader(f"🎯 Hedef: {target_player['Name']} ({target_player['Club']})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Mevki", target_player['Position'])
            with col2: st.metric("Güç (Overall)", target_player['Overall'])
            with col3: st.metric("Yaş", target_player['Age'])
            with col4: st.metric("Piyasa Değeri", f"£{target_player['Value(£)']:,}")
            
            # --- ANALİZ VE LİSTE ---
            st.markdown("---")
            st.subheader(f"✅ {target_player['Name']} Yerine Oynayabilecek {target_player['Position']} Alternatifleri")
            
            rec_df, error = calculate_similarity(df, target_player, feature_cols)
            
            if error:
                st.warning(f"⚠️ {target_player['Position']} mevkisinde yeterli veri yok.")
            else:
                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Benzerlik": st.column_config.ProgressColumn(
                            "Benzerlik Skoru",
                            format="%s",
                            min_value=0,
                            max_value=100,
                        ),
                        "Oyuncu": st.column_config.TextColumn("Oyuncu Adı", width="medium"),
                        "Not": st.column_config.TextColumn("Yapay Zeka Yorumu", width="small"),
                    }
                )
