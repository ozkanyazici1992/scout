import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. TASARIM VE TEMA (TURKUAZ & SİYAH)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Turquoise Scout AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Özel CSS: Siyah Arka Plan, Turkuaz Detaylar
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #00E5FF !important; /* Neon Turkuaz */
        font-family: 'Courier New', sans-serif;
        text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.3);
    }
    
    /* Metin Girdisi */
    .stTextInput>div>div>input {
        background-color: #161B22;
        color: #00E5FF;
        border: 1px solid #00E5FF;
    }
    
    /* Buton */
    .stButton>button {
        background-color: #008B8B; /* Koyu Turkuaz */
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00E5FF; /* Parlak Turkuaz */
        color: #000000;
        box-shadow: 0px 0px 15px #00E5FF;
    }
    
    /* Bilgi Kartları (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-top: 3px solid #00E5FF; /* Üst çizgi turkuaz */
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        color: #00E5FF !important;
    }
    
    /* Tablo */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363D;
    }
    
    /* Uyarı Kutuları */
    .stAlert {
        background-color: #161B22;
        color: #E0E0E0;
        border-left: 5px solid #00E5FF;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE HAZIRLIK
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    
    try:
        df = pd.read_csv(url)
        # Sütun isimlerindeki boşlukları temizle (Name hatasını önler)
        df.columns = df.columns.str.strip()
    except:
        return None, None

    # İsim Temizleme
    def normalize_name(text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.lower().strip()

    # Eğer 'Name' sütunu yoksa kullanıcıya seçtirmek yerine otomatik bulmaya çalış
    if 'Name' not in df.columns:
        # İçinde 'name' veya 'ad' geçen ilk sütunu bul
        possible_cols = [c for c in df.columns if 'name' in c.lower() or 'ad' in c.lower()]
        if possible_cols:
            df['Name'] = df[possible_cols[0]]
        else:
            return None, "İsim sütunu bulunamadı"

    df['Clean_Name'] = df['Name'].apply(normalize_name)
    
    # Work Rate Dönüştürme
    def work_rate_score(wr):
        if not isinstance(wr, str): return 1
        scores = {'Low': 1, 'Medium': 2, 'High': 3}
        parts = wr.split('/')
        if len(parts) == 2:
            return scores.get(parts[0].strip(), 1) + scores.get(parts[1].strip(), 1)
        return 2

    if 'Work Rate' in df.columns:
        df['Work_Rate_Score'] = df['Work Rate'].apply(work_rate_score)
    else:
        df['Work_Rate_Score'] = 2
        
    features = [
        'Overall', 'Potential', 'Value(£)', 'Wage(£)', 
        'Age', 'International Reputation', 'Skill Moves', 
        'Weak Foot', 'Special', 'Work_Rate_Score',
        'Height(cm.)', 'Weight(lbs.)'
    ]
    
    available_features = [f for f in features if f in df.columns]
    df[available_features] = df[available_features].fillna(df[available_features].median())
    
    return df, available_features

# Veriyi Yükle
with st.spinner('Veri tabanı yükleniyor...'):
    df, feature_cols = load_data()

if df is None:
    st.error("Veri yüklenemedi. Sütun isimlerini kontrol edin.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU
# -----------------------------------------------------------------------------
def get_player(df, name_input):
    clean_input = unicodedata.normalize('NFKD', name_input).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    # Tam Eşleşme
    matches = df[df['Clean_Name'].str.contains(clean_input, na=False)]
    if not matches.empty:
        return matches.sort_values(by='Overall', ascending=False).iloc[0], None
    
    # Benzerlik (Typo) Kontrolü
    all_names = df['Clean_Name'].unique().tolist()
    close = difflib.get_close_matches(clean_input, all_names, n=1, cutoff=0.6)
    
    if close:
        found = df[df['Clean_Name'] == close[0]].iloc[0]
        return found, f"Aradığınız isim '{found['Name']}' olabilir mi?"
    
    return None, None

def get_advice(player):
    """Oyuncu hakkında metin tabanlı tavsiyeler üretir"""
    advice = []
    
    # Fırsat Analizi
    if player['Value(£)'] > player['Release Clause(£)'] and player['Release Clause(£)'] > 0:
        kar = player['Value(£)'] - player['Release Clause(£)']
        advice.append(f"🔥 **KELEPİR FIRSAT:** Serbest kalma bedeli, piyasa değerinden £{kar:,} daha düşük! Hemen alınmalı.")
    
    # Gelişim Analizi
    diff = player['Potential'] - player['Overall']
    if diff >= 5:
        advice.append(f"📈 **YATIRIMLIK:** Oyuncu henüz zirve yapmadı. +{diff} puan daha gelişebilir.")
    elif player['Age'] <= 21 and player['Overall'] >= 80:
        advice.append("💎 **WONDERKID:** Nadir bulunan genç yetenek. Takımın geleceği olur.")
    elif player['Age'] > 32:
        advice.append("⚠️ **KISA VADELİ:** Oyuncu kariyerinin sonlarında. Maaş bütçesini zorlayabilir.")
        
    # Sözleşme
    if player['Contract Valid Until'] <= 2024:
        advice.append(f"⏳ **SÖZLEŞME:** Sözleşmesi yakında bitiyor ({int(player['Contract Valid Until'])}). Ucuza kapatılabilir.")
        
    if not advice:
        advice.append("✅ **STABİL:** Standart bir profil. Piyasa değerinde transfer edilebilir.")
        
    return advice

def find_similar(df, target, features):
    target_pos = target['Position']
    pool = df[df['Position'] == target_pos].copy()
    
    if len(pool) < 5: return None
    
    scaler = StandardScaler()
    scaled_pool = scaler.fit_transform(pool[features])
    
    k = min(len(pool), 11)
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(scaled_pool)
    
    target_vec = scaler.transform(target[features].to_frame().T)
    distances, indices = knn.kneighbors(target_vec)
    
    results = []
    for i, idx in enumerate(indices[0][1:]):
        n = pool.iloc[idx]
        score = max(0, 100 - (distances[0][i+1] * 5))
        
        # Kısa Yorum
        tag = "Benzer"
        if n['Value(£)'] < target['Value(£)']/2: tag = "📉 Daha Ucuz"
        elif n['Overall'] > target['Overall']: tag = "🏆 Daha İyi"
        elif n['Potential'] > target['Potential']: tag = "🚀 Potansiyelli"
        elif n['Age'] < target['Age']-3: tag = "👶 Daha Genç"
        
        results.append({
            "Oyuncu": n['Name'],
            "Takım": n['Club'],
            "Yaş": n['Age'],
            "Güç": n['Overall'],
            "Değer": f"£{n['Value(£)']:,}",
            "Uyumluluk": f"%{score:.1f}",
            "Durum": tag
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 4. ARAYÜZ
# -----------------------------------------------------------------------------
st.title("TURQUOISE SCOUT 💎")
st.markdown("Futbolcu analizi ve yapay zeka destekli alternatif öneri sistemi.")
st.divider()

# Arama Çubuğu
col1, col2 = st.columns([3, 1])
with col1:
    search_name = st.text_input("Oyuncu Adı Girin:", placeholder="Örn: Mbappe, Messi, Arda Guler...")
with col2:
    st.write("")
    st.write("")
    btn = st.button("ANALİZ ET 🔍")

if btn or search_name:
    if not search_name:
        st.warning("Lütfen bir isim yazın.")
    else:
        player, msg = get_player(df, search_name)
        
        if player is None:
            st.error("Oyuncu bulunamadı.")
        else:
            if msg: st.info(msg)
            
            # --- PROFİL KARTI ---
            st.subheader(f"{player['Name'].upper()} ({player['Club']})")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mevki", player['Position'])
            m2.metric("Güç", player['Overall'], f"Potansiyel: {player['Potential']}")
            m3.metric("Yaş", player['Age'])
            m4.metric("Değer", f"£{player['Value(£)']:,}")
            
            # --- TAVSİYELER ---
            st.markdown("### 📝 AI ANALİST TAVSİYESİ")
            advices = get_advice(player)
            for adv in advices:
                st.markdown(f"> {adv}")
            
            # --- BENZER OYUNCULAR ---
            st.markdown("---")
            st.markdown(f"### 🔄 {player['Name']} YERİNE ALINABİLECEK EN İYİ 10 {player['Position']}")
            
            sim_df = find_similar(df, player, feature_cols)
            
            if sim_df is not None:
                st.dataframe(
                    sim_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Uyumluluk": st.column_config.ProgressColumn(
                            "Benzerlik Oranı",
                            format="%s",
                            min_value=0,
                            max_value=100,
                        ),
                    }
                )
            else:
                st.warning("Yeterli veri yok.")
