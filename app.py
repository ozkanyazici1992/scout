import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# TASARIM VE TEMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Turquoise Scout AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3 { color: #00E5FF !important; font-family: 'Courier New', sans-serif; text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.3); }
    .stTextInput>div>div>input { background-color: #161B22; color: #00E5FF; border: 1px solid #00E5FF; }
    .stButton>button { background-color: #008B8B; color: white; border: none; border-radius: 5px; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { background-color: #00E5FF; color: #000000; box-shadow: 0px 0px 15px #00E5FF; }
    div[data-testid="stMetric"] { background-color: #161B22; border: 1px solid #30363D; border-top: 3px solid #00E5FF; padding: 10px; border-radius: 5px; }
    div[data-testid="stMetricValue"] { color: #00E5FF !important; }
    div[data-testid="stDataFrame"] { border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VERİ YÜKLEME
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # İsim sütununu bul
        name_col = None
        for col in df.columns:
            if 'name' in col.lower() and 'name' not in col.lower().replace('name', ''):
                try:
                    first_val = str(df[col].dropna().iloc[0])
                    if not first_val.isdigit():
                        name_col = col
                        break
                except:
                    continue
        
        if name_col is None:
            name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        df['Name'] = df[name_col].astype(str)
        df['Clean_Name'] = df['Name'].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ASCII', 'ignore').decode('utf-8').lower().strip())
        
        # Work Rate Score
        def work_rate_score(wr):
            if not isinstance(wr, str):
                return 2
            scores = {'Low': 1, 'Medium': 2, 'High': 3}
            parts = wr.split('/')
            if len(parts) == 2:
                return scores.get(parts[0].strip(), 1) + scores.get(parts[1].strip(), 1)
            return 2
        
        if 'Work Rate' in df.columns:
            df['Work_Rate_Score'] = df['Work Rate'].apply(work_rate_score)
        else:
            df['Work_Rate_Score'] = 2
        
        # Feature seçimi
        features = ['Overall', 'Potential', 'Value(£)', 'Wage(£)', 'Age', 
                   'International Reputation', 'Skill Moves', 'Weak Foot', 
                   'Special', 'Work_Rate_Score', 'Height(cm.)', 'Weight(lbs.)']
        
        feature_cols = [f for f in features if f in df.columns]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        return df, feature_cols
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None

with st.spinner('Sistem başlatılıyor...'):
    df, feature_cols = load_data()

if df is None:
    st.error("❌ Veri indirilemedi.")
    st.stop()

# -----------------------------------------------------------------------------
# ANALİZ FONKSİYONLARI
# -----------------------------------------------------------------------------
def find_player(df, name_input):
    clean_input = unicodedata.normalize('NFKD', name_input).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    # Tam eşleşme ara
    matches = df[df['Clean_Name'].str.contains(clean_input, na=False, regex=False)]
    if not matches.empty:
        return matches.sort_values(by='Overall', ascending=False).iloc[0], None
    
    # Yakın eşleşme ara
    all_names = df['Clean_Name'].unique().tolist()
    close = difflib.get_close_matches(clean_input, all_names, n=1, cutoff=0.5)
    
    if close:
        found = df[df['Clean_Name'] == close[0]].iloc[0]
        return found, f"Bunu mu demek istediniz: '{found['Name']}'?"
    
    return None, None

def get_advice(player):
    advice = []
    
    # Fırsat analizi
    if 'Value(£)' in player and 'Release Clause(£)' in player:
        val = player['Value(£)']
        clause = player['Release Clause(£)']
        if clause > 0 and val > clause:
            kar = val - clause
            advice.append(f"🔥 **KELEPİR FIRSAT:** Serbest kalma bedeli değerinden £{kar:,} düşük!")
    
    # Potansiyel analizi
    if 'Potential' in player and 'Overall' in player:
        diff = player['Potential'] - player['Overall']
        if diff >= 5:
            advice.append(f"📈 **YATIRIMLIK OYUNCU:** +{diff} puan daha gelişebilir.")
        elif diff < 2:
            advice.append(f"⭐ **ZİRVEDE:** Potansiyelinin zirvesine ulaşmış.")
    
    # Sözleşme durumu
    if 'Contract Valid Until' in player:
        contract_year = player['Contract Valid Until']
        if contract_year <= 2024:
            advice.append(f"⏳ **SÖZLEŞME BİTİYOR:** {int(contract_year)} yılında sona eriyor.")
    
    # Yaş analizi
    if 'Age' in player:
        age = player['Age']
        if age < 23:
            advice.append(f"👶 **GENÇ YETENEK:** {int(age)} yaşında, kariyer önünde.")
        elif age > 32:
            advice.append(f"🎯 **DENEYİMLİ:** {int(age)} yaşında, kısa vadeli transfer.")
    
    # Genel değerlendirme
    if not advice:
        advice.append("✅ **STABİL PROFIL:** Standart özelliklere sahip oyuncu.")
    
    return advice

def find_similar_players(df, target_player, features, n=10):
    target_pos = target_player['Position']
    pool = df[df['Position'] == target_pos].copy()
    
    if len(pool) < n + 1:
        return None
    
    scaler = StandardScaler()
    scaled_pool = scaler.fit_transform(pool[features])
    
    knn = NearestNeighbors(n_neighbors=n+1, metric='euclidean')
    knn.fit(scaled_pool)
    
    target_vec = scaler.transform(target_player[features].to_frame().T)
    distances, indices = knn.kneighbors(target_vec)
    
    results = []
    for i, idx in enumerate(indices[0][1:n+1]):
        player = pool.iloc[idx]
        similarity = max(0, 100 - (distances[0][i+1] * 5))
        
        # Durum etiketi
        tag = "⚖️ Benzer Seviye"
        if 'Value(£)' in player and player['Value(£)'] < target_player['Value(£)'] * 0.7:
            tag = "💰 Ekonomik"
        elif 'Overall' in player and player['Overall'] > target_player['Overall']:
            tag = "⬆️ Daha İyi"
        elif 'Age' in player and player['Age'] < target_player['Age'] - 3:
            tag = "🌱 Daha Genç"
        
        results.append({
            "Oyuncu": player['Name'],
            "Takım": player['Club'],
            "Yaş": int(player['Age']),
            "Güç": int(player['Overall']),
            "Potansiyel": int(player['Potential']) if 'Potential' in player else "-",
            "Değer": f"£{int(player['Value(£)']):,}" if 'Value(£)' in player else "-",
            "Benzerlik": f"{similarity:.1f}%",
            "Özellik": tag
        })
    
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# ARAYÜZ
# -----------------------------------------------------------------------------
st.title("💎 TURQUOISE SCOUT AI")
st.markdown("*Futbolcu analizi ve yapay zeka destekli alternatif öneri sistemi*")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    search_name = st.text_input("🔍 Oyuncu Adı:", placeholder="Örn: Messi, Ronaldo, Mbappe...")
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("ANALİZ ET", use_container_width=True)

if analyze_btn and search_name:
    player, msg = find_player(df, search_name)
    
    if player is None:
        st.error("❌ Oyuncu bulunamadı. Lütfen ismi kontrol edin.")
    else:
        if msg:
            st.info(msg)
        
        # Oyuncu Bilgileri
        st.markdown(f"## {player['Name'].upper()}")
        st.markdown(f"**🏟️ Takım:** {player['Club']}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mevki", player['Position'])
        col2.metric("Güç", int(player['Overall']))
        col3.metric("Potansiyel", int(player['Potential']) if 'Potential' in player else "-")
        col4.metric("Yaş", int(player['Age']))
        if 'Value(£)' in player:
            col5.metric("Değer", f"£{int(player['Value(£)']):,}")
        
        # AI Tavsiyesi
        st.markdown("---")
        st.markdown("### 🤖 AI ANALİST TAVSİYESİ")
        advices = get_advice(player)
        for adv in advices:
            st.markdown(f"> {adv}")
        
        # Benzer Oyuncular
        st.markdown("---")
        st.markdown(f"### 🔄 {player['Name']} YERİNE ALINAB İLECEK EN İYİ 10 ALTERNATİF")
        
        similar_df = find_similar_players(df, player, feature_cols, n=10)
        
        if similar_df is not None:
            st.dataframe(
                similar_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Benzerlik": st.column_config.ProgressColumn(
                        "Benzerlik",
                        format="%s",
                        min_value=0,
                        max_value=100
                    ),
                }
            )
        else:
            st.warning("⚠️ Bu mevkide yeterli sayıda oyuncu bulunamadı.")

elif analyze_btn and not search_name:
    st.warning("⚠️ Lütfen bir oyuncu adı girin.")
