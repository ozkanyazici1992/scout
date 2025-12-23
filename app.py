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
    initial_sidebar_state="expanded"
)

# Özel CSS
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
    .stSidebar { background-color: #161B22 !important; border-right: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE AKILLI KOLON SEÇİMİ
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_raw():
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip() # Boşlukları temizle
        return df
    except:
        return None

# Veriyi İndir
with st.spinner('Sistem başlatılıyor...'):
    df = load_data_raw()

if df is None:
    st.error("❌ Veri indirilemedi.")
    st.stop()

# --- AKILLI KOLON BULUCU ---
# Hedef: 'Name' sütununu bulmak.
# Strateji: Önce 'Name' ara, yoksa ilk 'Metin' (Object) sütununu al.

target_col = None

# 1. 'Name' veya 'Ad' içeren sütun var mı?
for col in df.columns:
    if 'name' in col.lower() or 'player' in col.lower():
        target_col = col
        break

# 2. Bulamazsa, ilk metin (string) sütununu al
if target_col is None:
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        target_col = text_cols[0]

# Sütunu 'Name' olarak ayarla
if target_col:
    df['Name'] = df[target_col].astype(str)
else:
    st.error("❌ Veri setinde isim içeren bir sütun bulunamadı.")
    st.stop()

# --- SIDEBAR: KONTROL PANELİ ---
st.sidebar.title("🛠️ Veri Kontrolü")
st.sidebar.info(f"İsim Sütunu Olarak Algılanan: **{target_col}**")
st.sidebar.markdown("---")
st.sidebar.write("Veri setinden rastgele 5 örnek:")
st.sidebar.write(df['Name'].sample(5).values)

# Veri Ön İşleme
def normalize_name(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()

df['Clean_Name'] = df['Name'].apply(normalize_name)

# Work Rate
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

# Özellikler
features_list = [
    'Overall', 'Potential', 'Value(£)', 'Wage(£)', 
    'Age', 'International Reputation', 'Skill Moves', 
    'Weak Foot', 'Special', 'Work_Rate_Score',
    'Height(cm.)', 'Weight(lbs.)'
]

# Mevcut olanları doldur
feature_cols = [f for f in features_list if f in df.columns]
if not feature_cols:
    st.error("❌ Analiz için gerekli sayısal sütunlar (Overall, Potential vb.) bulunamadı.")
    st.stop()

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU
# -----------------------------------------------------------------------------
def get_player(df, name_input):
    clean_input = unicodedata.normalize('NFKD', name_input).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    matches = df[df['Clean_Name'].str.contains(clean_input, na=False)]
    if not matches.empty:
        return matches.sort_values(by='Overall', ascending=False).iloc[0], None
    
    all_names = df['Clean_Name'].unique().tolist()
    # Eşik değerini düşürdüm (0.6 -> 0.5) daha esnek olsun diye
    close = difflib.get_close_matches(clean_input, all_names, n=1, cutoff=0.5)
    
    if close:
        found = df[df['Clean_Name'] == close[0]].iloc[0]
        return found, f"Bunu mu demek istediniz: '{found['Name']}'?"
    
    return None, None

def get_advice(player):
    advice = []
    if 'Value(£)' in player and 'Release Clause(£)' in player:
        if player['Value(£)'] > player['Release Clause(£)'] and player['Release Clause(£)'] > 0:
            kar = player['Value(£)'] - player['Release Clause(£)']
            advice.append(f"🔥 **KELEPİR FIRSAT:** Serbest kalma bedeli, değerinden £{kar:,} düşük!")
    
    if 'Potential' in player and 'Overall' in player:
        diff = player['Potential'] - player['Overall']
        if diff >= 5: advice.append(f"📈 **YATIRIMLIK:** +{diff} puan daha gelişebilir.")
    
    if 'Contract Valid Until' in player:
        if player['Contract Valid Until'] <= 2024:
            advice.append(f"⏳ **SÖZLEŞME:** Yakında bitiyor ({int(player['Contract Valid Until'])}).")
        
    if not advice: advice.append("✅ **STABİL:** Standart profil.")
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
        
        tag = "-"
        if 'Value(£)' in n and n['Value(£)'] < target['Value(£)']/2: tag = "📉 Daha Ucuz"
        elif 'Overall' in n and n['Overall'] > target['Overall']: tag = "🏆 Daha İyi"
        elif 'Age' in n and n['Age'] < target['Age']-3: tag = "👶 Daha Genç"
        
        results.append({
            "Oyuncu": n['Name'],
            "Takım": n['Club'],
            "Yaş": n['Age'],
            "Güç": n['Overall'],
            "Değer": f"£{n['Value(£)']:,}" if 'Value(£)' in n else "-",
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

col1, col2 = st.columns([3, 1])
with col1:
    search_name = st.text_input("Oyuncu Adı Girin:", placeholder="Örn: Mbappe, Messi...")
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
            # Hata ayıklama için kullanıcıya yardım
            with st.expander("❓ Sistemde kayıtlı isimleri kontrol et"):
                st.write(df['Name'].head(20))
        else:
            if msg: st.info(msg)
            
            st.subheader(f"{player['Name'].upper()} ({player['Club']})")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mevki", player['Position'])
            m2.metric("Güç", player['Overall'])
            m3.metric("Yaş", player['Age'])
            if 'Value(£)' in player:
                m4.metric("Değer", f"£{player['Value(£)']:,}")
            
            st.markdown("### 📝 AI ANALİST TAVSİYESİ")
            advices = get_advice(player)
            for adv in advices: st.markdown(f"> {adv}")
            
            st.markdown("---")
            st.markdown(f"### 🔄 {player['Name']} YERİNE ALINABİLECEK EN İYİ ALTERNATİFLER")
            
            sim_df = find_similar(df, player, feature_cols)
            
            if sim_df is not None:
                st.dataframe(
                    sim_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Uyumluluk": st.column_config.ProgressColumn("Benzerlik", format="%s", min_value=0, max_value=100),
                    }
                )
            else:
                st.warning("Bu mevkide yeterli veri yok.")
