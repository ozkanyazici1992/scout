import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests
import io
import warnings

warnings.filterwarnings('ignore')

# Sayfa ayarları
st.set_page_config(
    page_title="Turquoise Scout AI",
    page_icon="⚽",
    layout="wide"
)

# CSS ile mavi tema
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #1e3a8a;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        font-size: 18px;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    h1 {
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

def normalize_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text)
    text = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()

@st.cache_data
def load_data():
    file_id = '1nl2hcZP6GltTtjPFzjb8KuOmzRqDLjf6'
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            csv_content = io.StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_content)
            
            df.columns = df.columns.str.strip().str.lower()
            
            col_map = {
                'Name': ['name', 'player', 'full name', 'ad soyad'],
                'Club': ['club', 'team', 'current club', 'takim'],
                'Position': ['position', 'pos', 'bp', 'mevki'],
                'Overall': ['overall', 'ova', 'rating', 'guc'],
                'Potential': ['potential', 'pot'],
                'Age': ['age', 'yas'],
                'Value': ['value', 'market value', 'deger'],
                'Wage': ['wage', 'maas'],
                'Preferred Foot': ['foot', 'preferred foot', 'ayak'],
                'Finishing': ['finishing', 'bitiricilik'],
                'Heading': ['heading', 'headingaccuracy', 'kafa'],
                'Speed': ['sprint', 'speed', 'hiz'],
                'Dribbling': ['dribbling'],
                'Strength': ['strength', 'guc'],
                'LongShots': ['longshots']
            }
            
            rename_dict = {}
            for target, keywords in col_map.items():
                for col in df.columns:
                    if any(k in col for k in keywords) and target not in rename_dict.values():
                        rename_dict[col] = target
                        break
            df.rename(columns=rename_dict, inplace=True)
            
            for col in col_map.keys():
                if col not in df.columns:
                    df[col] = 0 if col not in ['Name', 'Club', 'Position', 'Preferred Foot'] else 'Bilinmiyor'
            
            df['Name'] = df['Name'].astype(str)
            df['Clean_Name'] = df['Name'].apply(normalize_text)
            
            for col in ['Value', 'Wage']:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('€', '', regex=False).str.replace('£', '', regex=False)
                    df[col] = df[col].str.replace('K', '000', regex=False).str.replace('M', '000000', regex=False)
                    df[col] = df[col].str.replace('.', '', regex=False)
                    extracted = df[col].str.extract(r'(\d+)', expand=False)
                    df[col] = pd.to_numeric(extracted, errors='coerce').fillna(0)
            
            num_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage', 'Finishing', 'Heading', 'Speed']
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df, ['Overall', 'Potential', 'Age', 'Value', 'Wage']
        else:
            st.error("Veri yüklenemedi!")
            return None, None
    except Exception as e:
        st.error(f"Hata: {e}")
        return None, None

def analyze_player(df, player_name, features):
    clean_input = normalize_text(player_name)
    
    matches = df[df['Clean_Name'].str.contains(clean_input, na=False, regex=False)]
    
    target = None
    if not matches.empty:
        target = matches.sort_values(by='Overall', ascending=False).iloc[0]
    else:
        all_names = df['Clean_Name'].unique().tolist()
        close = difflib.get_close_matches(clean_input, all_names, n=1, cutoff=0.5)
        if close:
            target = df[df['Clean_Name'] == close[0]].iloc[0]
    
    if target is None:
        return None, None
    
    target_pos = target.get('Position', None)
    pool = df[df['Position'] == target_pos].copy()
    if len(pool) < 2:
        pool = df.copy()
    
    scaler = StandardScaler()
    X = pool[features]
    X_scaled = scaler.fit_transform(X)
    
    knn = NearestNeighbors(n_neighbors=min(11, len(pool)), metric='euclidean')
    knn.fit(X_scaled)
    
    target_vec = scaler.transform(target[features].to_frame().T)
    distances, indices = knn.kneighbors(target_vec)
    
    recommendations = []
    for i, idx in enumerate(indices[0][1:]):
        n = pool.iloc[idx]
        score = max(0, 100 - (distances[0][i + 1] * 10))
        recommendations.append({
            "Oyuncu": n['Name'],
            "Takım": str(n.get('Club', '-'))[:25],
            "Güç": int(n.get('Overall', 0)),
            "Yaş": int(n.get('Age', 0)),
            "Değer (€)": f"{n.get('Value', 0):,.0f}",
            "Uyum": f"%{score:.0f}"
        })
    
    return target, recommendations

# Ana sayfa
st.title("⚽ TURQUOISE SCOUT AI")
st.markdown("### Futbolcu Analiz ve Öneri Sistemi")

df, features = load_data()

if df is not None:
    player_input = st.text_input("🔍 Oyuncu Adını Girin:", placeholder="Örnek: Messi, Ronaldo, Haaland...")
    
    if st.button("🎯 Analiz Et"):
        if player_input:
            with st.spinner("Analiz ediliyor..."):
                target, recommendations = analyze_player(df, player_input, features)
                
                if target is not None:
                    st.markdown("---")
                    st.subheader(f"📊 {target['Name'].upper()}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⚡ Güç", int(target.get('Overall', 0)))
                    with col2:
                        st.metric("🎂 Yaş", int(target.get('Age', 0)))
                    with col3:
                        st.metric("💎 Potansiyel", int(target.get('Potential', 0)))
                    with col4:
                        st.metric("💰 Değer", f"€{target.get('Value', 0):,.0f}")
                    
                    col5, col6 = st.columns(2)
                    with col5:
                        st.write(f"**🏟️ Takım:** {target.get('Club', '-')}")
                        st.write(f"**📍 Mevki:** {target.get('Position', '-')}")
                    with col6:
                        st.write(f"**🦶 Ayak:** {target.get('Preferred Foot', '-')}")
                        st.write(f"**💵 Maaş:** €{target.get('Wage', 0):,.0f}")
                    
                    st.markdown("---")
                    st.subheader("🔄 Benzer Oyuncular (Top 10)")
                    
                    rec_df = pd.DataFrame(recommendations)
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
                    
                else:
                    st.error(f"❌ '{player_input}' oyuncusu bulunamadı. Lütfen farklı bir isim deneyin.")
        else:
            st.warning("⚠️ Lütfen bir oyuncu adı girin.")
else:
    st.error("❌ Veri tabanı yüklenemedi. Lütfen tekrar deneyin.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: white;'>💙 Turquoise Scout AI | Powered by Streamlit</div>",
    unsafe_allow_html=True
)
