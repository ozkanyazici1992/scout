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
# 1. TASARIM VE TEMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Turquoise Scout AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3 { color: #00E5FF !important; font-family: 'Courier New', sans-serif; }
    .stTextInput>div>div>input { background-color: #161B22; color: #00E5FF; border: 1px solid #00E5FF; }
    .stButton>button { background-color: #008B8B; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #00E5FF; color: #000000; }
    div[data-testid="stMetric"] { background-color: #161B22; border: 1px solid #30363D; border-top: 3px solid #00E5FF; }
    div[data-testid="stMetricValue"] { color: #00E5FF !important; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ YÜKLEME - "KURŞUN GEÇİRMEZ" MODÜL
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_robust():
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    
    try:
        df = pd.read_csv(url)
        # Tüm sütun isimlerini küçük harfe çevirip boşlukları temizleyelim
        df.columns = df.columns.str.strip().str.lower()
        
        # --- SÜTUN EŞLEŞTİRME HARİTASI ---
        # Veri setinde olabilecek tüm varyasyonları buraya yazıyoruz
        column_mapping = {
            'Name': ['name', 'player', 'full name', 'ad soyad'],
            'Club': ['club', 'team', 'current club', 'takim', 'kulup'],
            'Position': ['position', 'pos', 'bp', 'mevki'],
            'Overall': ['overall', 'ova', 'rating', 'guc'],
            'Potential': ['potential', 'pot', 'potansiyel'],
            'Age': ['age', 'yas'],
            'Value': ['value', 'market value', 'deger'],
            'Wage': ['wage', 'salary', 'maas']
        }

        # Sütunları standartlaştır (Örn: 'team' -> 'Club')
        found_cols = {}
        for target, keywords in column_mapping.items():
            for col in df.columns:
                if any(keyword in col for keyword in keywords):
                    # Eğer bu sütun daha önce kullanılmadıysa eşleştir
                    if col not in found_cols.values():
                        found_cols[target] = col
                        break
        
        # Sütun isimlerini değiştir
        df.rename(columns={v: k for k, v in found_cols.items()}, inplace=True)

        # --- EKSİK SÜTUN GARANTİSİ ---
        # Eğer eşleşme bulunamadıysa, program çökmesin diye boş sütun oluştur
        required_cols = ['Name', 'Club', 'Position', 'Overall', 'Potential', 'Age', 'Value', 'Wage']
        for col in required_cols:
            if col not in df.columns:
                if col in ['Overall', 'Potential', 'Age']:
                    df[col] = 0
                else:
                    df[col] = "Bilinmiyor"

        # --- VERİ TEMİZLEME ---
        # İsim temizliği (ID numaralarını atlamak için)
        # Eğer Name sütunu sayısal ise, string'e çevir veya yanlış sütunsa düzeltmeye çalış
        df['Name'] = df['Name'].astype(str)
        
        # Temiz İsim (Arama için)
        def normalize_name(text):
            if not isinstance(text, str): return ""
            return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

        df['Clean_Name'] = df['Name'].apply(normalize_name)
        
        # Sayısal Değerleri Temizle ('€100M' gibi ifadeleri sayıya çevir)
        for col in ['Value', 'Wage']:
            if df[col].dtype == 'object':
                df[col] = (df[col].astype(str).str.replace('€', '')
                                             .str.replace('£', '')
                                             .str.replace('K', '000')
                                             .str.replace('M', '000000')
                                             .str.replace('.', '')
                                             .str.extract('(\d+)').astype(float))
        
        # Sayısal sütunlardaki boşlukları doldur
        num_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Feature Sütunları (KNN Modeli için)
        feature_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage']
        
        return df, feature_cols

    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None

# Veriyi Yükle
with st.spinner('Saha taranıyor...'):
    df, feature_cols = load_data_robust()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. MANTIKSAL FONKSİYONLAR
# -----------------------------------------------------------------------------
def get_player(df, name_input):
    clean_input = unicodedata.normalize('NFKD', name_input).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    # Tam Eşleşme
    matches = df[df['Clean_Name'].str.contains(clean_input, na=False)]
    if not matches.empty:
        return matches.sort_values(by='Overall', ascending=False).iloc[0], None
    
    # Benzerlik
    all_names = df['Clean_Name'].unique().tolist()
    close = difflib.get_close_matches(clean_input, all_names, n=1, cutoff=0.5)
    
    if close:
        found = df[df['Clean_Name'] == close[0]].iloc[0]
        return found, f"Bunu mu demek istediniz: **{found['Name']}**?"
    
    return None, None

def get_advice(player):
    advice = []
    # Potansiyel
    if player['Potential'] - player['Overall'] >= 5:
        advice.append(f"📈 **YATIRIMLIK:** Oyuncu +{int(player['Potential'] - player['Overall'])} puan daha gelişebilir.")
    
    # Yaş
    if player['Age'] <= 21:
        advice.append(f"👶 **GENÇ YETENEK:** Henüz {int(player['Age'])} yaşında.")
    elif player['Age'] >= 33:
        advice.append("⚠️ **RİSKLİ YAŞ:** Fiziksel düşüş yaşayabilir.")
        
    # Değer (Basit mantık)
    if player['Overall'] > 85 and player['Value'] < 50000000:
        advice.append("🔥 **FIRSAT:** Gücüne göre piyasa değeri uygun.")

    if not advice:
        advice.append("✅ **STABİL:** Standart profil.")
    return advice

def find_similar(df, target, features):
    # Pozisyon Kilidi
    target_pos = target['Position']
    pool = df[df['Position'] == target_pos].copy()
    
    if len(pool) < 5:
        # Eğer pozisyonda yeterli adam yoksa tüm havuza bak (Çökmemesi için)
        pool = df.copy()
    
    scaler = StandardScaler()
    # Özellikleri ölçeklendir
    X = pool[features]
    X_scaled = scaler.fit_transform(X)
    
    k = min(len(pool), 11)
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_scaled)
    
    # Hedef vektör
    target_vec = scaler.transform(target[features].to_frame().T)
    distances, indices = knn.kneighbors(target_vec)
    
    results = []
    for i, idx in enumerate(indices[0][1:]):
        n = pool.iloc[idx]
        score = max(0, 100 - (distances[0][i+1] * 10)) # Skorlama
        
        # Etiket
        tag = "Benzer"
        if n['Value'] < target['Value'] * 0.7: tag = "💰 Daha Ucuz"
        elif n['Overall'] > target['Overall']: tag = "🏆 Daha Güçlü"
        elif n['Age'] < target['Age'] - 3: tag = "👶 Genç"
        
        results.append({
            "Oyuncu": n['Name'],
            "Kulüp": n['Club'],
            "Yaş": int(n['Age']),
            "Güç": int(n['Overall']),
            "Değer": f"€{n['Value']:,.0f}",
            "Uyumluluk": f"%{score:.0f}",
            "Durum": tag
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 4. ARAYÜZ
# -----------------------------------------------------------------------------
st.title("💎 TURQUOISE SCOUT AI")
st.markdown("*Futbolcu Analizi ve Akıllı Transfer Önerileri*")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    search_name = st.text_input("Oyuncu Adı:", placeholder="Örn: Messi, Arda Guler...")
with col2:
    st.write("")
    st.write("")
    btn = st.button("ANALİZ ET")

if btn or search_name:
    if not search_name:
        st.warning("Lütfen isim girin.")
    else:
        player, msg = get_player(df, search_name)
        
        if player is None:
            st.error("Oyuncu bulunamadı.")
            # Debug için sütunları göster (Opsiyonel)
            # st.write("Mevcut Sütunlar:", df.columns.tolist())
        else:
            if msg: st.info(msg)
            
            # --- OYUNCU KARTI ---
            # Hata veren kısım burasıydı, artık 'Club' sütunu garanti var.
            club_name = player.get('Club', 'Kulüp Bilinmiyor')
            st.subheader(f"{player['Name'].upper()} ({club_name})")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mevki", player['Position'])
            c2.metric("Güç", int(player['Overall']))
            c3.metric("Yaş", int(player['Age']))
            c4.metric("Değer", f"€{player['Value']:,.0f}")
            
            # --- AI TAVSİYESİ ---
            st.markdown("### 🤖 ANALİST RAPORU")
            for advice in get_advice(player):
                st.markdown(f"> {advice}")
            
            # --- BENZER OYUNCULAR ---
            st.markdown("---")
            st.subheader(f"🔄 {player['Name']} ALTERNATİFLERİ")
            
            sim_df = find_similar(df, player, feature_cols)
            
            if sim_df is not None:
                st.dataframe(
                    sim_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Uyumluluk": st.column_config.ProgressColumn("Benzerlik", min_value=0, max_value=100, format="%s")
                    }
                )
