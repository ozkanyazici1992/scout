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
# 2. VERİ YÜKLEME - "KURŞUN GEÇİRMEZ" MODÜL
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_robust():
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    
    try:
        df = pd.read_csv(url)
        # Sütun isimlerini temizle (küçük harf, boşluksuz)
        df.columns = df.columns.str.strip().str.lower()
        
        # --- SÜTUN EŞLEŞTİRME ---
        # Veri setindeki olası isimleri standart isimlere çeviriyoruz
        col_map = {
            'Name': ['name', 'player', 'full name', 'ad soyad'],
            'Club': ['club', 'team', 'current club', 'takim', 'kulup'],
            'Position': ['position', 'pos', 'bp', 'mevki'],
            'Overall': ['overall', 'ova', 'rating', 'guc'],
            'Potential': ['potential', 'pot', 'potansiyel'],
            'Age': ['age', 'yas'],
            'Value': ['value', 'market value', 'deger'],
            'Wage': ['wage', 'salary', 'maas']
        }

        # Mevcut sütunları tara ve eşleştir
        rename_dict = {}
        for target, keywords in col_map.items():
            for col in df.columns:
                # Eğer sütun ismi anahtar kelimelerden birini içeriyorsa
                if any(k in col for k in keywords):
                    if target not in rename_dict.values(): # Zaten atanmamışsa
                        rename_dict[col] = target
                        break
        
        # İsimleri değiştir
        df.rename(columns=rename_dict, inplace=True)

        # --- EKSİK SÜTUN GARANTİSİ (HATA ÖNLEYİCİ) ---
        # Eğer eşleşme sonrası hala 'Club' vb. yoksa, boş oluştur.
        required_cols = ['Name', 'Club', 'Position', 'Overall', 'Potential', 'Age', 'Value', 'Wage']
        for req in required_cols:
            if req not in df.columns:
                if req in ['Overall', 'Potential', 'Age', 'Value', 'Wage']:
                    df[req] = 0 # Sayısalları 0 yap
                else:
                    df[req] = "Bilinmiyor" # Metinleri 'Bilinmiyor' yap

        # --- İSİM TEMİZLİĞİ (ID NO HATASI İÇİN) ---
        # Eğer Name sütunu sayısal ise (ID sütunu karışmışsa), temizle
        df['Name'] = df['Name'].astype(str)
        
        # Clean Name oluştur
        def normalize_name(text):
            if not isinstance(text, str): return ""
            return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

        df['Clean_Name'] = df['Name'].apply(normalize_name)
        
        # Sayısal Değerleri Temizle (Örn: "€100M" -> 100000000)
        for col in ['Value', 'Wage']:
            if df[col].dtype == 'object':
                df[col] = (df[col].astype(str).str.replace('€', '')
                                             .str.replace('£', '')
                                             .str.replace('K', '000')
                                             .str.replace('M', '000000')
                                             .str.replace('.', '')
                                             .str.extract('(\d+)').astype(float))
        
        # Feature Sütunları
        feature_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage']
        df[feature_cols] = df[feature_cols].fillna(0) # NaN garantisi
        
        return df, feature_cols

    except Exception as e:
        return None, str(e)

# Yüklemeyi Başlat
with st.spinner('Saha taranıyor...'):
    df, features_or_error = load_data_robust()

# Hata Kontrolü
if df is None:
    st.error(f"❌ Veri yükleme hatası: {features_or_error}")
    st.stop()
else:
    feature_cols = features_or_error

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
    # Güvenli erişim için .get() kullanıyoruz
    pot = float(player.get('Potential', 0))
    ovr = float(player.get('Overall', 0))
    val = float(player.get('Value', 0))
    age = float(player.get('Age', 0))

    if pot - ovr >= 5:
        advice.append(f"📈 **YATIRIMLIK:** Oyuncu +{int(pot - ovr)} puan daha gelişebilir.")
    
    if age <= 21:
        advice.append(f"👶 **GENÇ YETENEK:** Henüz {int(age)} yaşında.")
    elif age >= 33:
        advice.append("⚠️ **RİSKLİ YAŞ:** Fiziksel düşüş yaşayabilir.")
        
    if ovr > 85 and val < 50000000:
        advice.append("🔥 **FIRSAT:** Gücüne göre piyasa değeri uygun.")

    if not advice:
        advice.append("✅ **STABİL:** Standart profil.")
    return advice

def find_similar(df, target, features):
    # Pozisyon Kilidi
    target_pos = target.get('Position', None)
    
    if target_pos:
        pool = df[df['Position'] == target_pos].copy()
    else:
        pool = df.copy()
    
    if len(pool) < 5:
        pool = df.copy()
    
    scaler = StandardScaler()
    X = pool[features]
    X_scaled = scaler.fit_transform(X)
    
    k = min(len(pool), 11)
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_scaled)
    
    target_vec = scaler.transform(target[features].to_frame().T)
    distances, indices = knn.kneighbors(target_vec)
    
    results = []
    for i, idx in enumerate(indices[0][1:]):
        n = pool.iloc[idx]
        score = max(0, 100 - (distances[0][i+1] * 10)) 
        
        # Etiketler (Güvenli Erişim)
        tag = "Benzer"
        try:
            if n['Value'] < target['Value'] * 0.7: tag = "💰 Daha Ucuz"
            elif n['Overall'] > target['Overall']: tag = "🏆 Daha Güçlü"
            elif n['Age'] < target['Age'] - 3: tag = "👶 Genç"
        except:
            pass
        
        results.append({
            "Oyuncu": n.get('Name', 'Bilinmiyor'),
            "Kulüp": n.get('Club', '-'),
            "Yaş": int(n.get('Age', 0)),
            "Güç": int(n.get('Overall', 0)),
            "Değer": f"€{n.get('Value', 0):,.0f}",
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
        else:
            if msg: st.info(msg)
            
            # --- OYUNCU KARTI (GÜVENLİ ERİŞİM) ---
            # KeyError hatasını engelleyen kısım burası: .get() kullanımı
            p_name = player.get('Name', 'İsimsiz')
            p_club = player.get('Club', 'Takımsız')
            p_pos = player.get('Position', '-')
            p_ovr = int(player.get('Overall', 0))
            p_age = int(player.get('Age', 0))
            p_val = float(player.get('Value', 0))

            st.subheader(f"{p_name.upper()} ({p_club})")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mevki", p_pos)
            c2.metric("Güç", p_ovr)
            c3.metric("Yaş", p_age)
            c4.metric("Değer", f"€{p_val:,.0f}")
            
            # --- AI TAVSİYESİ ---
            st.markdown("### 🤖 ANALİST RAPORU")
            for advice in get_advice(player):
                st.markdown(f"> {advice}")
            
            # --- BENZER OYUNCULAR ---
            st.markdown("---")
            st.subheader(f"🔄 {p_name} ALTERNATİFLERİ")
            
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
