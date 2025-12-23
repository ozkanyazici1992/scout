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
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3 { color: #00E5FF !important; font-family: 'Courier New', sans-serif; text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.3); }
    .stTextInput>div>div>input { background-color: #161B22; color: #00E5FF; border: 1px solid #00E5FF; }
    .stButton>button { background-color: #008B8B; color: white; border: none; border-radius: 5px; font-weight: bold; }
    .stButton>button:hover { background-color: #00E5FF; color: #000000; box-shadow: 0px 0px 15px #00E5FF; }
    div[data-testid="stMetric"] { background-color: #161B22; border: 1px solid #30363D; border-top: 3px solid #00E5FF; padding: 10px; border-radius: 5px; }
    div[data-testid="stMetricValue"] { color: #00E5FF !important; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VERİ YÜKLEME - DİNAMİK SÜTUN TESPİTİ
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_map_data():
    """Veriyi yükle ve sütun isimlerini otomatik eşleştir"""
    file_id = '1MUbla2YNYsd7sq61F8QL4OBnitw8tsEE'
    url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
    
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # Sütun mapping - Gerçek sütun isimlerini bul
        col_map = {}
        all_cols = df.columns.tolist()
        
        # İsim sütunu
        for col in all_cols:
            if 'name' in col.lower() and 'short' not in col.lower():
                # İçeriği kontrol et
                try:
                    sample = str(df[col].dropna().iloc[0])
                    if not sample.isdigit():
                        col_map['Name'] = col
                        break
                except:
                    pass
        
        # Diğer sütunları bul
        col_mappings = {
            'Club': ['club', 'team'],
            'Position': ['position', 'pos'],
            'Overall': ['overall', 'ovr'],
            'Potential': ['potential', 'pot'],
            'Age': ['age'],
            'Value': ['value', 'market value'],
            'Wage': ['wage', 'salary'],
            'Height': ['height'],
            'Weight': ['weight'],
        }
        
        for target_col, search_terms in col_mappings.items():
            for col in all_cols:
                if any(term in col.lower() for term in search_terms):
                    col_map[target_col] = col
                    break
        
        # Eğer isim bulunamadıysa ikinci sütunu kullan
        if 'Name' not in col_map:
            col_map['Name'] = all_cols[1] if len(all_cols) > 1 else all_cols[0]
        
        # Standart isimlerle yeni dataframe oluştur
        df_clean = pd.DataFrame()
        
        for std_name, orig_name in col_map.items():
            if orig_name in df.columns:
                df_clean[std_name] = df[orig_name]
        
        # Eksik temel sütunları varsayılan değerle ekle
        required_cols = ['Name', 'Club', 'Position', 'Overall', 'Age']
        for col in required_cols:
            if col not in df_clean.columns:
                df_clean[col] = 'N/A' if col in ['Club', 'Position'] else 50
        
        # Temizlenmiş isim sütunu ekle
        df_clean['Clean_Name'] = df_clean['Name'].astype(str).apply(
            lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
        )
        
        # Sayısal sütunları düzelt
        numeric_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage', 'Height', 'Weight']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Feature listesi oluştur
        feature_cols = [col for col in numeric_cols if col in df_clean.columns]
        
        debug_info = {
            'total_rows': len(df_clean),
            'columns_found': list(df_clean.columns),
            'column_mapping': col_map,
            'sample_names': df_clean['Name'].head(10).tolist()
        }
        
        return df_clean, feature_cols, debug_info
        
    except Exception as e:
        return None, None, {'error': str(e)}

# Veriyi yükle
with st.spinner('🔄 Sistem başlatılıyor...'):
    df, features, debug = load_and_map_data()

if df is None:
    st.error("❌ Veri yüklenemedi!")
    st.json(debug)
    st.stop()

# Debug bilgisi
with st.expander("🛠️ Sistem Bilgisi (Debug)"):
    st.write(f"**Toplam Oyuncu:** {debug['total_rows']}")
    st.write(f"**Bulunan Sütunlar:** {', '.join(debug['columns_found'])}")
    st.write("**Örnek İsimler:**")
    for name in debug['sample_names'][:5]:
        st.write(f"- {name}")

# -----------------------------------------------------------------------------
# ANALİZ FONKSİYONLARI
# -----------------------------------------------------------------------------
def find_player(df, search_term):
    """Oyuncu ara - esnek eşleştirme"""
    if not search_term or len(search_term) < 2:
        return None, "Lütfen en az 2 karakter girin."
    
    clean_search = unicodedata.normalize('NFKD', search_term).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    # Tam içerme araması
    mask = df['Clean_Name'].str.contains(clean_search, na=False, case=False, regex=False)
    matches = df[mask]
    
    if not matches.empty:
        # En yüksek Overall'e göre sırala
        if 'Overall' in matches.columns:
            best = matches.sort_values('Overall', ascending=False).iloc[0]
        else:
            best = matches.iloc[0]
        return best, None
    
    # Benzer isim ara
    all_names = df['Clean_Name'].dropna().unique().tolist()
    similar = difflib.get_close_matches(clean_search, all_names, n=3, cutoff=0.4)
    
    if similar:
        suggestions = []
        for sim in similar:
            player = df[df['Clean_Name'] == sim].iloc[0]
            suggestions.append(player['Name'])
        return None, f"Bulunamadı. Şunları mı demek istediniz: {', '.join(suggestions)}?"
    
    return None, f"'{search_term}' bulunamadı."

def get_advice(player):
    """Oyuncu için AI tavsiyesi"""
    advice = []
    
    # Potansiyel analizi
    if 'Potential' in player and 'Overall' in player:
        try:
            diff = float(player['Potential']) - float(player['Overall'])
            if diff >= 5:
                advice.append(f"📈 **YATIRIMLIK:** +{int(diff)} puan gelişim potansiyeli var.")
            elif diff < 2:
                advice.append(f"⭐ **ZİRVEDE:** Potansiyelinin zirvesinde.")
        except:
            pass
    
    # Yaş analizi
    if 'Age' in player:
        try:
            age = int(player['Age'])
            if age < 23:
                advice.append(f"👶 **GENÇ YETENEK:** {age} yaşında, geleceğe yatırım.")
            elif age > 32:
                advice.append(f"🎯 **DENEYİMLİ:** {age} yaşında, kısa vadeli çözüm.")
        except:
            pass
    
    # Değer analizi
    if 'Value' in player:
        try:
            value = float(player['Value'])
            if value < 1000000:
                advice.append("💰 **EKONOMİK:** Düşük maliyetli alternatif.")
            elif value > 50000000:
                advice.append("💎 **YILDIZ:** Yüksek değerli oyuncu.")
        except:
            pass
    
    if not advice:
        advice.append("✅ **STABİL PROFIL:** Standart özelliklere sahip.")
    
    return advice

def find_similar(df, target, feature_cols, n=10):
    """Benzer oyuncular bul"""
    if 'Position' not in target or pd.isna(target['Position']):
        return None
    
    # Aynı pozisyondaki oyuncular
    position = target['Position']
    pool = df[df['Position'] == position].copy()
    
    if len(pool) < n + 1:
        pool = df.copy()  # Yeterli oyuncu yoksa hepsini al
    
    # Kullanılabilir feature'ları filtrele
    valid_features = [f for f in feature_cols if f in pool.columns and f in target.index]
    
    if len(valid_features) < 2:
        return None
    
    try:
        # Scaling ve KNN
        scaler = StandardScaler()
        X = pool[valid_features].fillna(pool[valid_features].median())
        X_scaled = scaler.fit_transform(X)
        
        target_vector = scaler.transform(target[valid_features].to_frame().T)
        
        knn = NearestNeighbors(n_neighbors=min(n+1, len(pool)), metric='euclidean')
        knn.fit(X_scaled)
        
        distances, indices = knn.kneighbors(target_vector)
        
        results = []
        for i, idx in enumerate(indices[0][1:n+1]):
            p = pool.iloc[idx]
            similarity = max(0, 100 - distances[0][i+1] * 5)
            
            # Etiket belirle
            tag = "⚖️ Benzer"
            try:
                if 'Overall' in p and float(p['Overall']) > float(target['Overall']):
                    tag = "⬆️ Daha İyi"
                elif 'Age' in p and float(p['Age']) < float(target['Age']) - 3:
                    tag = "🌱 Genç"
                elif 'Value' in p and float(p.get('Value', 0)) < float(target.get('Value', 999999)) * 0.7:
                    tag = "💰 Ucuz"
            except:
                pass
            
            results.append({
                "Oyuncu": str(p['Name']),
                "Takım": str(p.get('Club', 'N/A')),
                "Yaş": int(p.get('Age', 0)),
                "Güç": int(p.get('Overall', 0)),
                "Potansiyel": int(p.get('Potential', 0)) if 'Potential' in p else "-",
                "Benzerlik": f"{similarity:.0f}%",
                "Durum": tag
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Benzerlik hesaplama hatası: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# ARAYÜZ
# -----------------------------------------------------------------------------
st.title("💎 TURQUOISE SCOUT AI")
st.markdown("*Futbolcu analizi ve yapay zeka destekli alternatif öneri sistemi*")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    player_name = st.text_input("🔍 Oyuncu Adı Girin:", placeholder="Örn: Messi, Ronaldo, Haaland...")
with col2:
    st.write("")
    st.write("")
    search_btn = st.button("ANALİZ ET", use_container_width=True)

if search_btn and player_name:
    player, message = find_player(df, player_name)
    
    if player is None:
        st.error(f"❌ {message}")
    else:
        if message:
            st.info(message)
        
        # Oyuncu başlığı
        st.markdown(f"## {str(player['Name']).upper()}")
        club = player.get('Club', 'Bilinmiyor')
        st.markdown(f"**🏟️ Takım:** {club}")
        
        # Metrikler
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Mevki", str(player.get('Position', '-')))
        col2.metric("Güç", int(player.get('Overall', 0)))
        
        if 'Potential' in player:
            col3.metric("Potansiyel", int(player.get('Potential', 0)))
        else:
            col3.metric("Potansiyel", "-")
        
        col4.metric("Yaş", int(player.get('Age', 0)))
        
        if 'Value' in player:
            val = int(player.get('Value', 0))
            col5.metric("Değer (£)", f"{val:,}" if val > 0 else "-")
        else:
            col5.metric("Değer", "-")
        
        # AI Tavsiyesi
        st.markdown("---")
        st.markdown("### 🤖 AI ANALİST TAVSİYESİ")
        advices = get_advice(player)
        for adv in advices:
            st.markdown(f"> {adv}")
        
        # Benzer Oyuncular
        st.markdown("---")
        st.markdown(f"### 🔄 {str(player['Name']).upper()} YERİNE ALINABİLECEK EN İYİ 10 ALTERNATİF")
        
        similar_df = find_similar(df, player, features, n=10)
        
        if similar_df is not None and not similar_df.empty:
            st.dataframe(
                similar_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("⚠️ Bu oyuncu için benzer alternatif bulunamadı.")

elif search_btn:
    st.warning("⚠️ Lütfen bir oyuncu adı girin.")

st.markdown("---")
st.caption("💎 Turquoise Scout AI - Powered by Machine Learning")
