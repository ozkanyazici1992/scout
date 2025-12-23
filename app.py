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
    .expander { background-color: #161B22; }
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
        
        # Tüm sütunları kontrol et
        debug_info = {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'first_row': {}
        }
        
        # İsim sütununu akıllıca bul
        name_col = None
        
        # Öncelik 1: "Name" içeren sütunlar
        for col in df.columns:
            col_lower = col.lower()
            if 'name' in col_lower and 'short' not in col_lower and 'long' not in col_lower:
                try:
                    first_vals = df[col].dropna().head(3).astype(str).tolist()
                    # Eğer ilk değerler sayı değilse ve boş değilse
                    if first_vals and not all(val.isdigit() for val in first_vals):
                        name_col = col
                        debug_info['name_column'] = col
                        debug_info['sample_names'] = first_vals
                        break
                except:
                    continue
        
        # Öncelik 2: İkinci sütun (genelde Name olur)
        if name_col is None and len(df.columns) > 1:
            name_col = df.columns[1]
            debug_info['name_column'] = f"{name_col} (2. sütun)"
            debug_info['sample_names'] = df[name_col].dropna().head(3).astype(str).tolist()
        
        # Öncelik 3: İlk sütun
        if name_col is None:
            name_col = df.columns[0]
            debug_info['name_column'] = f"{name_col} (1. sütun)"
            debug_info['sample_names'] = df[name_col].dropna().head(3).astype(str).tolist()
        
        df['Name'] = df[name_col].astype(str).str.strip()
        
        # Temizlenmiş isim sütunu
        def normalize_name(text):
            text = str(text)
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
            return text.lower().strip()
        
        df['Clean_Name'] = df['Name'].apply(normalize_name)
        
        # Work Rate Score
        def work_rate_score(wr):
            if not isinstance(wr, str):
                return 2
            scores = {'Low': 1, 'Medium': 2, 'High': 3}
            parts = str(wr).split('/')
            if len(parts) == 2:
                att = scores.get(parts[0].strip(), 1)
                def_val = scores.get(parts[1].strip(), 1)
                return att + def_val
            return 2
        
        if 'Work Rate' in df.columns:
            df['Work_Rate_Score'] = df['Work Rate'].apply(work_rate_score)
        else:
            df['Work_Rate_Score'] = 2
        
        # Feature seçimi - mevcut sütunları kontrol et
        possible_features = ['Overall', 'Potential', 'Value(£)', 'Wage(£)', 'Age', 
                           'International Reputation', 'Skill Moves', 'Weak Foot', 
                           'Special', 'Work_Rate_Score', 'Height(cm.)', 'Weight(lbs.)']
        
        feature_cols = [f for f in possible_features if f in df.columns]
        
        # Eksik değerleri doldur
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        
        debug_info['features_found'] = feature_cols
        debug_info['total_players'] = len(df)
        
        return df, feature_cols, debug_info
    except Exception as e:
        return None, None, {'error': str(e)}

with st.spinner('🔄 Sistem başlatılıyor...'):
    df, feature_cols, debug_info = load_data()

if df is None:
    st.error("❌ Veri indirilemedi.")
    st.json(debug_info)
    st.stop()

# Debug bilgisi göster (isteğe bağlı)
with st.expander("🛠️ Sistem Bilgisi (Debug)"):
    st.write(f"**Toplam Oyuncu:** {debug_info.get('total_players', 0)}")
    st.write(f"**İsim Sütunu:** {debug_info.get('name_column', 'Bulunamadı')}")
    st.write(f"**Örnek İsimler:** {', '.join(debug_info.get('sample_names', [])[:5])}")
    st.write(f"**Kullanılan Özellikler:** {len(feature_cols)} adet")
    if st.checkbox("Tüm sütunları göster"):
        st.write(debug_info.get('columns', []))

# -----------------------------------------------------------------------------
# ANALİZ FONKSİYONLARI
# -----------------------------------------------------------------------------
def find_player(df, name_input):
    if not name_input or len(name_input) < 2:
        return None, "Lütfen en az 2 karakter girin."
    
    # Temizle
    clean_input = unicodedata.normalize('NFKD', name_input).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    
    # Tam içerme araması
    exact_matches = df[df['Clean_Name'].str.contains(clean_input, na=False, regex=False, case=False)]
    
    if not exact_matches.empty:
        best = exact_matches.sort_values(by='Overall', ascending=False).iloc[0]
        return best, None
    
    # Benzer isim araması
    all_names = df['Clean_Name'].dropna().unique().tolist()
    close_matches = difflib.get_close_matches(clean_input, all_names, n=3, cutoff=0.4)
    
    if close_matches:
        suggestions = []
        for match in close_matches:
            player = df[df['Clean_Name'] == match].iloc[0]
            suggestions.append(player['Name'])
        
        return None, f"Bulunamadı. Şunları mı aramak istediniz: {', '.join(suggestions)}?"
    
    return None, f"'{name_input}' bulunamadı. Farklı bir isim deneyin."

def get_advice(player):
    advice = []
    
    # Fırsat analizi
    if 'Value(£)' in player and 'Release Clause(£)' in player:
        val = player.get('Value(£)', 0)
        clause = player.get('Release Clause(£)', 0)
        if clause > 0 and val > clause:
            kar = val - clause
            advice.append(f"🔥 **KELEPİR FIRSAT:** Serbest kalma bedeli değerinden £{kar:,.0f} düşük!")
    
    # Potansiyel analizi
    if 'Potential' in player and 'Overall' in player:
        pot = player.get('Potential', 0)
        ovr = player.get('Overall', 0)
        diff = pot - ovr
        if diff >= 5:
            advice.append(f"📈 **YATIRIMLIK OYUNCU:** +{diff:.0f} puan daha gelişebilir.")
        elif diff < 2:
            advice.append(f"⭐ **ZİRVEDE:** Potansiyelinin zirvesine ulaşmış.")
    
    # Sözleşme durumu
    if 'Contract Valid Until' in player:
        try:
            contract_year = float(player['Contract Valid Until'])
            if contract_year <= 2025:
                advice.append(f"⏳ **SÖZLEŞME BİTİYOR:** {int(contract_year)} yılında sona eriyor.")
        except:
            pass
    
    # Yaş analizi
    if 'Age' in player:
        age = player.get('Age', 25)
        if age < 23:
            advice.append(f"👶 **GENÇ YETENEK:** {int(age)} yaşında, gelecek vaat ediyor.")
        elif age > 32:
            advice.append(f"🎯 **DENEYİMLİ:** {int(age)} yaşında, kısa vadeli çözüm.")
    
    # Genel değerlendirme
    if not advice:
        advice.append("✅ **STABİL PROFIL:** Standart özelliklere sahip oyuncu.")
    
    return advice

def find_similar_players(df, target_player, features, n=10):
    if 'Position' not in target_player or pd.isna(target_player['Position']):
        return None
    
    target_pos = target_player['Position']
    pool = df[df['Position'] == target_pos].copy()
    
    if len(pool) < n + 1:
        pool = df.copy()  # Pozisyon yeterli değilse tüm oyuncuları kullan
    
    # Özellikleri hazırla
    valid_features = [f for f in features if f in pool.columns and f in target_player.index]
    
    if len(valid_features) < 3:
        return None
    
    try:
        scaler = StandardScaler()
        scaled_pool = scaler.fit_transform(pool[valid_features])
        
        knn = NearestNeighbors(n_neighbors=min(n+1, len(pool)), metric='euclidean')
        knn.fit(scaled_pool)
        
        target_vec = scaler.transform(target_player[valid_features].to_frame().T)
        distances, indices = knn.kneighbors(target_vec)
        
        results = []
        for i, idx in enumerate(indices[0][1:n+1]):
            player = pool.iloc[idx]
            similarity = max(0, 100 - (distances[0][i+1] * 5))
            
            # Durum etiketi
            tag = "⚖️ Benzer"
            try:
                if 'Value(£)' in player and player['Value(£)'] < target_player['Value(£)'] * 0.7:
                    tag = "💰 Ucuz"
                elif 'Overall' in player and player['Overall'] > target_player['Overall']:
                    tag = "⬆️ Daha İyi"
                elif 'Age' in player and player['Age'] < target_player['Age'] - 3:
                    tag = "🌱 Genç"
            except:
                pass
            
            results.append({
                "Oyuncu": player.get('Name', 'N/A'),
                "Takım": player.get('Club', 'N/A'),
                "Yaş": int(player.get('Age', 0)),
                "Güç": int(player.get('Overall', 0)),
                "Potansiyel": int(player.get('Potential', 0)) if 'Potential' in player else "-",
                "Değer (£)": f"{int(player.get('Value(£)', 0)):,}" if 'Value(£)' in player else "-",
                "Benzerlik": f"{similarity:.0f}%",
                "Durum": tag
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Benzer oyuncu arama hatası: {e}")
        return None

# -----------------------------------------------------------------------------
# ARAYÜZ
# -----------------------------------------------------------------------------
st.title("💎 TURQUOISE SCOUT AI")
st.markdown("*Futbolcu analizi ve yapay zeka destekli alternatif öneri sistemi*")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    search_name = st.text_input("🔍 Oyuncu Adı:", placeholder="Örn: Messi, Ronaldo, Haaland...")
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("ANALİZ ET", use_container_width=True)

if analyze_btn and search_name:
    player, msg = find_player(df, search_name)
    
    if player is None:
        st.error(f"❌ {msg}")
    else:
        if msg:
            st.info(msg)
        
        # Oyuncu Bilgileri
        st.markdown(f"## {player['Name'].upper()}")
        st.markdown(f"**🏟️ Takım:** {player.get('Club', 'Bilinmiyor')}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mevki", player.get('Position', '-'))
        col2.metric("Güç", int(player.get('Overall', 0)))
        col3.metric("Potansiyel", int(player.get('Potential', 0)) if 'Potential' in player else "-")
        col4.metric("Yaş", int(player.get('Age', 0)))
        if 'Value(£)' in player:
            col5.metric("Değer (£)", f"{int(player.get('Value(£)', 0)):,}")
        
        # AI Tavsiyesi
        st.markdown("---")
        st.markdown("### 🤖 AI ANALİST TAVSİYESİ")
        advices = get_advice(player)
        for adv in advices:
            st.markdown(f"> {adv}")
        
        # Benzer Oyuncular
        st.markdown("---")
        st.markdown(f"### 🔄 {player['Name']} YERİNE ALINABİLECEK EN İYİ 10 ALTERNATİF")
        
        similar_df = find_similar_players(df, player, feature_cols, n=10)
        
        if similar_df is not None and not similar_df.empty:
            st.dataframe(
                similar_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Benzerlik": st.column_config.TextColumn("Benzerlik"),
                }
            )
        else:
            st.warning("⚠️ Bu oyuncu için benzer alternatif bulunamadı.")

elif analyze_btn and not search_name:
    st.warning("⚠️ Lütfen bir oyuncu adı girin.")

# Alt bilgi
st.markdown("---")
st.caption("💎 Turquoise Scout AI - Powered by Machine Learning")
