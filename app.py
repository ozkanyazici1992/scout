import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import requests
import io

# -----------------------------------------------------------------------------
# 1. SAYFA VE TASARIM AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Futbolist AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')

# --- ÖZEL CSS (GELİŞMİŞ KART TASARIMI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    /* Ana Arka Plan */
    .stApp { background-color: #E0F7FA; }
    
    /* Genel Yazı Fontu ve Rengi */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, li {
        font-family: 'Montserrat', sans-serif;
        color: #004D40 !important;
    }

    /* Başlık Stilleri */
    .main-title {
        text-align: center; font-size: 3.5rem; font-weight: 900;
        color: #006064 !important; letter-spacing: -2px;
        text-transform: uppercase; text-shadow: 2px 2px 0px #ffffff;
        margin-top: 20px;
    }
    .sub-title {
        text-align: center; font-size: 1.2rem; color: #00838F !important;
        margin-bottom: 35px; font-weight: 500;
    }

    /* Arama Kutusu */
    .stTextInput > div > div > input {
        text-align: center; font-size: 1.3rem; padding: 12px;
        border-radius: 30px; border: 2px solid #4DD0E1;
        background-color: #ffffff; color: #006064 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #006064; box-shadow: 0 0 15px rgba(0, 96, 100, 0.2);
    }

    /* --- OYUNCU KARTI TASARIMI (YENİ) --- */
    .player-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0, 77, 64, 0.08); /* Hafif gölge */
        border: 1px solid #B2DFDB;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .player-card:hover {
        transform: translateY(-5px);
        border-color: #009688;
        box-shadow: 0 8px 15px rgba(0, 77, 64, 0.15);
    }
    
    .card-header {
        font-size: 1.1rem; font-weight: 800; color: #004D40 !important;
        margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .card-sub {
        font-size: 0.8rem; color: #546E7A !important; margin-bottom: 12px;
        height: 35px; display: flex; align-items: center; justify-content: center; line-height: 1.1;
    }
    
    /* Kart İçi İstatistik Kutucukları */
    .stat-row {
        display: flex; justify-content: space-between; margin-bottom: 6px;
        background-color: #F0F4C3; border-radius: 6px; padding: 4px 8px;
    }
    .stat-label { font-size: 0.85rem; font-weight: 600; color: #558B2F !important; }
    .stat-val { font-size: 0.9rem; font-weight: 800; color: #33691E !important; }

    .price-tag {
        background-color: #E0F2F1; border-radius: 6px; padding: 4px 8px; margin-bottom: 8px;
        font-size: 0.9rem; font-weight: 700; color: #00695C !important;
    }

    .match-badge {
        display: inline-block; padding: 5px 15px; border-radius: 20px;
        color: white !important; font-weight: bold; font-size: 0.85rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Metrikler ve Toast */
    div[data-testid="stMetricValue"] { color: #000000 !important; }
    div[data-testid="stMetricLabel"] { color: #006064 !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ VE FONKSİYONLAR
# -----------------------------------------------------------------------------

def normalize_text(text):
    if pd.isna(text) or text == "": return ""
    text = str(text)
    text = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()

@st.cache_data(show_spinner=False)
def load_data_robust():
    file_id = '1nl2hcZP6GltTtjPFzjb8KuOmzRqDLjf6'
    url = f'https://drive.google.com/uc?id={file_id}&export=download'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            csv_content = io.StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_content)
        else:
            return None, None

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
        # Hatalı isim sütunu düzeltmesi
        if str(df['Name'].iloc[0]).replace('.', '').isdigit():
            obj_cols = df.select_dtypes(include=['object']).columns
            for c in obj_cols:
                if not str(df[c].iloc[0]).replace('.', '').isdigit() and len(str(df[c].iloc[0])) > 2:
                    df['Name'] = df[c]
                    break

        df['Clean_Name'] = df['Name'].apply(normalize_text)

        for col in ['Value', 'Wage']:
            if df[col].dtype == 'object':
                df[col] = (df[col].astype(str).str.replace('€', '').str.replace('£', '')
                           .str.replace('K', '000').str.replace('M', '000000')
                           .str.replace('.', '').str.extract('(\d+)').astype(float))

        num_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage', 'Finishing', 'Heading', 'Speed']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df, ['Overall', 'Potential', 'Age', 'Value', 'Wage']

    except Exception:
        return None, None

def yazili_analiz_uret(p):
    analizler = []
    if float(p.get('Finishing', 0)) > 82: analizler.append("🎯 Keskin Nişancı")
    if float(p.get('Heading', 0)) > 80: analizler.append("🦅 Hava Hakimi")
    if float(p.get('Speed', 0)) > 85: analizler.append("⚡ Çok Hızlı")
    
    pot = float(p.get('Potential', 0))
    ovr = float(p.get('Overall', 0))
    if pot - ovr >= 4: analizler.append(f"💎 Yüksek Potansiyel")
    if float(p.get('Age', 0)) < 21 and ovr > 75: analizler.append("🌟 Wonderkid")
    
    if not analizler: return "Standart Profil"
    return "   •   ".join(analizler)

def find_smart_match(df, user_input):
    clean_input = normalize_text(user_input)
    matches = df[df['Clean_Name'].str.contains(clean_input, na=False)]
    
    if not matches.empty:
        return matches.sort_values(by='Overall', ascending=False).iloc[0], "Tam"
    
    all_names = df['Clean_Name'].unique().tolist()
    close_matches = difflib.get_close_matches(clean_input, all_names, n=1, cutoff=0.5)
    
    if close_matches:
        found_name_clean = close_matches[0]
        target_row = df[df['Clean_Name'] == found_name_clean].sort_values(by='Overall', ascending=False).iloc[0]
        return target_row, "Tahmin"
        
    return None, None

def format_money(val):
    if val >= 1000000: return f"€{val/1000000:.1f}M"
    elif val >= 1000: return f"€{val/1000:.0f}K"
    else: return "€0"

# -----------------------------------------------------------------------------
# 3. ANA UYGULAMA AKIŞI
# -----------------------------------------------------------------------------
def main():
    # --- HEADER ---
    st.markdown('<div class="main-title">FUTBOLIST AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Yapay Zeka Destekli Scout Analizi</div>', unsafe_allow_html=True)

    # Veri Yükleme
    df, features = load_data_robust()
    if df is None:
        st.error("Veri bağlantısı kurulamadı. Lütfen sayfayı yenileyin.")
        st.stop()

    # --- MERKEZİ ARAMA KUTUSU ---
    c1, c2, c3 = st.columns([1, 2, 1])
    
    target = None
    search_query = ""

    with c2:
        search_query = st.text_input("", placeholder="Oyuncu ara... (Örn: Icardi, Messi, Arda Güler)", label_visibility="collapsed")
        
        if search_query:
            target, match_type = find_smart_match(df, search_query)
            if target is None:
                st.toast(f"❌ '{search_query}' bulunamadı.", icon="⚠️")
            elif match_type == "Tahmin":
                st.toast(f"✅ Düzeltildi: {target['Name']}", icon="✨")

    # --- SONUÇ EKRANI ---
    if target is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # OYUNCU KARTI (HEADER)
        with st.container():
            col_img, col_info, col_stats = st.columns([1, 2, 2])
            
            with col_info:
                st.subheader(f"🦁 {target['Name']}")
                st.markdown(f"**{target['Club']}** | {target['Position']}")
                st.markdown(f"_{int(target.get('Age', 0))} Yaş, {str(target.get('Preferred Foot', '-')).title()} Ayak_")
                
                tags = yazili_analiz_uret(target)
                if tags:
                    st.success(f"💡 {tags}")

            with col_stats:
                m1, m2 = st.columns(2)
                m1.metric("Genel Güç", int(target['Overall']), delta=int(target['Potential'] - target['Overall']))
                m2.metric("Piyasa Değeri", format_money(target.get('Value', 0)))
                
                st.progress(int(target['Overall'])/100, text="Potansiyel Doluluk Oranı")

        # --- AI BENZERLİK ANALİZİ (YENİ KART TASARIMI) ---
        st.divider()
        st.markdown("#### 🧬 Futbolist AI Scout Önerileri")
        
        target_pos = target.get('Position', None)
        # Sadece aynı veya benzer mevkileri al (Veri setine göre esnetilebilir)
        pool = df[df['Position'] == target_pos].copy() if target_pos else df.copy()
        if len(pool) < 6: pool = df.copy()

        # KNN
        scaler = StandardScaler()
        X = pool[features]
        X_scaled = scaler.fit_transform(X)

        knn = NearestNeighbors(n_neighbors=min(6, len(pool)), metric='euclidean')
        knn.fit(X_scaled)

        target_vec = scaler.transform(target[features].to_frame().T)
        distances, indices = knn.kneighbors(target_vec)

        # Kartları Göster
        cols = st.columns(5)
        
        suggestions = indices[0][1:6] 
        suggestion_dists = distances[0][1:6]

        for i, idx in enumerate(suggestions):
            n = pool.iloc[idx]
            dist = suggestion_dists[i]
            score = max(0, 100 - (dist * 10))
            
            # Renk ve Stil Mantığı
            if score >= 90: badge_color = "#43A047" # Canlı Yeşil
            elif score >= 80: badge_color = "#FB8C00" # Turuncu
            else: badge_color = "#E53935" # Kırmızı

            val_str = format_money(n['Value'])
            club_str = n.get('Club', 'Bilinmiyor')
            if len(str(club_str)) > 15: club_str = str(club_str)[:13] + ".."

            # --- DÜZELTME: HTML KODLARI SOLA YASLANDI ---
            # Streamlit'in HTML'i kod bloğu sanmaması için girintiyi sildik.
            card_html = f"""<div class="player-card">
<div class="card-header">{n['Name']}</div>
<div class="card-sub">{club_str}<br>{n.get('Position','-')} • {int(n.get('Age',0))} Yaş</div>
<div class="stat-row">
<span class="stat-label">GÜÇ</span>
<span class="stat-val">{int(n['Overall'])}</span>
</div>
<div class="stat-row">
<span class="stat-label">POTANSİYEL</span>
<span class="stat-val">{int(n['Potential'])}</span>
</div>
<div class="price-tag">{val_str}</div>
<div class="match-badge" style="background-color: {badge_color}">%{score:.0f} UYUM</div>"""
                
            # Kelepir Kontrolü
            if n['Value'] > 0 and n['Value'] < target['Value'] * 0.6: 
                card_html += '<div style="margin-top:8px; font-size:0.8rem; color:#2E7D32; font-weight:800;">💰 FIRSAT</div>'
            
            card_html += "</div>" # Kapanış Div
            
            with cols[i]:
                st.markdown(card_html, unsafe_allow_html=True)

    elif not search_query:
        st.markdown(
            """
            <div style='text-align: center; color: #006064; margin-top: 100px; opacity: 0.8; font-weight: bold;'>
            Futbolist AI Database v2.0 • Powered by Python
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
