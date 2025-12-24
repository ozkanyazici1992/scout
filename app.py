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
# 1. SAYFA VE AYARLAR
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Turquoise Scout AI",
    page_icon="💎",
    layout="wide"
)

# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 2. FONKSİYONLAR
# -----------------------------------------------------------------------------

def normalize_text(text):
    if pd.isna(text) or text == "": return ""
    text = str(text)
    text = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()

# Streamlit Cache: Veriyi sadece bir kez indirir ve hafızada tutar
@st.cache_data(show_spinner=True)
def load_data_robust():
    file_id = '1nl2hcZP6GltTtjPFzjb8KuOmzRqDLjf6'
    url = f'https://drive.google.com/uc?id={file_id}&export=download'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            csv_content = io.StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_content)
        else:
            st.error(f"⚠️ İndirme başarısız (Kod: {response.status_code}).")
            return None, None

        # Sütunları temizle
        df.columns = df.columns.str.strip().str.lower()

        # Sütun Eşleştirme
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

        # Eksikleri doldur
        for col in col_map.keys():
            if col not in df.columns:
                df[col] = 0 if col not in ['Name', 'Club', 'Position', 'Preferred Foot'] else 'Bilinmiyor'

        # İsim Sütunu Temizliği
        df['Name'] = df['Name'].astype(str)
        if df['Name'].iloc[0].replace('.', '').isdigit():
            obj_cols = df.select_dtypes(include=['object']).columns
            for c in obj_cols:
                if not str(df[c].iloc[0]).replace('.', '').isdigit() and len(str(df[c].iloc[0])) > 2:
                    df['Name'] = df[c]
                    break

        df['Clean_Name'] = df['Name'].apply(normalize_text)

        # Para Birimi Temizliği
        for col in ['Value', 'Wage']:
            if df[col].dtype == 'object':
                df[col] = (df[col].astype(str).str.replace('€', '').str.replace('£', '')
                           .str.replace('K', '000').str.replace('M', '000000')
                           .str.replace('.', '').str.extract('(\d+)').astype(float))

        # Sayısal Temizlik
        num_cols = ['Overall', 'Potential', 'Age', 'Value', 'Wage', 'Finishing', 'Heading', 'Speed']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df, ['Overall', 'Potential', 'Age', 'Value', 'Wage']

    except Exception as e:
        st.error(f"❌ Kritik Hata: {e}")
        return None, None

def yazili_analiz_uret(p):
    analizler = []
    ayak = str(p.get('Preferred Foot', '')).lower()
    if 'left' in ayak: analizler.append("🔸 Sol ayaklı.")
    elif 'right' in ayak: analizler.append("🔸 Sağ ayaklı.")

    if float(p.get('Finishing', 0)) > 82: analizler.append("🎯 Bitirici forvet.")
    if float(p.get('Heading', 0)) > 80: analizler.append("🦅 Hava hakimiyeti yüksek.")
    if float(p.get('Speed', 0)) > 85: analizler.append("⚡ Çok süratli.")

    pot = float(p.get('Potential', 0))
    ovr = float(p.get('Overall', 0))
    if pot - ovr >= 3: analizler.append(f"💎 Gelişime açık (Pot: {int(pot)}).")

    if not analizler: analizler.append("ℹ️ Dengeli profil.")
    return " ".join(analizler)

# -----------------------------------------------------------------------------
# 3. ANA UYGULAMA AKIŞI
# -----------------------------------------------------------------------------

def main():
    # Başlık
    st.title("💎 Turquoise Scout AI")
    st.markdown("**Veri Odaklı Futbolcu Analiz ve Benzer Oyuncu Bulma Aracı**")
    st.markdown("---")

    # Veriyi Yükle
    with st.spinner("Veri tabanı Google Drive'dan çekiliyor..."):
        df, features = load_data_robust()

    if df is None:
        st.stop()

    # Kenar Çubuğu (Sidebar)
    st.sidebar.header("🔍 Arama Paneli")
    
    # Selectbox: Kullanıcı yazarken filtreleme yapar, çok daha pratiktir.
    # Benzersiz isimleri alıp sıralıyoruz
    player_list = sorted(df['Name'].unique().tolist())
    selected_player_name = st.sidebar.selectbox("Oyuncu Seçiniz:", player_list, index=None, placeholder="Örn: Messi")

    if selected_player_name:
        # Seçilen oyuncuyu bul
        target = df[df['Name'] == selected_player_name].iloc[0]
        
        # --- ANA GÖVDE ---
        
        # 1. Bölüm: Oyuncu Kartı
        col1, col2, col3, col4 = st.columns(4)
        
        val_formatted = f"€{target.get('Value', 0):,.0f}"
        wage_formatted = f"€{target.get('Wage', 0):,.0f}"
        
        col1.metric("Güç (Overall)", int(target['Overall']), delta=int(target['Potential'] - target['Overall']))
        col2.metric("Piyasa Değeri", val_formatted)
        col3.metric("Yaş", int(target['Age']))
        col4.metric("Mevki", target['Position'])

        st.info(f"📋 **Analiz Özeti:** {yazili_analiz_uret(target)}")
        
        # Detaylar
        with st.expander(f"{target['Name']} - Detaylı Özellikler"):
            st.json(target[['Club', 'Preferred Foot', 'Speed', 'Finishing', 'Heading']].to_dict())

        st.markdown("---")
        st.subheader("🔄 Benzer Profildeki Oyuncular (Alternatifler)")

        # 2. Bölüm: KNN Analizi
        target_pos = target.get('Position', None)
        pool = df[df['Position'] == target_pos].copy()
        
        # Havuz çok küçükse tüm veriyi kullan
        if len(pool) < 2: 
            pool = df.copy()

        # Model Kurulumu
        scaler = StandardScaler()
        X = pool[features]
        X_scaled = scaler.fit_transform(X)

        knn = NearestNeighbors(n_neighbors=min(11, len(pool)), metric='euclidean')
        knn.fit(X_scaled)

        target_vec = scaler.transform(target[features].to_frame().T)
        distances, indices = knn.kneighbors(target_vec)

        # Sonuçları Hazırla
        results = []
        for i, idx in enumerate(indices[0][1:]): # İlk sonuç kendisi olduğu için atlıyoruz
            n = pool.iloc[idx]
            dist = distances[0][i + 1]
            score = max(0, 100 - (dist * 10)) # Basit bir benzerlik skoru

            tag = "Normal"
            val = target.get('Value', 0)
            
            if n['Value'] < val * 0.6:
                tag = "💰 Kelepir"
            elif n['Overall'] > target['Overall']:
                tag = "⭐ Daha İyi"

            results.append({
                "Oyuncu": n['Name'],
                "Takım": n.get('Club', '-'),
                "Mevki": n.get('Position', '-'),
                "Güç": int(n.get('Overall', 0)),
                "Potansiyel": int(n.get('Potential', 0)),
                "Değer": f"€{n.get('Value', 0):,.0f}",
                "Benzerlik": f"%{score:.0f}",
                "Durum": tag
            })

        # Tabloyu Göster
        if results:
            res_df = pd.DataFrame(results)
            
            # Renklendirme fonksiyonu (Opsiyonel görselleştirme)
            def highlight_bargain(row):
                if "Kelepir" in row['Durum']:
                    return ['background-color: #d4edda'] * len(row)
                elif "Daha İyi" in row['Durum']:
                    return ['background-color: #cce5ff'] * len(row)
                else:
                    return [''] * len(row)

            st.dataframe(
                res_df.style.apply(highlight_bargain, axis=1),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Yeterli veri bulunamadığı için benzer oyuncu önerilemiyor.")

    else:
        st.info("👈 Lütfen sol menüden veya yukarıdan bir oyuncu seçiniz.")

if __name__ == "__main__":
    main()
