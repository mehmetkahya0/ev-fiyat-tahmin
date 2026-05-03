# =============================================================================
# Makine Öğrenmesi Final Projesi
# Ev Fiyat Tahmin Sınıflandırması (İstanbul Konut Veri Seti)
# =============================================================================
# Veri Seti: https://www.kaggle.com/datasets/aselasel/house-price-dataset
# Açıklama: İstanbul'daki satılık evlerin bilgilerini ve fiyatlarını içeren
#           gerçek dünya veri seti üzerinde sınıflandırma analizi.
# =============================================================================

# --- 1. Kütüphanelerin İçe Aktarılması ---
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmadan grafik kaydetmek için
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# Görselleştirme ayarları
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

# =============================================================================
# 2. VERİ YÜKLEME
# =============================================================================
# Kaggle'dan veriyi otomatik indirme (kagglehub yüklü ise)
try:
    import kagglehub
    dataset_path = kagglehub.dataset_download("aselasel/house-price-dataset")
    csv_path = f"{dataset_path}/HouseData.csv"
    print(f"[INFO] Veri seti kagglehub ile indirildi: {csv_path}")
except Exception:
    # Eğer kagglehub yoksa, aynı klasördeki CSV dosyasını kullan
    csv_path = "HouseData.csv"
    print(f"[INFO] Yerel dosya kullanılıyor: {csv_path}")

df = pd.read_csv(csv_path)
print(f"\n{'='*60}")
print(f"VERİ SETİ YÜKLEME TAMAMLANDI")
print(f"Satır sayısı : {df.shape[0]}")
print(f"Sütun sayısı : {df.shape[1]}")
print(f"{'='*60}")

# =============================================================================
# 3. VERİ ÖN İŞLEME (Preprocessing)
# =============================================================================
print("\n--- 3.1 Gereksiz Sütunların Çıkarılması ---")

# Analiz için kullanılmayacak sütunlar (tarih, adres, sabit değer, yüksek NaN)
drop_cols = [
    "Unnamed: 0",       # Index sütunu
    "address",           # Adres (çok yüksek kardinalite, model için uygunsuz)
    "AdUpdateDate",      # İlan güncelleme tarihi
    # NOT: AdCreationDate enflasyon duzeltmesi icin gerekli, sonra silinecek
    "Category",          # Tüm değerler aynı (Satılık)
    "Type",              # Tüm değerler aynı (Konut)
    "PriceStatus",       # Fiyat durumu - belirsiz
    "TitleStatus",       # %61 eksik veri
    "MortgageStatus",    # %76 eksik veri
    "RentalIncome",      # %77 eksik veri
    "NumberOfBalconies",  # %79 eksik veri
    "BalconyType",       # %81 eksik veri
    "HallSquareMeters",  # %94 eksik veri
    "WCSquareMeters",    # %97 eksik veri
    "IsItVideoNavigable?",  # %78 eksik veri
    "Subscription",      # %82 eksik veri
    "BathroomSquareMeters",  # %96 eksik veri
    "BalconySquareMeters",   # %96 eksik veri
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
print(f"  Kalan sütun sayısı: {df.shape[1]}")

# -------------------------------------------------------------------------
# 3.2 Fiyat Sütununu Sayısal Formata Dönüştürme
# -------------------------------------------------------------------------
print("\n--- 3.2 Fiyat Sütununu Sayısal Formata Dönüştürme ---")

# Fiyat sütunu "3,100,000TL" formatında → temizle
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace("TL", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
print(f"  Fiyat dönüşümü sonrası NaN sayısı: {df['price'].isna().sum()}")

# Fiyatı NaN olan satırları sil (hedef değişken eksik olamaz)
df.dropna(subset=["price"], inplace=True)
print(f"  Fiyat temizliği sonrası satır sayısı: {df.shape[0]}")

# -------------------------------------------------------------------------
# 3.2.1 Enflasyon Duzeltmesi - Fiyatlari 2026'ya Guncelleme
# -------------------------------------------------------------------------
print("\n--- 3.2.1 Enflasyon Duzeltmesi (TUIK Verileri ile 2026'ya Guncelleme) ---")

# TUIK yillik TUFE enflasyon oranlari (Aralik - Aralik, yillik bazda)
# Kaynak: TUIK, TradingEconomics, Anadolu Ajansi
# 2026 icin: Ocak-Nisan 2026 donemi icin tahmini ~%7.5 (yilin ilk 4 ayi)
enflasyon_oranlari = {
    2022: 0.6427,   # 2022 yillik TUFE: %64.27
    2023: 0.6477,   # 2023 yillik TUFE: %64.77
    2024: 0.4438,   # 2024 yillik TUFE: %44.38
    2025: 0.3065,   # 2025 yillik TUFE: %30.65
    2026: 0.075,    # 2026 Ocak-Nisan donemi (tahmini %7.5)
}

# Ilan tarihinden yili cikar
df["AdYear"] = df["AdCreationDate"].str.extract(r"(\d{4})")[0].astype(float)
# Yili bulunamayan ilanlari 2022 olarak varsay (cogunluk 2022)
df["AdYear"] = df["AdYear"].fillna(2022).astype(int)

print(f"  Ilan yili dagilimi:")
for year, count in df["AdYear"].value_counts().sort_index().items():
    print(f"    {year}: {count} ilan")

def hesapla_enflasyon_carpani(ilan_yili, hedef_yil=2026):
    """
    Bir ilanin yilindan hedef yila kadar kumulatif enflasyon carpanini hesaplar.
    Ornek: 2021 -> 2026: (1+0.6427) * (1+0.6477) * (1+0.4438) * (1+0.3065) * (1+0.075)
    """
    carpan = 1.0
    # Ilan yilindan sonraki her yil icin enflasyon uygula
    for yil in range(ilan_yili + 1, hedef_yil + 1):
        oran = enflasyon_oranlari.get(yil, 0.30)  # Bilinmeyen yillar icin %30 varsayim
        carpan *= (1 + oran)
    return carpan

# Her ilan icin enflasyon carpanini hesapla
df["enflasyon_carpani"] = df["AdYear"].apply(hesapla_enflasyon_carpani)

# Orjinal fiyati sakla
df["price_original"] = df["price"].copy()

# Fiyatlari 2026'ya guncelle
df["price"] = df["price"] * df["enflasyon_carpani"]

print(f"\n  Enflasyon carpani ornekleri:")
for year in sorted(df["AdYear"].unique()):
    carpan = hesapla_enflasyon_carpani(year)
    count = (df["AdYear"] == year).sum()
    print(f"    {year} -> 2026: x{carpan:.2f} ({count} ilan)")

print(f"\n  Fiyat guncelleme ozeti:")
print(f"    Orjinal ortalama fiyat : {df['price_original'].mean():>15,.0f} TL")
print(f"    Guncel ortalama fiyat  : {df['price'].mean():>15,.0f} TL (2026)")
print(f"    Orjinal medyan fiyat   : {df['price_original'].median():>15,.0f} TL")
print(f"    Guncel medyan fiyat    : {df['price'].median():>15,.0f} TL (2026)")

# Gecici sutunlari temizle (model icin gereksiz)
df.drop(columns=["enflasyon_carpani", "price_original", "AdYear", "AdCreationDate"], inplace=True)

# -------------------------------------------------------------------------
# 3.3 Sayısal Sütunları Temizleme (m2, oda sayısı vb.)
# -------------------------------------------------------------------------
print("\n--- 3.3 Sayısal Sütunları Temizleme ---")

def clean_numeric(series):
    """'160 m2' gibi değerleri sayıya çevirir."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(" m2", "", regex=False)
        .str.replace("m2", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

# Metrekare sütunlarını temizle
for col in ["GrossSquareMeters", "NetSquareMeters"]:
    if col in df.columns:
        df[col] = clean_numeric(df[col])
        print(f"  {col}: NaN = {df[col].isna().sum()}")

# NumberOfBathrooms ve NumberOfWCs → sayısal
for col in ["NumberOfBathrooms", "NumberOfWCs"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
        print(f"  {col}: NaN = {df[col].isna().sum()}")

# -------------------------------------------------------------------------
# 3.4 NumberOfRooms → Toplam oda sayısı hesaplama
# -------------------------------------------------------------------------
print("\n--- 3.4 Oda Sayısını Sayısal Formata Dönüştürme ---")

def parse_rooms(val):
    """'3+1' → 4, '8+ Oda' → 9, '5 Oda' → 5 gibi dönüşüm."""
    val = str(val).strip()
    if "+" in val:
        parts = val.replace("Oda", "").strip().split("+")
        try:
            return sum(int(p.strip()) for p in parts if p.strip().isdigit())
        except ValueError:
            return np.nan
    else:
        digits = "".join(c for c in val if c.isdigit())
        return int(digits) if digits else np.nan

df["NumberOfRooms"] = df["NumberOfRooms"].apply(parse_rooms)
print(f"  NumberOfRooms: NaN = {df['NumberOfRooms'].isna().sum()}")

# -------------------------------------------------------------------------
# 3.5 BuildingAge → Ordinal Encoding (sıralı kategorik)
# -------------------------------------------------------------------------
print("\n--- 3.5 Bina Yaşı Dönüşümü ---")

building_age_map = {
    "0":    0,
    "1":    1,
    "2":    2,
    "3":    3,
    "4":    4,
    "5-10": 7,
    "11-15": 13,
    "16-20": 18,
    "21 Ve Üzeri": 25,
}

# Türkçe karakter sorunlarını çözmek için normalize edin
df["BuildingAge"] = (
    df["BuildingAge"]
    .astype(str)
    .str.strip()
)

# Map ile eşleştirme - eşleşmeyenler için fuzzy matching
def map_building_age(val):
    val_str = str(val).strip()
    for key, num in building_age_map.items():
        if key in val_str or val_str in key:
            return num
    # Sayısal değeri bulmaya çalış
    digits = "".join(c for c in val_str if c.isdigit())
    if digits:
        return int(digits)
    return np.nan

df["BuildingAge"] = df["BuildingAge"].apply(map_building_age)
print(f"  BuildingAge: NaN = {df['BuildingAge'].isna().sum()}")

# -------------------------------------------------------------------------
# 3.6 Eksik Değerlerin (NaN) Analizi ve Doldurulması
# -------------------------------------------------------------------------
print("\n--- 3.6 Eksik Değer Analizi ---")

nan_summary = df.isnull().sum()
nan_pct = (nan_summary / len(df) * 100).round(2)
nan_df = pd.DataFrame({"Eksik Sayı": nan_summary, "Oran (%)": nan_pct})
nan_df = nan_df[nan_df["Eksik Sayı"] > 0].sort_values("Oran (%)", ascending=False)
print(nan_df.to_string())

# Sayısal sütunlarda eksik değerleri medyan ile doldur
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col} -> medyan ({median_val}) ile dolduruldu")

# Kategorik sütunlarda eksik değerleri mod (en sık değer) ile doldur
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  {col} -> mod ('{mode_val}') ile dolduruldu")

# Eksik %70'den fazla olan sütunları kaldır (esik yukseltildi - daha fazla ozellik korunur)
high_nan_cols = [c for c in df.columns if df[c].isna().sum() / len(df) > 0.70]
if high_nan_cols:
    df.drop(columns=high_nan_cols, inplace=True)
    print(f"  Yuksek NaN orani nedeniyle silinen sutunlar: {high_nan_cols}")

print(f"\n  Eksik değer doldurma sonrası toplam NaN: {df.isnull().sum().sum()}")

# -------------------------------------------------------------------------
# 3.7 Aykırı Değer (Outlier) Tespiti ve Temizliği - IQR Yöntemi
# -------------------------------------------------------------------------
print("\n--- 3.7 Aykırı Değer Tespiti ve Temizliği (IQR) ---")

outlier_cols = ["price", "GrossSquareMeters", "NetSquareMeters"]
initial_rows = len(df)

for col in outlier_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        print(f"  {col}: Q1={Q1:,.0f}, Q3={Q3:,.0f}, IQR={IQR:,.0f} -> {removed} aykiri deger silindi")

print(f"  Aykiri deger temizligi: {initial_rows} -> {len(df)} satir (toplam {initial_rows - len(df)} silindi)")

# -------------------------------------------------------------------------
# 3.8 Ozellik Muhendisligi (Feature Engineering)
# -------------------------------------------------------------------------
print("\n--- 3.8 Ozellik Muhendisligi (Feature Engineering) ---")

# NOT: m2 basina fiyat (PricePerM2) ozelliklerini eklemedik cunku
# fiyat hedef degiskenimiz. Fiyattan tureyen ozellikler data leakage
# olusturur ve modelin gercek performansini yaniltir.

# Net/Brt metrekare orani (verimlilik gostergesi)
if "NetSquareMeters" in df.columns and "GrossSquareMeters" in df.columns:
    df["NetGrossRatio"] = df["NetSquareMeters"] / df["GrossSquareMeters"].replace(0, np.nan)
    df["NetGrossRatio"].fillna(df["NetGrossRatio"].median(), inplace=True)
    print("  NetGrossRatio olusturuldu")

# Oda yogunlugu (oda sayisi / metrekare)
if "NumberOfRooms" in df.columns and "NetSquareMeters" in df.columns:
    df["RoomDensity"] = df["NumberOfRooms"] / df["NetSquareMeters"].replace(0, np.nan)
    df["RoomDensity"].fillna(df["RoomDensity"].median(), inplace=True)
    print("  RoomDensity olusturuldu")

# Toplam islak alan (banyo + wc)
if "NumberOfBathrooms" in df.columns and "NumberOfWCs" in df.columns:
    df["TotalWetAreas"] = df["NumberOfBathrooms"] + df["NumberOfWCs"]
    print("  TotalWetAreas olusturuldu")

# Kat basina oda (bina kat sayisi / oda)
if "NumberFloorsofBuilding" in df.columns and "NumberOfRooms" in df.columns:
    df["FloorsPerRoom"] = df["NumberFloorsofBuilding"] / df["NumberOfRooms"].replace(0, np.nan)
    df["FloorsPerRoom"].fillna(df["FloorsPerRoom"].median(), inplace=True)
    print("  FloorsPerRoom olusturuldu")

print(f"  Toplam ozellik sayisi: {df.shape[1]}")

# =============================================================================
# 4. SINIFLANDIRMA DÖNÜŞÜMÜ (pd.qcut ile 3 sınıf)
# =============================================================================
print(f"\n{'='*60}")
print("4. SINIFLANDIRMA DÖNÜŞÜMÜ")
print(f"{'='*60}")

# Fiyatı 3 eşit frekanslı sınıfa böl
df["PriceClass"] = pd.qcut(
    df["price"],
    q=3,
    labels=["Ekonomik", "Orta", "Luks"]
)

print("\nFiyat Sınıfı Dağılımı:")
print(df["PriceClass"].value_counts().to_string())
print(f"\nSınıf aralıkları:")
for label in ["Ekonomik", "Orta", "Luks"]:
    subset = df[df["PriceClass"] == label]["price"]
    print(f"  {label:10s}: {subset.min():>12,.0f} TL  -  {subset.max():>12,.0f} TL")

# Fiyat sınıfı dağılım grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Sınıf dağılımı bar chart
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
df["PriceClass"].value_counts().plot(kind="bar", ax=axes[0], color=colors, edgecolor="black")
axes[0].set_title("Fiyat Sınıfı Dağılımı", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Sınıf")
axes[0].set_ylabel("Frekans")
axes[0].tick_params(axis="x", rotation=0)

# Sağ: Fiyat dağılımı histogram
axes[1].hist(df["price"], bins=50, color="#3498db", edgecolor="black", alpha=0.7)
axes[1].set_title("Fiyat Dağılımı (Aykırı Değer Temizliği Sonrası)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Fiyat (TL)")
axes[1].set_ylabel("Frekans")

plt.tight_layout()
plt.savefig("sinif_dagilimi.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] sinif_dagilimi.png kaydedildi.")

# =============================================================================
# 5. ÖZELLİK MÜHENDİSLİĞİ ve ENCODING
# =============================================================================
print(f"\n{'='*60}")
print("5. ENCODING ve OLCEKLENDIRME")
print(f"{'='*60}")

from sklearn.preprocessing import StandardScaler

# Hedef değişken ve orijinal fiyat sütununu ayır
y_col = "PriceClass"
X_df = df.drop(columns=["price", y_col])

# --- Frequency Encoding: district icin (yuksek kardinalite) ---
# Label Encoding sirasiz verilerde yaniltici olabilir; frequency encoding
# her kategoriyi veri setindeki sikligina gore sayisallastirir.
cat_cols_final = X_df.select_dtypes(include=["object"]).columns.tolist()
freq_encode_cols = ["district"]  # Yuksek kardinaliteli sutunlar
label_encode_cols = [c for c in cat_cols_final if c not in freq_encode_cols]

print(f"\nFrequency Encoding ({len(freq_encode_cols)} sutun):")
for col in freq_encode_cols:
    if col in X_df.columns:
        freq_map = X_df[col].value_counts(normalize=True).to_dict()
        X_df[col] = X_df[col].map(freq_map)
        print(f"  {col}: {len(freq_map)} benzersiz kategori -> frekans degerleri")

# --- Label Encoding: diger kategorik sutunlar ---
label_encoders = {}
print(f"\nLabel Encoding ({len(label_encode_cols)} sutun):")
for col in label_encode_cols:
    if col in X_df.columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} benzersiz kategori")

# Hedef değişkeni sayısala çevir
target_le = LabelEncoder()
y = target_le.fit_transform(df[y_col])
class_names = target_le.classes_
print(f"\nHedef siniflar: {list(class_names)} -> {list(range(len(class_names)))}")

# Son kontrol - kalan NaN varsa doldur
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df.fillna(X_df.median(), inplace=True)

# --- StandardScaler: sayisal ozellikleri olceklendir ---
scaler = StandardScaler()
X_df_scaled = pd.DataFrame(
    scaler.fit_transform(X_df),
    columns=X_df.columns,
    index=X_df.index
)
print(f"\nStandardScaler uygulanarak ozellikler olceklendirildi.")

print(f"\nFinal ozellik matrisi: {X_df.shape}")
print(f"Kullanilan ozellikler ({len(X_df.columns)}): {X_df.columns.tolist()}")

# =============================================================================
# 6. EĞİTİM/TEST BÖLME (%80/%20)
# =============================================================================
print(f"\n{'='*60}")
print("6. EĞİTİM/TEST BÖLME")
print(f"{'='*60}")

X_train, X_test, y_train, y_test = train_test_split(
    X_df_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Olceklenmemis versiyonu da feature importance icin saklayalim
X_train_raw, X_test_raw, _, _ = train_test_split(
    X_df, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Egitim seti : {X_train.shape[0]} ornek ({X_train.shape[0]/len(X_df)*100:.1f}%)")
print(f"  Test seti   : {X_test.shape[0]} ornek ({X_test.shape[0]/len(X_df)*100:.1f}%)")

# =============================================================================
# 7. MODEL EĞİTİMİ
# =============================================================================
print(f"\n{'='*60}")
print("7. MODEL EĞİTİMİ")
print(f"{'='*60}")

# --- 7.1 Random Forest Classifier (Optimize Edilmis) ---
print("\n--- 7.1 Random Forest Classifier (Optimize Edilmis) ---")
rf_model = RandomForestClassifier(
    n_estimators=500,          # 200 -> 500 (daha fazla agac)
    max_depth=30,              # 20 -> 30 (daha derin agaclar)
    min_samples_split=3,       # 5 -> 3
    min_samples_leaf=1,        # 2 -> 1
    max_features="sqrt",       # Varsayilan, acikca belirtildi
    class_weight="balanced",   # Sinif dengesizligini telafi et
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("  Random Forest egitimi tamamlandi.")

# --- 7.2 XGBoost Classifier (Optimize Edilmis) ---
print("\n--- 7.2 XGBoost Classifier (Optimize Edilmis) ---")
xgb_model = XGBClassifier(
    n_estimators=500,          # 200 -> 500
    max_depth=10,              # 8 -> 10
    learning_rate=0.05,        # 0.1 -> 0.05 (daha yavas ogrenim = daha iyi genelleme)
    subsample=0.85,            # 0.8 -> 0.85
    colsample_bytree=0.85,     # 0.8 -> 0.85
    min_child_weight=3,        # Overfitting onleme
    gamma=0.1,                 # Minimum kayip azalmasi
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("  XGBoost egitimi tamamlandi.")

# =============================================================================
# 8. DEĞERLENDİRME ve KARŞILAŞTIRMA
# =============================================================================
print(f"\n{'='*60}")
print("8. DEĞERLENDİRME ve KARŞILAŞTIRMA")
print(f"{'='*60}")

def evaluate_model(name, y_true, y_pred, class_names):
    """Model performans metriklerini hesaplar ve yazdırır."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    pre = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  {name:^37s}  │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)          │")
    print(f"  │  F1-Score  : {f1:.4f}                      │")
    print(f"  │  Precision : {pre:.4f}                      │")
    print(f"  │  Recall    : {rec:.4f}                      │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  Detaylı Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {"Accuracy": acc, "F1-Score": f1, "Precision": pre, "Recall": rec}

rf_metrics  = evaluate_model("Random Forest", y_test, rf_pred, class_names)
xgb_metrics = evaluate_model("XGBoost", y_test, xgb_pred, class_names)

# --- Karşılaştırma Tablosu ---
print("\n--- Model Karşılaştırma Tablosu ---")
comparison_df = pd.DataFrame({
    "Metrik": ["Accuracy", "F1-Score", "Precision", "Recall"],
    "Random Forest": [rf_metrics["Accuracy"], rf_metrics["F1-Score"],
                      rf_metrics["Precision"], rf_metrics["Recall"]],
    "XGBoost": [xgb_metrics["Accuracy"], xgb_metrics["F1-Score"],
                xgb_metrics["Precision"], xgb_metrics["Recall"]],
})
comparison_df["Fark (XGB - RF)"] = comparison_df["XGBoost"] - comparison_df["Random Forest"]
print(comparison_df.to_string(index=False))

# =============================================================================
# 9. CONFUSION MATRIX GÖRSELLEŞTİRMESİ
# =============================================================================
print(f"\n{'='*60}")
print("9. CONFUSION MATRIX GÖRSELLEŞTİRMESİ")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, name, y_pred, cmap in [
    (axes[0], "Random Forest", rf_pred, "Blues"),
    (axes[1], "XGBoost", xgb_pred, "Oranges"),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="gray"
    )
    ax.set_title(f"Confusion Matrix - {name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] confusion_matrix.png kaydedildi.")

# =============================================================================
# 10. ÖZELLİK ÖNEM DÜZEYLERİ (Feature Importance)
# =============================================================================
print(f"\n{'='*60}")
print("10. ÖZELLİK ÖNEM DÜZEYLERİ")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, model, name, color in [
    (axes[0], rf_model, "Random Forest", "#3498db"),
    (axes[1], xgb_model, "XGBoost", "#e67e22"),
]:
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_df.columns).sort_values(ascending=True)

    # En önemli 15 özelliği göster
    top_n = feat_imp.tail(15)
    top_n.plot(kind="barh", ax=ax, color=color, edgecolor="black", alpha=0.85)
    ax.set_title(f"Özellik Önem Düzeyleri - {name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Önem Skoru", fontsize=12)
    ax.set_ylabel("")

print("\nRandom Forest - En Önemli 5 Özellik:")
rf_imp = pd.Series(rf_model.feature_importances_, index=X_df.columns).sort_values(ascending=False)
for i, (feat, imp) in enumerate(rf_imp.head(5).items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print("\nXGBoost - En Önemli 5 Özellik:")
xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_df.columns).sort_values(ascending=False)
for i, (feat, imp) in enumerate(xgb_imp.head(5).items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] feature_importance.png kaydedildi.")

# =============================================================================
# 11. TEKNİK YORUM ve SONUÇ
# =============================================================================
print(f"\n{'='*60}")
print("11. TEKNİK YORUM ve SONUÇ")
print(f"{'='*60}")

# Hangi model daha iyi?
rf_acc  = rf_metrics["Accuracy"]
xgb_acc = xgb_metrics["Accuracy"]
rf_f1   = rf_metrics["F1-Score"]
xgb_f1  = xgb_metrics["F1-Score"]

if xgb_acc > rf_acc:
    winner = "XGBoost"
    diff_acc = (xgb_acc - rf_acc) * 100
    diff_f1  = (xgb_f1 - rf_f1) * 100
else:
    winner = "Random Forest"
    diff_acc = (rf_acc - xgb_acc) * 100
    diff_f1  = (rf_f1 - xgb_f1) * 100

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    TEKNİK DEĞERLENDİRME                      ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Kazanan Model: {winner:<15s}                                 ║
║  Accuracy Farkı: {diff_acc:.2f}%                                     ║
║  F1-Score Farkı: {diff_f1:.2f}%                                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

--- Detaylı Teknik Analiz ---

1. RANDOM FOREST:
   - Ensemble (topluluk) yöntemi olup birden fazla karar ağacının 
     çoğunluk oylamasıyla karar verir.
   - Overfitting'e karşı dayanıklıdır; bagging yöntemi kullanır.
   - Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}

2. XGBOOST:
   - Gradient boosting tabanlı, sıralı öğrenme algoritmasıdır.
   - Her yeni ağaç, önceki ağaçların hatalarını düzeltmeye çalışır.
   - Regularization (L1/L2) sayesinde overfitting'i kontrol eder.
   - Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f}

3. KARŞILAŞTIRMA:
   - {winner} modeli bu veri setinde daha yüksek performans göstermiştir.
   - XGBoost, boosting yaklaşımıyla hataları iteratif olarak minimize 
     ederken; Random Forest, bağımsız ağaçların oylamasını kullanır.
   - Emlak fiyatlandırmasında konum (district), metrekare ve oda sayısı
     gibi özellikler en belirleyici faktörler olarak öne çıkmaktadır.
   - Her iki model de kategorik değişkenlerin label encoding ile 
     dönüştürülmesinden etkilenebilir; one-hot encoding alternatif 
     olarak denenebilir.

--- Proje Tamamlandi ---
""")

# =============================================================================
# 12. EK GORSELLESTIRMELER (Sunum icin)
# =============================================================================
print(f"\n{'='*60}")
print("12. EK GORSELLESTIRMELER (Sunum icin)")
print(f"{'='*60}")

# -------------------------------------------------------------------------
# 12.1 Korelasyon Isı Haritasi (Correlation Heatmap)
# -------------------------------------------------------------------------
print("\n--- 12.1 Korelasyon Isi Haritasi ---")

fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = X_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Ozellikler Arasi Korelasyon Matrisi", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("korelasyon_haritasi.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] korelasyon_haritasi.png kaydedildi.")

# -------------------------------------------------------------------------
# 12.2 Ilcelere Gore Fiyat Dagilimi (Boxplot - Top 15)
# -------------------------------------------------------------------------
print("\n--- 12.2 Ilcelere Gore Fiyat Dagilimi ---")

# Orijinal veriyi tekrar yukle (ilce bazli analiz icin)
try:
    df_viz = pd.read_csv(csv_path)
    df_viz["price"] = (
        df_viz["price"].astype(str)
        .str.replace("TL", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df_viz["price"] = pd.to_numeric(df_viz["price"], errors="coerce")
    df_viz.dropna(subset=["price"], inplace=True)

    # Enflasyon duzeltmesi (basitlestirilmis - cogunluk 2022)
    df_viz["price"] = df_viz["price"] * 3.34  # Ortalama carpan

    # En cok ilana sahip 15 ilce
    top_districts = df_viz["district"].value_counts().head(15).index
    df_top = df_viz[df_viz["district"].isin(top_districts)]

    # Fiyat aykiri degerleri kırp (gorsel icin)
    upper_limit = df_top["price"].quantile(0.95)
    df_top = df_top[df_top["price"] <= upper_limit]

    # Ilceleri medyan fiyata gore sirala
    district_order = df_top.groupby("district")["price"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(
        data=df_top, x="district", y="price", order=district_order,
        palette="viridis", ax=ax, fliersize=2
    )
    ax.set_title("Ilcelere Gore Ev Fiyat Dagilimi (Top 15 Ilce, 2026 TL)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Ilce", fontsize=12)
    ax.set_ylabel("Fiyat (TL)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig("ilce_fiyat_dagilimi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[GRAFIK] ilce_fiyat_dagilimi.png kaydedildi.")
except Exception as e:
    print(f"  [UYARI] Ilce grafigi olusturulamadi: {e}")

# -------------------------------------------------------------------------
# 12.3 Model Metrik Karsilastirma (Grouped Bar Chart)
# -------------------------------------------------------------------------
print("\n--- 12.3 Model Metrik Karsilastirma ---")

fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ["Accuracy", "F1-Score", "Precision", "Recall"]
rf_values = [rf_metrics[m] for m in metrics_names]
xgb_values = [xgb_metrics[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, rf_values, width, label="Random Forest",
               color="#3498db", edgecolor="black", alpha=0.85)
bars2 = ax.bar(x + width/2, xgb_values, width, label="XGBoost",
               color="#e67e22", edgecolor="black", alpha=0.85)

# Deger etiketleri
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Skor", fontsize=12)
ax.set_title("Model Performans Karsilastirmasi", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.legend(fontsize=11, loc="lower right")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_karsilastirma.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] model_karsilastirma.png kaydedildi.")

# -------------------------------------------------------------------------
# 12.4 Sinif Bazli Fiyat Dagilimi (Violin Plot)
# -------------------------------------------------------------------------
print("\n--- 12.4 Sinif Bazli Fiyat Dagilimi (Violin Plot) ---")

fig, ax = plt.subplots(figsize=(12, 6))
colors_violin = {"Ekonomik": "#2ecc71", "Orta": "#f39c12", "Luks": "#e74c3c"}
sns.violinplot(
    data=df, x="PriceClass", y="price", order=["Ekonomik", "Orta", "Luks"],
    palette=colors_violin, ax=ax, inner="box", linewidth=1.5
)
ax.set_title("Fiyat Sinifina Gore Fiyat Dagilimi (Violin Plot)", fontsize=14, fontweight="bold")
ax.set_xlabel("Fiyat Sinifi", fontsize=12)
ax.set_ylabel("Fiyat (TL)", fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("sinif_violin_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] sinif_violin_plot.png kaydedildi.")

# -------------------------------------------------------------------------
# 12.5 Brut m2 vs Oda Sayisi (Scatter - sinif renkleriyle)
# -------------------------------------------------------------------------
print("\n--- 12.5 Ozellik Iliskisi Scatter Plot ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter_colors = {"Ekonomik": "#2ecc71", "Orta": "#f39c12", "Luks": "#e74c3c"}

# Sol: Brut m2 vs Fiyat
for label in ["Ekonomik", "Orta", "Luks"]:
    subset = df[df["PriceClass"] == label]
    axes[0].scatter(
        subset["GrossSquareMeters"], subset["price"],
        c=scatter_colors[label], label=label, alpha=0.3, s=10, edgecolors="none"
    )
axes[0].set_title("Brut Metrekare vs Fiyat", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Brut Metrekare (m2)", fontsize=11)
axes[0].set_ylabel("Fiyat (TL)", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))

# Sag: Oda Sayisi vs Fiyat
for label in ["Ekonomik", "Orta", "Luks"]:
    subset = df[df["PriceClass"] == label]
    axes[1].scatter(
        subset["NumberOfRooms"], subset["price"],
        c=scatter_colors[label], label=label, alpha=0.3, s=10, edgecolors="none"
    )
axes[1].set_title("Oda Sayisi vs Fiyat", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Toplam Oda Sayisi", fontsize=11)
axes[1].set_ylabel("Fiyat (TL)", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))

plt.tight_layout()
plt.savefig("ozellik_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] ozellik_scatter.png kaydedildi.")

# -------------------------------------------------------------------------
# 12.6 Sinif Dagilimi Pasta Grafigi (Pie Chart)
# -------------------------------------------------------------------------
print("\n--- 12.6 Sinif Dagilimi Pasta Grafigi ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sol: Sinif dagilimi
class_counts = df["PriceClass"].value_counts()
pie_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
explode = (0.03, 0.03, 0.03)
axes[0].pie(
    class_counts, labels=class_counts.index, autopct="%1.1f%%",
    colors=pie_colors, explode=explode, shadow=True, startangle=90,
    textprops={"fontsize": 12, "fontweight": "bold"}
)
axes[0].set_title("Fiyat Sinifi Dagilimi", fontsize=14, fontweight="bold")

# Sag: Ilce dagilimi (Top 10)
district_counts = df["district"].value_counts().head(10)
axes[1].barh(
    district_counts.index[::-1], district_counts.values[::-1],
    color=sns.color_palette("viridis", len(district_counts)),
    edgecolor="black"
)
axes[1].set_title("En Cok Ilana Sahip 10 Ilce", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Ilan Sayisi", fontsize=11)

plt.tight_layout()
plt.savefig("sinif_ve_ilce_dagilimi.png", dpi=150, bbox_inches="tight")
plt.show()
print("[GRAFIK] sinif_ve_ilce_dagilimi.png kaydedildi.")

# -------------------------------------------------------------------------
# 12.7 Eksik Veri Gorsellestirmesi
# -------------------------------------------------------------------------
print("\n--- 12.7 Eksik Veri Haritasi ---")

try:
    df_raw = pd.read_csv(csv_path)
    nan_pcts = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=True)
    nan_pcts = nan_pcts[nan_pcts > 0]  # Sadece eksik olanlari goster

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_nan = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#2ecc71" for v in nan_pcts.values]
    nan_pcts.plot(kind="barh", ax=ax, color=colors_nan, edgecolor="black", alpha=0.85)
    ax.set_title("Sutunlardaki Eksik Veri Oranlari (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Eksik Veri Orani (%)", fontsize=12)
    ax.axvline(x=50, color="red", linestyle="--", linewidth=1.5, label="Kritik esik (%50)")
    ax.axvline(x=70, color="darkred", linestyle="--", linewidth=1.5, label="Silme esigi (%70)")
    ax.legend(fontsize=10)

    # Yuzde etiketleri
    for i, (val, name) in enumerate(zip(nan_pcts.values, nan_pcts.index)):
        ax.text(val + 0.5, i, f"%{val:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("eksik_veri_haritasi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[GRAFIK] eksik_veri_haritasi.png kaydedildi.")
except Exception as e:
    print(f"  [UYARI] Eksik veri grafigi olusturulamadi: {e}")

print(f"\n{'='*60}")
print("TUM GORSELLESTIRMELER TAMAMLANDI")
print(f"{'='*60}")
print("""
Olusturulan grafik dosyalari:
  1. sinif_dagilimi.png        - Fiyat sinifi dagilimi ve histogram
  2. confusion_matrix.png      - Karmasiklik matrisleri (RF & XGBoost)
  3. feature_importance.png    - Ozellik onem duzeyleri
  4. korelasyon_haritasi.png   - Ozellikler arasi korelasyon
  5. ilce_fiyat_dagilimi.png   - Ilcelere gore fiyat (boxplot)
  6. model_karsilastirma.png   - Model metrik karsilastirmasi
  7. sinif_violin_plot.png     - Sinif bazli fiyat violin plot
  8. ozellik_scatter.png       - Brut m2 & Oda vs Fiyat scatter
  9. sinif_ve_ilce_dagilimi.png - Pasta grafigi + ilce dagilimi
  10. eksik_veri_haritasi.png   - Eksik veri oranlari
""")
