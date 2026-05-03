"""Veri yukleme ve on isleme modulu - Streamlit icin"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    r2_score,
)
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

ENFLASYON = {2022: 0.6427, 2023: 0.6477, 2024: 0.4438, 2025: 0.3065, 2026: 0.075}
BUILDING_AGE_MAP = {"0": 0, "1": 1, "2-5":3, "5-10":7, "11-15":13, "16-20":18, "21 Ve Üzeri":25}


def use_fast_mode():
    raw = os.getenv("FAST_MODE")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def hesapla_carpan(yil, hedef=2026):
    c = 1.0
    for y in range(yil+1, hedef+1):
        c *= (1 + ENFLASYON.get(y, 0.30))
    return c

def load_and_process():
    local_csv = Path(__file__).resolve().parent / "HouseData.csv"
    if local_csv.exists():
        csv = str(local_csv)
    else:
        try:
            import kagglehub

            path = kagglehub.dataset_download("aselasel/house-price-dataset")
            remote_csv = Path(path) / "HouseData.csv"
            if not remote_csv.exists():
                raise FileNotFoundError(f"Indirilen veri dizininde HouseData.csv bulunamadi: {path}")
            csv = str(remote_csv)
        except Exception as exc:
            raise FileNotFoundError(
                "HouseData.csv bulunamadi. Streamlit Cloud icin projeye HouseData.csv ekleyin "
                "veya Kaggle erisimi (kagglehub) saglayin."
            ) from exc
    
    df = pd.read_csv(csv)
    raw_df = df.copy()
    
    drop_cols = ["Unnamed: 0","address","AdUpdateDate","Category","Type","PriceStatus",
                 "TitleStatus","MortgageStatus","RentalIncome","NumberOfBalconies","BalconyType",
                 "HallSquareMeters","WCSquareMeters","IsItVideoNavigable?","Subscription",
                 "BathroomSquareMeters","BalconySquareMeters"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    # Fiyat temizleme
    df["price"] = df["price"].astype(str).str.replace("TL","",regex=False).str.replace(",","",regex=False).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(subset=["price"], inplace=True)
    
    # Enflasyon
    df["AdYear"] = df["AdCreationDate"].str.extract(r"(\d{4})")[0].astype(float).fillna(2022).astype(int)
    df["price"] = df["price"] * df["AdYear"].apply(hesapla_carpan)
    df.drop(columns=["AdYear","AdCreationDate"], inplace=True, errors="ignore")
    
    # m2 temizleme
    for col in ["GrossSquareMeters","NetSquareMeters"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("m2","",regex=False).str.replace(",","",regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    for col in ["NumberOfBathrooms","NumberOfWCs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    
    # Oda parse
    def parse_rooms(val):
        val = str(val).strip()
        if "+" in val:
            parts = val.replace("Oda","").strip().split("+")
            try: return sum(int(p.strip()) for p in parts if p.strip().isdigit())
            except: return np.nan
        digits = "".join(c for c in val if c.isdigit())
        return int(digits) if digits else np.nan
    if "NumberOfRooms" in df.columns:
        df["NumberOfRooms"] = df["NumberOfRooms"].apply(parse_rooms)
    
    # Bina yasi
    if "BuildingAge" in df.columns:
        df["BuildingAge"] = df["BuildingAge"].map(BUILDING_AGE_MAP)
        df["BuildingAge"].fillna(df["BuildingAge"].median(), inplace=True)
    
    # NaN doldur
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    high_nan = [c for c in df.columns if df[c].isna().sum()/len(df) > 0.70]
    if high_nan: df.drop(columns=high_nan, inplace=True)
    
    # Outlier IQR
    for col in ["price","GrossSquareMeters","NetSquareMeters"]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1-1.5*IQR) & (df[col] <= Q3+1.5*IQR)]
    
    # Feature engineering
    if "NetSquareMeters" in df.columns and "GrossSquareMeters" in df.columns:
        df["NetGrossRatio"] = df["NetSquareMeters"] / df["GrossSquareMeters"].replace(0, np.nan)
        df["NetGrossRatio"].fillna(df["NetGrossRatio"].median(), inplace=True)
    if "NumberOfRooms" in df.columns and "NetSquareMeters" in df.columns:
        df["RoomDensity"] = df["NumberOfRooms"] / df["NetSquareMeters"].replace(0, np.nan)
        df["RoomDensity"].fillna(df["RoomDensity"].median(), inplace=True)
    if "NumberOfBathrooms" in df.columns and "NumberOfWCs" in df.columns:
        df["TotalWetAreas"] = df["NumberOfBathrooms"] + df["NumberOfWCs"]
    if "NumberFloorsofBuilding" in df.columns and "NumberOfRooms" in df.columns:
        df["FloorsPerRoom"] = df["NumberFloorsofBuilding"] / df["NumberOfRooms"].replace(0, np.nan)
        df["FloorsPerRoom"].fillna(df["FloorsPerRoom"].median(), inplace=True)
    
    # Siniflandirma
    df["PriceClass"] = pd.qcut(df["price"], q=3, labels=["Ekonomik","Orta","Luks"])
    
    return df, raw_df

def train_models(df):
    y_col = "PriceClass"
    fast_mode = use_fast_mode()
    train_df = df
    if fast_mode and len(df) > 8000:
        train_df, _ = train_test_split(
            df,
            train_size=8000,
            random_state=42,
            stratify=df[y_col],
        )

    X_df = train_df.drop(columns=["price", y_col])
    
    cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    freq_cols = ["district"] if "district" in cat_cols else []
    label_cols = [c for c in cat_cols if c not in freq_cols]
    
    encoders = {}
    for col in freq_cols:
        if col in X_df.columns:
            fm = X_df[col].value_counts(normalize=True).to_dict()
            X_df[col] = X_df[col].map(fm)
            encoders[col] = fm
    for col in label_cols:
        if col in X_df.columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            encoders[col] = le
    
    target_le = LabelEncoder()
    y = target_le.fit_transform(train_df[y_col])
    class_names = target_le.classes_
    
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df.fillna(X_df.median(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns, index=X_df.index)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_params = {
        "n_estimators": 500,
        "max_depth": 30,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if fast_mode:
        rf_params.update({"n_estimators": 120, "max_depth": 16})
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    if fast_mode or XGBClassifier is None:
        xgb = RandomForestClassifier(
            n_estimators=140 if fast_mode else 260,
            max_depth=16 if fast_mode else 24,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=7,
            n_jobs=-1,
        )
    else:
        xgb_params = {
            "n_estimators": 320,
            "max_depth": 9,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cpu",
        }
        xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    
    def metrics(y_true, y_pred):
        return {"Accuracy": accuracy_score(y_true, y_pred),
                "F1-Score": f1_score(y_true, y_pred, average="weighted"),
                "Precision": precision_score(y_true, y_pred, average="weighted"),
                "Recall": recall_score(y_true, y_pred, average="weighted")}
    
    # --- Regresyon modeli (fiyat tahmini) ---
    X_reg = train_df.drop(columns=["price", y_col])
    # Ayni encoding uygula
    for col in freq_cols:
        if col in X_reg.columns:
            X_reg[col] = X_reg[col].map(encoders[col])
    for col in label_cols:
        if col in X_reg.columns and col in encoders:
            le = encoders[col]
            X_reg[col] = X_reg[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    X_reg = X_reg.apply(pd.to_numeric, errors="coerce")
    X_reg.fillna(X_reg.median(), inplace=True)
    X_reg_scaled = pd.DataFrame(scaler.transform(X_reg), columns=X_df.columns, index=X_reg.index)
    
    y_price = train_df["price"].values
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_scaled, y_price, test_size=0.2, random_state=42)
    
    if fast_mode or XGBRegressor is None:
        reg_model = RandomForestRegressor(
            n_estimators=180 if fast_mode else 280,
            max_depth=16 if fast_mode else 24,
            random_state=42,
            n_jobs=-1,
        )
    else:
        reg_params = {
            "n_estimators": 320,
            "max_depth": 9,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cpu",
        }
        reg_model = XGBRegressor(**reg_params)
    reg_model.fit(Xr_train, yr_train)
    reg_pred = reg_model.predict(Xr_test)
    reg_metrics = {
        "MAE": mean_absolute_error(yr_test, reg_pred),
        "R2": r2_score(yr_test, reg_pred),
    }

    # --- Ilce istatistikleri ---
    stats_df = df.copy()
    if "NetSquareMeters" in stats_df.columns:
        stats_df["price_per_m2"] = stats_df["price"] / stats_df["NetSquareMeters"].clip(lower=1)
    else:
        stats_df["price_per_m2"] = np.nan
    district_stats = (
        stats_df.groupby("district")
        .agg(
            medyan=("price", "median"),
            ortalama=("price", "mean"),
            min=("price", "min"),
            max=("price", "max"),
            q25=("price", lambda s: s.quantile(0.25)),
            q75=("price", lambda s: s.quantile(0.75)),
            ilan_sayisi=("price", "count"),
            medyan_m2=("price_per_m2", "median"),
        )
        .reset_index()
    )
    
    # --- Kategori secenekleri ---
    cat_options = {}
    original_df = df.copy()
    for col in label_cols:
        if col in original_df.columns:
            opts = (
                original_df[col]
                .dropna()
                .astype(str)
                .str.strip()
            )
            opts = opts[opts != ""]
            cat_options[col] = sorted(opts.unique().tolist())
    districts = sorted(df["district"].unique().tolist()) if "district" in df.columns else []
    
    # --- Sinif fiyat araliklari ---
    class_ranges = {}
    for label in ["Ekonomik","Orta","Luks"]:
        subset = df[df["PriceClass"]==label]["price"]
        class_ranges[label] = {"min": subset.min(), "max": subset.max(), "median": subset.median()}

    numeric_stats = {}
    numeric_input_cols = [
        "GrossSquareMeters",
        "NetSquareMeters",
        "NumberOfRooms",
        "BuildingAge",
        "NumberFloorsofBuilding",
        "NumberOfBathrooms",
        "NumberOfWCs",
    ]
    for col in numeric_input_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                q01 = float(values.quantile(0.01))
                q99 = float(values.quantile(0.99))
                if q99 <= q01:
                    q01 = float(values.min())
                    q99 = float(values.max())
                numeric_stats[col] = {
                    "min": q01,
                    "max": q99,
                    "median": float(values.median()),
                }

    category_defaults = {}
    for col in label_cols:
        if col in df.columns and len(df[col].mode()) > 0:
            category_defaults[col] = str(df[col].mode().iloc[0]).strip()

    return {
        "rf_model": rf, "xgb_model": xgb, "reg_model": reg_model,
        "rf_pred": rf_pred, "xgb_pred": xgb_pred,
        "rf_metrics": metrics(y_test, rf_pred),
        "xgb_metrics": metrics(y_test, xgb_pred),
        "reg_metrics": reg_metrics,
        "X_df": X_df, "X_test": X_test, "y_test": y_test,
        "class_names": class_names, "target_le": target_le,
        "scaler": scaler, "encoders": encoders,
        "feature_names": X_df.columns.tolist(),
        "districts": districts, "cat_options": cat_options,
        "district_stats": district_stats, "class_ranges": class_ranges,
        "numeric_stats": numeric_stats,
        "category_defaults": category_defaults,
        "feature_defaults": X_df.median().to_dict(),
        "fast_mode": fast_mode,
        "train_rows": len(train_df),
        "df": df
    }
