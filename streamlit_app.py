import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ev Fiyat Siniflandirma", page_icon="🏠", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap');

:root {
    --ink: #0f172a;
    --muted: #475569;
    --soft: #e2e8f0;
    --accent: #e74c3c;
    --accent-2: #f39c12;
    --accent-3: #2ecc71;
    --panel: #0f172a;
    --panel-2: #1e293b;
}

html, body, [class*="css"]  {
    font-family: 'Montserrat', sans-serif;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    background: white;
}

.main-header {
    font-family: 'Montserrat', sans-serif;
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: 0.4px;
    color: #0f172a;
    text-align: center;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 1.05rem;
    margin-top: 0.2rem;
}

.section-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.35rem;
}

.metric-card {
    background: black;
    padding: 1.2rem;
    border-radius: 14px;
    text-align: center;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
    font-weight: 600 !important;
}

.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #7dd3fc;
}

.metric-label {
    font-size: 0.85rem;
    color: #cbd5f5;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #0f172a;
    background: #fef3c7;
    border: 1px solid #fde68a;
}

.soft-panel {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border: 1px solid var(--soft);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
}

.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #0f172a;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    border: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1f2937 100%);
}
section[data-testid="stSidebar"] * {color: #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ---------- CACHE DATA ----------
@st.cache_resource(show_spinner="Veri seti yukleniyor ve modeller egitiliyor...")
def get_data():
    from data_loader import load_and_process, train_models
    df, raw_df = load_and_process()
    results = train_models(df)
    return df, raw_df, results

df, raw_df, res = get_data()

# ---------- SIDEBAR ----------
st.sidebar.image("https://img.icons8.com/3d-fluency/94/home.png", width=80)
st.sidebar.title("🏠 Navigasyon")
page = st.sidebar.radio("Sayfa Secin:", ["📌 Proje Ozeti", "📊 Veri Analizi", "🤖 Model Sonuclari",
                                         "📈 Gorsellestirmeler", "🔮 Tahmin Yap", "📋 Teknik Bilgi"])
st.sidebar.markdown("---")
st.sidebar.info(f"**Veri:** {len(df):,} satir | **Ozellik:** {len(res['feature_names'])} | **Sinif:** 3")
if res.get("fast_mode"):
    st.sidebar.caption(f"Cloud hizli mod aktif | Egitim ornegi: {res.get('train_rows', len(df)):,}")

# ========== PAGE: PROJE OZETI ==========
if page == "📌 Proje Ozeti":
    st.markdown('<h1 class="main-header">İstanbul Ev Fiyat Sınıflandırması</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="hero-subtitle">Makine Öğrenmesi Final Projesi | 2026</h2>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="font-weight:600;">Mehmet Kahya | Boğaç Övuç</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-panel" style="text-align:center;">
            <span class="badge">Canli Demo</span>
            <span style="margin:0 10px; color:#0f172a; font-weight:600;">Model: XGBoost + Random Forest</span>
            <span style="margin:0 10px; color:#0f172a; font-weight:600;">Veri: 2026 guncel</span>
            <span style="margin:0 10px; color:#0f172a; font-weight:600;">Sinif: 3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(raw_df):,}</div><div class="metric-label">Toplam Veri</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df):,}</div><div class="metric-label">Temiz Veri</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{res["xgb_metrics"]["Accuracy"]*100:.1f}%</div><div class="metric-label">XGBoost Accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">39</div><div class="metric-label">Istanbul Ilce</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🎯 Proje Amaci</div>', unsafe_allow_html=True)
        st.write("""Istanbul'daki satilik evlerin fiyatlarini **Ekonomik**, **Orta** ve **Luks** 
        olarak siniflandiran bir ML sistemi. TUIK enflasyon verileriyle fiyatlar 2026'ya guncellendi.""")
    with col2:
        st.markdown('<div class="section-title">⚙ Kullanilan Yontemler</div>', unsafe_allow_html=True)
        st.write("- Random Forest (Bagging)\n- XGBoost (Gradient Boosting)\n- Feature Engineering\n- StandardScaler + Frequency Encoding")

# ========== PAGE: VERI ANALIZI ==========
elif page == "📊 Veri Analizi":
    st.header("📊 Veri Seti Analizi")
    
    tab1, tab2, tab3 = st.tabs(["Veri Onizleme", "Eksik Veri", "Istatistikler"])
    
    with tab1:
        st.dataframe(df.head(50), use_container_width=True, height=400)
    
    with tab2:
        nan_pcts = (raw_df.isnull().sum() / len(raw_df) * 100).sort_values(ascending=True)
        nan_pcts = nan_pcts[nan_pcts > 0]
        colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#2ecc71" for v in nan_pcts.values]
        fig = go.Figure(go.Bar(x=nan_pcts.values, y=nan_pcts.index, orientation="h",
                               marker_color=colors, text=[f"%{v:.1f}" for v in nan_pcts.values], textposition="outside"))
        fig.update_layout(title="Eksik Veri Oranlari (%)", xaxis_title="%", height=600,
                         shapes=[dict(type="line",x0=70,x1=70,y0=-0.5,y1=len(nan_pcts)-0.5,line=dict(color="red",dash="dash",width=2))])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

# ========== PAGE: MODEL SONUCLARI ==========
elif page == "🤖 Model Sonuclari":
    st.header("🤖 Model Performans Karsilastirmasi")
    
    c1, c2 = st.columns(2)
    for col, name, metrics, color in [(c1,"Random Forest",res["rf_metrics"],"#3498db"),(c2,"XGBoost",res["xgb_metrics"],"#e67e22")]:
        with col:
            st.metric(f"🎯 {name} Accuracy", f"{metrics['Accuracy']*100:.2f}%")
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
            st.metric("Precision", f"{metrics['Precision']:.4f}")
            st.metric("Recall", f"{metrics['Recall']:.4f}")
    
    st.markdown("---")
    # Metrik karsilastirma
    metrics_names = ["Accuracy","F1-Score","Precision","Recall"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Random Forest", x=metrics_names, y=[res["rf_metrics"][m] for m in metrics_names],
                         marker_color="#3498db", text=[f"{res['rf_metrics'][m]:.3f}" for m in metrics_names], textposition="outside"))
    fig.add_trace(go.Bar(name="XGBoost", x=metrics_names, y=[res["xgb_metrics"][m] for m in metrics_names],
                         marker_color="#e67e22", text=[f"{res['xgb_metrics'][m]:.3f}" for m in metrics_names], textposition="outside"))
    fig.update_layout(title="Model Metrik Karsilastirmasi", barmode="group", yaxis_range=[0,1.05], height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrix")
    cn = res["class_names"]
    c1, c2 = st.columns(2)
    for col, name, pred, colorscale in [(c1,"Random Forest",res["rf_pred"],"Blues"),(c2,"XGBoost",res["xgb_pred"],"Oranges")]:
        cm = confusion_matrix(res["y_test"], pred)
        fig = px.imshow(cm, x=cn, y=cn, text_auto=True, color_continuous_scale=colorscale,
                       labels=dict(x="Tahmin", y="Gercek"), title=f"CM - {name}")
        fig.update_layout(height=400)
        with col:
            st.plotly_chart(fig, use_container_width=True)

# ========== PAGE: GORSELLESTIRMELER ==========
elif page == "📈 Gorsellestirmeler":
    st.header("📈 Interaktif Gorsellestirmeler")
    
    viz = st.selectbox("Grafik Secin:", ["Sinif Dagilimi","Ilcelere Gore Fiyat","Ozellik Onem Duzeyleri",
                                        "Korelasyon Haritasi","Violin Plot","Scatter Plot"])
    
    if viz == "Sinif Dagilimi":
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names="PriceClass", title="Fiyat Sinifi Dagilimi",
                        color_discrete_map={"Ekonomik":"#2ecc71","Orta":"#f39c12","Luks":"#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="price", nbins=50, title="Fiyat Dagilimi (2026 TL)",
                             color_discrete_sequence=["#3498db"])
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Ilcelere Gore Fiyat":
        top_n = st.slider("Ilce sayisi:", 5, 30, 15)
        top_d = df["district"].value_counts().head(top_n).index
        fig = px.box(df[df["district"].isin(top_d)], x="district", y="price", color="district",
                    title=f"Top {top_n} Ilce - Fiyat Dagilimi (2026 TL)")
        fig.update_layout(height=500, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Ozellik Onem Duzeyleri":
        c1, c2 = st.columns(2)
        for col, model, name, color in [(c1,res["rf_model"],"Random Forest","#3498db"),(c2,res["xgb_model"],"XGBoost","#e67e22")]:
            imp = pd.Series(model.feature_importances_, index=res["feature_names"]).sort_values()
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", title=f"Feature Importance - {name}",
                        color_discrete_sequence=[color])
            fig.update_layout(height=500, xaxis_title="Onem Skoru", yaxis_title="")
            with col:
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Korelasyon Haritasi":
        corr = res["X_df"].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                       title="Ozellikler Arasi Korelasyon", zmin=-1, zmax=1)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Violin Plot":
        fig = px.violin(df, x="PriceClass", y="price", color="PriceClass", box=True,
                       category_orders={"PriceClass":["Ekonomik","Orta","Luks"]},
                       color_discrete_map={"Ekonomik":"#2ecc71","Orta":"#f39c12","Luks":"#e74c3c"},
                       title="Sinif Bazli Fiyat Dagilimi")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Scatter Plot":
        x_feat = st.selectbox("X ekseni:", ["GrossSquareMeters","NetSquareMeters","NumberOfRooms","NumberOfBathrooms"])
        fig = px.scatter(df, x=x_feat, y="price", color="PriceClass", opacity=0.4,
                        color_discrete_map={"Ekonomik":"#2ecc71","Orta":"#f39c12","Luks":"#e74c3c"},
                        title=f"{x_feat} vs Fiyat", category_orders={"PriceClass":["Ekonomik","Orta","Luks"]})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ========== PAGE: TAHMIN YAP ==========
elif page == "🔮 Tahmin Yap":
    st.header("🔮 Ilce Bazli Detayli Fiyat Tahmini")
    st.markdown(
        "Kullanici secimlerine gore **fiyat sinifi + fiyat tahmini + ilce bazli analiz** uretilir. "
        "Model tahmini, ilce m² medyani ile dengelenerek hesaplanir."
    )

    numeric_stats = res.get("numeric_stats", {})
    category_defaults = res.get("category_defaults", {})

    def get_num_bounds(col_name, fallback_min, fallback_max, fallback_med):
        col_stats = numeric_stats.get(col_name, {})
        min_v = int(round(col_stats.get("min", fallback_min)))
        max_v = int(round(col_stats.get("max", fallback_max)))
        med_v = int(round(col_stats.get("median", fallback_med)))
        if max_v <= min_v:
            max_v = min_v + 1
        med_v = min(max(med_v, min_v), max_v)
        return min_v, max_v, med_v

    gross_min, gross_max, gross_med = get_num_bounds("GrossSquareMeters", 30, 300, 120)
    net_min, net_max, net_med = get_num_bounds("NetSquareMeters", 20, 260, 95)
    room_min, room_max, room_med = get_num_bounds("NumberOfRooms", 1, 8, 4)
    age_min, age_max, age_med = get_num_bounds("BuildingAge", 0, 30, 7)
    floor_min, floor_max, floor_med = get_num_bounds("NumberFloorsofBuilding", 1, 40, 6)
    bath_min, bath_max, bath_med = get_num_bounds("NumberOfBathrooms", 1, 5, 1)
    wc_min, wc_max, wc_med = get_num_bounds("NumberOfWCs", 0, 4, 1)

    st.subheader("📍 Konum ve Temel Ozellikler")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        district = st.selectbox(
            "🏘 Ilce",
            res["districts"],
            index=res["districts"].index("kadikoy") if "kadikoy" in res["districts"] else 0,
        )
    with c2:
        gross_m2 = st.slider("📐 Brut m2", gross_min, gross_max, gross_med)
        net_default = min(max(int(gross_m2 * 0.8), net_min), min(net_max, gross_m2))
        net_m2 = st.slider("📏 Net m2", net_min, min(net_max, gross_m2), net_default)
    with c3:
        rooms = st.slider("🚪 Oda Sayisi", room_min, room_max, room_med)
        building_age = st.slider("🏗 Bina Yasi", age_min, age_max, age_med)
    with c4:
        bathrooms = st.slider("🚿 Banyo Sayisi", bath_min, bath_max, bath_med)
        wcs = st.slider("🚽 WC Sayisi", wc_min, wc_max, wc_med)
        budget = st.number_input("💰 Butce (Opsiyonel)", min_value=0, value=0, step=250000)

    st.subheader("⚙ Yapisal ve Donanim Ozellikleri")
    d1, d2, d3 = st.columns(3)
    with d1:
        floors = st.slider("🏢 Bina Kat Sayisi", floor_min, floor_max, floor_med)
        floor_loc_opts = res["cat_options"].get("FloorLocation", ["Giris", "Ara Kat", "Cati Kati"])
        floor_loc_default = category_defaults.get("FloorLocation", floor_loc_opts[0])
        floor_loc = st.selectbox(
            "📍 Bulundugu Kat",
            floor_loc_opts,
            index=floor_loc_opts.index(floor_loc_default) if floor_loc_default in floor_loc_opts else 0,
        )

    with d2:
        heating_opts = res["cat_options"].get("HeatingType", ["Kombi", "Merkezi", "Soba"])
        heating_default = category_defaults.get("HeatingType", heating_opts[0])
        heating = st.selectbox(
            "🌡 Isitma Turu",
            heating_opts,
            index=heating_opts.index(heating_default) if heating_default in heating_opts else 0,
        )
        using_opts = res["cat_options"].get("UsingStatus", ["Bos"])
        using_default = category_defaults.get("UsingStatus", using_opts[0])
        using_status = st.selectbox(
            "🛋 Kullanim Durumu",
            using_opts,
            index=using_opts.index(using_default) if using_default in using_opts else 0,
        )
        inside_site_opts = res["cat_options"].get("InsideTheSite", ["Evet", "Hayir"])
        inside_site_default = category_defaults.get("InsideTheSite", inside_site_opts[0])
        inside_site = st.selectbox(
            "🏘 Site Icinde mi?",
            inside_site_opts,
            index=inside_site_opts.index(inside_site_default) if inside_site_default in inside_site_opts else 0,
        )

    with d3:
        build_status_opts = res["cat_options"].get("BuildStatus", ["Ikinci El"])
        build_status_default = category_defaults.get("BuildStatus", build_status_opts[0])
        build_status = st.selectbox(
            "🏗 Yapi Durumu",
            build_status_opts,
            index=build_status_opts.index(build_status_default) if build_status_default in build_status_opts else 0,
        )
        item_status_opts = res["cat_options"].get("ItemStatus", ["Bos"])
        item_status_default = category_defaults.get("ItemStatus", item_status_opts[0])
        item_status = st.selectbox(
            "🪑 Esya Durumu",
            item_status_opts,
            index=item_status_opts.index(item_status_default) if item_status_default in item_status_opts else 0,
        )
        structure_opts = res["cat_options"].get("StructureType", ["Betonarme"])
        structure_default = category_defaults.get("StructureType", structure_opts[0])
        structure_type = st.selectbox(
            "🏛 Yapi Tipi",
            structure_opts,
            index=structure_opts.index(structure_default) if structure_default in structure_opts else 0,
        )

    with st.expander("➕ Finansman ve Ek Ozellikler"):
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            credit_opts = res["cat_options"].get("CreditEligibility", ["Bilinmiyor"])
            credit_default = category_defaults.get("CreditEligibility", credit_opts[0])
            credit = st.selectbox(
                "🏦 Krediye Uygunluk",
                credit_opts,
                index=credit_opts.index(credit_default) if credit_default in credit_opts else 0,
            )
        with e2:
            invest_opts = res["cat_options"].get("EligibilityForInvestment", ["Bilinmiyor"])
            invest_default = category_defaults.get("EligibilityForInvestment", invest_opts[0])
            investment = st.selectbox(
                "📈 Yatirima Uygunluk",
                invest_opts,
                index=invest_opts.index(invest_default) if invest_default in invest_opts else 0,
            )
        with e3:
            swap_opts = res["cat_options"].get("Swap", ["Hayir"])
            swap_default = category_defaults.get("Swap", swap_opts[0])
            swap = st.selectbox(
                "🔄 Takas Durumu",
                swap_opts,
                index=swap_opts.index(swap_default) if swap_default in swap_opts else 0,
            )
        with e4:
            balcony_opts = res["cat_options"].get("Balcony", ["Yok"])
            balcony_default = category_defaults.get("Balcony", balcony_opts[0])
            balcony = st.selectbox(
                "🌇 Balkon",
                balcony_opts,
                index=balcony_opts.index(balcony_default) if balcony_default in balcony_opts else 0,
            )

    st.markdown("---")

    if st.button("🔍 Tahmin Uret", use_container_width=True, type="primary"):
        feature_vals = res.get("feature_defaults", {}).copy()
        if not feature_vals:
            feature_vals = {feat: float(res["X_df"][feat].median()) for feat in res["feature_names"]}

        feature_vals["GrossSquareMeters"] = gross_m2
        feature_vals["NetSquareMeters"] = net_m2
        feature_vals["NumberOfRooms"] = rooms
        feature_vals["NumberOfBathrooms"] = bathrooms
        feature_vals["NumberOfWCs"] = wcs
        feature_vals["NumberFloorsofBuilding"] = floors
        feature_vals["BuildingAge"] = building_age

        if "NetGrossRatio" in feature_vals:
            feature_vals["NetGrossRatio"] = net_m2 / max(gross_m2, 1)
        if "RoomDensity" in feature_vals:
            feature_vals["RoomDensity"] = rooms / max(net_m2, 1)
        if "TotalWetAreas" in feature_vals:
            feature_vals["TotalWetAreas"] = bathrooms + wcs
        if "FloorsPerRoom" in feature_vals:
            feature_vals["FloorsPerRoom"] = floors / max(rooms, 1)

        if "district" in feature_vals and "district" in res["encoders"]:
            district_map = res["encoders"]["district"]
            feature_vals["district"] = district_map.get(district, np.median(list(district_map.values())))

        categorical_user_inputs = {
            "FloorLocation": floor_loc,
            "HeatingType": heating,
            "InsideTheSite": inside_site,
            "UsingStatus": using_status,
            "EligibilityForInvestment": investment,
            "BuildStatus": build_status,
            "ItemStatus": item_status,
            "CreditEligibility": credit,
            "StructureType": structure_type,
            "Swap": swap,
            "Balcony": balcony,
        }
        for col_name, selected_val in categorical_user_inputs.items():
            if col_name in feature_vals and col_name in res["encoders"]:
                le = res["encoders"][col_name]
                if selected_val in le.classes_:
                    feature_vals[col_name] = int(le.transform([selected_val])[0])
                else:
                    default_val = category_defaults.get(col_name, le.classes_[0])
                    feature_vals[col_name] = int(le.transform([default_val])[0])

        X_new = pd.DataFrame([feature_vals])[res["feature_names"]]
        X_new_scaled = pd.DataFrame(res["scaler"].transform(X_new), columns=res["feature_names"])

        xgb_proba = res["xgb_model"].predict_proba(X_new_scaled)[0]
        rf_proba = res["rf_model"].predict_proba(X_new_scaled)[0]
        cls_proba = (0.60 * xgb_proba) + (0.40 * rf_proba)
        cls_idx = int(np.argmax(cls_proba))
        sinif = res["class_names"][cls_idx]

        model_price = float(res["reg_model"].predict(X_new_scaled)[0])
        district_stats = res["district_stats"]
        district_row = district_stats[district_stats["district"] == district]
        district_price = model_price
        district_q25, district_q75 = model_price * 0.85, model_price * 1.15
        if not district_row.empty:
            district_m2 = float(district_row["medyan_m2"].iloc[0])
            district_price = district_m2 * max(net_m2, 1)
            district_q25 = float(district_row["q25"].iloc[0])
            district_q75 = float(district_row["q75"].iloc[0])

        fiyat_tahmin = (0.70 * model_price) + (0.30 * district_price)
        mae = float(res.get("reg_metrics", {}).get("MAE", fiyat_tahmin * 0.12))
        band = mae + max(district_q75 - district_q25, 0) * 0.20
        fiyat_alt = max(0, fiyat_tahmin - band)
        fiyat_ust = fiyat_tahmin + band

        st.markdown("---")
        emoji_map = {"Ekonomik": "💚", "Orta": "🟡", "Luks": "💎"}
        color_map = {"Ekonomik": "#2ecc71", "Orta": "#f39c12", "Luks": "#e74c3c"}

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-val" style="color:{color_map.get(sinif, '#fff')}">{emoji_map.get(sinif, '')} {sinif}</div>
                <div class="metric-label">Tahmini Sinif</div></div>""",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-val" style="font-size:1.4rem">₺{fiyat_tahmin:,.0f}</div>
                <div class="metric-label">Tahmini Fiyat</div></div>""",
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-val" style="font-size:1.05rem">₺{fiyat_alt:,.0f} — ₺{fiyat_ust:,.0f}</div>
                <div class="metric-label">Guven Araligi</div></div>""",
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-val" style="font-size:1.2rem">₺{fiyat_tahmin/max(net_m2,1):,.0f}</div>
                <div class="metric-label">Tahmini m² Fiyati</div></div>""",
                unsafe_allow_html=True,
            )

        class_range = res["class_ranges"].get(sinif, {})
        if class_range:
            st.info(
                f"**{sinif} Sinifi Fiyat Bandi (Egitim Verisi):** "
                f"₺{class_range['min']:,.0f} - ₺{class_range['max']:,.0f} | Medyan: ₺{class_range['median']:,.0f}"
            )
        if budget > 0:
            fark = budget - fiyat_tahmin
            if fark >= 0:
                st.success(f"Butceye Gore Yaklasik **₺{fark:,.0f}** Pay Kaliyor.")
            else:
                st.error(f"Butceye Gore Yaklasik **₺{abs(fark):,.0f}** Ek Kaynak Gerekiyor.")

        tab_s, tab_i, tab_b = st.tabs(["📊 Olasiliklar", "🏘 Ilce Analizi", "🔍 Benzer ve Alternatifler"])

        with tab_s:
            desired_order = [x for x in ["Ekonomik", "Orta", "Luks"] if x in res["class_names"]]
            class_prob_df = pd.DataFrame({"Sinif": res["class_names"], "Olasilik": cls_proba})
            class_prob_df = class_prob_df.set_index("Sinif").reindex(desired_order).reset_index()
            bar_colors = [color_map.get(lbl, "#3498db") for lbl in class_prob_df["Sinif"]]
            fig = go.Figure(
                go.Bar(
                    x=class_prob_df["Olasilik"],
                    y=class_prob_df["Sinif"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"%{p*100:.1f}" for p in class_prob_df["Olasilik"]],
                    textposition="outside",
                )
            )
            fig.update_layout(title="Sinif Olasiliklari (XGB %60 + RF %40)", height=280, xaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Regresyon MAE: ₺{res.get('reg_metrics', {}).get('MAE', 0):,.0f} | "
                f"R²: {res.get('reg_metrics', {}).get('R2', 0):.3f}"
            )

        with tab_i:
            ds = res["district_stats"]
            ilce_row = ds[ds["district"] == district]
            if not ilce_row.empty:
                ir = ilce_row.iloc[0]
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric(f"📍 {district} Medyan", f"₺{ir['medyan']:,.0f}")
                ci2.metric("Ilce Ortalama", f"₺{ir['ortalama']:,.0f}")
                ci3.metric("Ilce Medyan m²", f"₺{ir['medyan_m2']:,.0f}/m²")
                ci4.metric("Ilan Sayisi", f"{int(ir['ilan_sayisi']):,}")

                pct = (fiyat_tahmin - ir["min"]) / max(ir["max"] - ir["min"], 1) * 100
                pct = min(max(pct, 0), 100)
                st.progress(int(pct), text=f"Tahmininiz {district} dagiliminda %{pct:.0f} seviyesinde")

            top10 = ds.nlargest(10, "ilan_sayisi").sort_values("medyan", ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=top10["medyan"],
                    y=top10["district"],
                    orientation="h",
                    marker_color=["#e74c3c" if d == district else "#3498db" for d in top10["district"]],
                    text=[f"₺{v:,.0f}" for v in top10["medyan"]],
                    textposition="outside",
                )
            )
            fig.update_layout(title="En Cok Ilan Olan 10 Ilce - Medyan Fiyat", height=420, xaxis_title="TL")
            st.plotly_chart(fig, use_container_width=True)

        with tab_b:
            data_df = res["df"]
            benzer = data_df[data_df["district"] == district].copy()
            benzer["sim_score"] = (
                (benzer["GrossSquareMeters"] - gross_m2).abs() / max(gross_m2, 1) * 0.25
                + (benzer["NetSquareMeters"] - net_m2).abs() / max(net_m2, 1) * 0.30
                + (benzer["NumberOfRooms"] - rooms).abs() / max(rooms, 1) * 0.20
                + (benzer["BuildingAge"] - building_age).abs() / max(building_age + 1, 1) * 0.10
                + (benzer["NumberOfBathrooms"] - bathrooms).abs() / max(bathrooms, 1) * 0.15
            )
            benzer = benzer.sort_values("sim_score").head(20)
            if len(benzer) > 0:
                st.success(f"✅ {district} icin en yakin **{len(benzer)}** ilan listelendi.")
                benzer_show = benzer[
                    ["district", "price", "GrossSquareMeters", "NetSquareMeters", "NumberOfRooms", "BuildingAge", "PriceClass", "sim_score"]
                ].copy()
                benzer_show.columns = ["Ilce", "Fiyat (TL)", "Brut m2", "Net m2", "Oda", "Bina Yasi", "Sinif", "Benzerlik Skoru"]
                benzer_show["Fiyat (TL)"] = benzer_show["Fiyat (TL)"].apply(lambda x: f"₺{x:,.0f}")
                benzer_show["Benzerlik Skoru"] = benzer_show["Benzerlik Skoru"].round(3)
                st.dataframe(benzer_show, use_container_width=True, hide_index=True)

                benzer_median = benzer["price"].median()
                st.metric(
                    "Benzer Ilan Medyani",
                    f"₺{benzer_median:,.0f}",
                    delta=f"Tahmin farki: ₺{fiyat_tahmin - benzer_median:,.0f}",
                )

            alt_df = res["district_stats"][["district", "medyan_m2"]].copy()
            alt_df["bu_ozellikte_tahmini"] = alt_df["medyan_m2"] * max(net_m2, 1)
            alt_df["fiyat_farki"] = alt_df["bu_ozellikte_tahmini"] - fiyat_tahmin
            alt_df = alt_df.sort_values("bu_ozellikte_tahmini").head(10)
            alt_df.columns = ["Ilce", "Medyan m²", "Bu Ozellikte Tahmini Fiyat", "Mevcut Tahminden Fark"]
            alt_df["Medyan m²"] = alt_df["Medyan m²"].map(lambda x: f"₺{x:,.0f}/m²")
            alt_df["Bu Ozellikte Tahmini Fiyat"] = alt_df["Bu Ozellikte Tahmini Fiyat"].map(lambda x: f"₺{x:,.0f}")
            alt_df["Mevcut Tahminden Fark"] = alt_df["Mevcut Tahminden Fark"].map(lambda x: f"₺{x:,.0f}")
            st.markdown("**Ayni net m² icin potansiyel daha uygun ilceler (medyan m² bazli):**")
            st.dataframe(alt_df, use_container_width=True, hide_index=True)

# ========== PAGE: TEKNIK BILGI ==========
elif page == "📋 Teknik Bilgi":
    st.header("📋 Teknik Bilgi ve Metodoloji")
    
    st.markdown("""
    ### Pipeline Ozeti
    1. **Veri Yukleme:** Kaggle'dan 25,000+ ilan
    2. **Temizlik:** Fiyat/m2/oda parsing, NaN doldurma (medyan/mod)
    3. **Enflasyon:** TUIK 2022-2026 TUFE ile 2026'ya guncelleme
    4. **Outlier:** IQR yontemiyle aykiri deger temizligi
    5. **Feature Eng.:** NetGrossRatio, RoomDensity, TotalWetAreas, FloorsPerRoom
    6. **Encoding:** Frequency (district) + Label (diger kategorik)
    7. **Scaling:** StandardScaler
    8. **Siniflandirma:** pd.qcut ile 3 sinif (Ekonomik/Orta/Luks)
    9. **Modeller:** RF (500 agac) + XGBoost (500 tur, lr=0.05)
    10. **Degerlendirme:** Accuracy, F1, Precision, Recall
    """)
    
    st.markdown("### Enflasyon Carpanlari")
    enf_df = pd.DataFrame({"Yil":[2022,2023,2024,2025,2026], "TUFE (%)":[64.27,64.77,44.38,30.65,7.5]})
    st.dataframe(enf_df, use_container_width=True, hide_index=True)
    
    st.markdown("### Kullanilan Kutuphaneler")
    st.code("pandas, numpy, scikit-learn, xgboost, streamlit, plotly, matplotlib, seaborn, kagglehub")

st.sidebar.markdown("---")
st.sidebar.caption("🎓 Makine Ogrenmesi Final Projesi | 2026")
