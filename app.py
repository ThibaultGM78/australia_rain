"""
🌧️ Australian Rain Prediction Dashboard
Features exactes après cleaning :
  Gardées (raw)  : MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am,
                   WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am,
                   Pressure3pm, Temp9am, Temp3pm, RainToday
  Droppées       : Evaporation, Sunshine, WindGustDir, WindDir9am,
                   Cloud9am, Cloud3pm
  Engineerées    : TempRange, Humidity_Avg, Pressure_Diff,
                   Month_sin, Month_cos,
                   WindDir3pm_sin, WindDir3pm_cos,
                   City_Encoded
"""

import warnings, os, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌧️ Rain Prediction – Australia",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "XGBoost":             "saved_models/xgboost.joblib",
    "Logistic Regression": "saved_models/logistic_regression.joblib",
}

WIND_DIRS = {
    "N":0,"NNE":22.5,"NE":45,"ENE":67.5,
    "E":90,"ESE":112.5,"SE":135,"SSE":157.5,
    "S":180,"SSW":202.5,"SW":225,"WSW":247.5,
    "W":270,"WNW":292.5,"NW":315,"NNW":337.5,
}

MONTHS = {
    "Janvier":1,"Février":2,"Mars":3,"Avril":4,
    "Mai":5,"Juin":6,"Juillet":7,"Août":8,
    "Septembre":9,"Octobre":10,"Novembre":11,"Décembre":12,
}

# Médianes australiennes (valeurs par défaut des sliders)
BASELINE = {
    "MinTemp":      12.0,
    "MaxTemp":      23.0,
    "Rainfall":      0.0,
    "WindGustSpeed":39.0,
    "WindSpeed9am": 14.0,
    "WindSpeed3pm": 19.0,
    "Humidity9am":  69.0,
    "Humidity3pm":  51.0,
    "Pressure9am": 1017.6,
    "Pressure3pm": 1015.3,
    "Temp9am":      16.9,
    "Temp3pm":      21.7,
}

# Metadata pour les sliders
SLIDER_META = {
    "MinTemp":       {"min":-10,  "max":45,   "step":0.5, "unit":"°C",   "icon":"🌡️"},
    "MaxTemp":       {"min":-5,   "max":50,   "step":0.5, "unit":"°C",   "icon":"🌡️"},
    "Rainfall":      {"min":0,    "max":100,  "step":0.5, "unit":"mm",   "icon":"🌧️"},
    "WindGustSpeed": {"min":0,    "max":140,  "step":1.0, "unit":"km/h", "icon":"💨"},
    "WindSpeed9am":  {"min":0,    "max":80,   "step":1.0, "unit":"km/h", "icon":"💨"},
    "WindSpeed3pm":  {"min":0,    "max":80,   "step":1.0, "unit":"km/h", "icon":"💨"},
    "Humidity9am":   {"min":0,    "max":100,  "step":1.0, "unit":"%",    "icon":"💦"},
    "Humidity3pm":   {"min":0,    "max":100,  "step":1.0, "unit":"%",    "icon":"💦"},
    "Pressure9am":   {"min":980,  "max":1040, "step":0.5, "unit":"hPa",  "icon":"🔵"},
    "Pressure3pm":   {"min":980,  "max":1040, "step":0.5, "unit":"hPa",  "icon":"🔵"},
    "Temp9am":       {"min":-5,   "max":45,   "step":0.5, "unit":"°C",   "icon":"🌡️"},
    "Temp3pm":       {"min":-5,   "max":50,   "step":0.5, "unit":"°C",   "icon":"🌡️"},
}

SLIDER_GROUPS = {
    "🌡️ Températures":   ["MinTemp","MaxTemp","Temp9am","Temp3pm"],
    "🌧️ Précipitations": ["Rainfall"],
    "💨 Vent":            ["WindGustSpeed","WindSpeed9am","WindSpeed3pm"],
    "💦 Humidité":        ["Humidity9am","Humidity3pm"],
    "🔵 Pression":        ["Pressure9am","Pressure3pm"],
}

# Villes avec coordonnées et encodage (ordre alphabétique = label encoding classique)
AU_CITIES = {
    "Adelaide":         {"lat":-34.93,"lon":138.60,"enc":0},
    "Albany":           {"lat":-35.02,"lon":117.88,"enc":1},
    "Albury":           {"lat":-36.07,"lon":146.91,"enc":2},
    "AliceSprings":     {"lat":-23.70,"lon":133.88,"enc":3},
    "BadgerysCreek":    {"lat":-33.88,"lon":150.74,"enc":4},
    "Ballarat":         {"lat":-37.56,"lon":143.86,"enc":5},
    "Bendigo":          {"lat":-36.76,"lon":144.28,"enc":6},
    "Brisbane":         {"lat":-27.47,"lon":153.02,"enc":7},
    "Cairns":           {"lat":-16.92,"lon":145.77,"enc":8},
    "Canberra":         {"lat":-35.28,"lon":149.13,"enc":9},
    "Cobar":            {"lat":-31.50,"lon":145.84,"enc":10},
    "CoffsHarbour":     {"lat":-30.30,"lon":153.11,"enc":11},
    "Dartmoor":         {"lat":-37.92,"lon":141.27,"enc":12},
    "Darwin":           {"lat":-12.46,"lon":130.84,"enc":13},
    "GoldCoast":        {"lat":-28.00,"lon":153.43,"enc":14},
    "Hobart":           {"lat":-42.88,"lon":147.33,"enc":15},
    "Katherine":        {"lat":-14.46,"lon":132.27,"enc":16},
    "Launceston":       {"lat":-41.43,"lon":147.14,"enc":17},
    "Melbourne":        {"lat":-37.81,"lon":144.96,"enc":18},
    "MelbourneAirport": {"lat":-37.67,"lon":144.84,"enc":19},
    "Mildura":          {"lat":-34.22,"lon":142.16,"enc":20},
    "Moree":            {"lat":-29.46,"lon":149.85,"enc":21},
    "MountGambier":     {"lat":-37.83,"lon":140.78,"enc":22},
    "MountGinini":      {"lat":-35.53,"lon":148.77,"enc":23},
    "Nhil":             {"lat":-36.33,"lon":141.65,"enc":24},
    "NorahHead":        {"lat":-33.28,"lon":151.57,"enc":25},
    "NorfolkIsland":    {"lat":-29.05,"lon":167.96,"enc":26},
    "Nuriootpa":        {"lat":-34.47,"lon":138.99,"enc":27},
    "PearceRAAF":       {"lat":-31.67,"lon":116.02,"enc":28},
    "Penrith":          {"lat":-33.75,"lon":150.69,"enc":29},
    "Perth":            {"lat":-31.95,"lon":115.86,"enc":30},
    "PerthAirport":     {"lat":-31.94,"lon":115.97,"enc":31},
    "Portland":         {"lat":-38.34,"lon":141.60,"enc":32},
    "Richmond":         {"lat":-33.60,"lon":150.75,"enc":33},
    "Sale":             {"lat":-38.10,"lon":147.07,"enc":34},
    "SalmonGums":       {"lat":-32.98,"lon":121.64,"enc":35},
    "Sydney":           {"lat":-33.87,"lon":151.21,"enc":36},
    "SydneyAirport":    {"lat":-33.94,"lon":151.18,"enc":37},
    "Townsville":       {"lat":-19.26,"lon":146.82,"enc":38},
    "Tuggeranong":      {"lat":-35.42,"lon":149.09,"enc":39},
    "Uluru":            {"lat":-25.34,"lon":131.04,"enc":40},
    "WaggaWagga":       {"lat":-35.16,"lon":147.37,"enc":41},
    "Walpole":          {"lat":-34.98,"lon":116.73,"enc":42},
    "Watsonia":         {"lat":-37.71,"lon":145.08,"enc":43},
    "Williamtown":      {"lat":-32.80,"lon":151.84,"enc":44},
    "Witchcliffe":      {"lat":-33.90,"lon":115.10,"enc":45},
    "Wollongong":       {"lat":-34.42,"lon":150.89,"enc":46},
    "Woomera":          {"lat":-31.16,"lon":136.82,"enc":47},
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  ← doit reproduire exactement le preprocessing training
# ─────────────────────────────────────────────────────────────────────────────
def build_row(sliders: dict, month: int, wind_dir: str,
              rain_today: int, city_enc: int) -> pd.DataFrame:
    """
    Construit le vecteur de features complet attendu par le modèle.

    Features raw gardées (13) :
        MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm,
        Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Temp9am, Temp3pm,
        RainToday

    Features engineerées (8) :
        TempRange      = MaxTemp - MinTemp
        Humidity_Avg   = (Humidity9am + Humidity3pm) / 2
        Pressure_Diff  = Pressure9am - Pressure3pm
        Month_sin/cos  = encodage cyclique du mois
        WindDir3pm_sin/cos = encodage cyclique de la direction
        City_Encoded   = label encoding de la ville
    """
    deg = WIND_DIRS.get(wind_dir, 0.0)

    return pd.DataFrame([{
        # ── Raw ──────────────────────────────────────────────────────────────
        "MinTemp":        sliders["MinTemp"],
        "MaxTemp":        sliders["MaxTemp"],
        "Rainfall":       sliders["Rainfall"],
        "WindGustSpeed":  sliders["WindGustSpeed"],
        "WindSpeed9am":   sliders["WindSpeed9am"],
        "WindSpeed3pm":   sliders["WindSpeed3pm"],
        "Humidity9am":    sliders["Humidity9am"],
        "Humidity3pm":    sliders["Humidity3pm"],
        "Pressure9am":    sliders["Pressure9am"],
        "Pressure3pm":    sliders["Pressure3pm"],
        "Temp9am":        sliders["Temp9am"],
        "Temp3pm":        sliders["Temp3pm"],
        "RainToday":      float(rain_today),
        # ── Engineerées ──────────────────────────────────────────────────────
        "TempRange":      sliders["MaxTemp"]     - sliders["MinTemp"],
        "Humidity_Avg":  (sliders["Humidity9am"] + sliders["Humidity3pm"]) / 2,
        "Pressure_Diff":  sliders["Pressure9am"] - sliders["Pressure3pm"],
        "Month_sin":      math.sin(2 * math.pi * month / 12),
        "Month_cos":      math.cos(2 * math.pi * month / 12),
        "WindDir3pm_sin": math.sin(math.radians(deg)),
        "WindDir3pm_cos": math.cos(math.radians(deg)),
        "City_Encoded":   float(city_enc),
    }])


def align_to_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """Réordonne / filtre les colonnes selon ce qu'attend le modèle."""
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                expected = list(step.feature_names_in_)
                break
    if expected is None:
        return df
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected]


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name: str):
    path = MODEL_PATHS[name]
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_proba(model, df: pd.DataFrame) -> float:
    df = align_to_model(df, model)
    return float(model.predict_proba(df)[:, 1][0])


def predict_all_cities(model, sliders, month, wind_dir, rain_today) -> pd.DataFrame:
    dfs, cities = [], list(AU_CITIES.keys())
    for city in cities:
        dfs.append(build_row(sliders, month, wind_dir, rain_today,
                             AU_CITIES[city]["enc"]))
    big = pd.concat(dfs, ignore_index=True)
    big = align_to_model(big, model)
    probs = model.predict_proba(big)[:, 1]
    return pd.DataFrame([{
        "ville":        c,
        "lat":          AU_CITIES[c]["lat"],
        "lon":          AU_CITIES[c]["lon"],
        "prob_pluie":   round(float(p), 3),
        "pluie_prevue": "🌧️ Pluie" if p >= 0.5 else "☀️ Sec",
    } for c, p in zip(cities, probs)])


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────
def make_map(df):
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="prob_pluie",
        size=[14]*len(df), size_max=18,
        hover_name="ville",
        hover_data={"prob_pluie":":.1%","pluie_prevue":True,"lat":False,"lon":False},
        color_continuous_scale=[[0,"#27ae60"],[0.35,"#f1c40f"],
                                 [0.5,"#e67e22"],[1,"#2980b9"]],
        range_color=[0,1], mapbox_style="carto-positron",
        zoom=3.5, center={"lat":-25.5,"lon":134.0},
        labels={"prob_pluie":"P(pluie)"},
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(title="P(pluie)", tickformat=".0%", len=0.7),
        height=520,
    )
    return fig


def make_tornado(model, sliders, month, wind_dir, rain_today, n=10):
    base_df   = build_row(sliders, month, wind_dir, rain_today, 0)
    base_prob = predict_proba(model, base_df)

    rows = []
    for feat, meta in SLIDER_META.items():
        std   = (meta["max"] - meta["min"]) / 6
        cur   = sliders[feat]
        lo_v  = max(meta["min"], cur - std)
        hi_v  = min(meta["max"], cur + std)

        s_lo  = {**sliders, feat: lo_v}
        s_hi  = {**sliders, feat: hi_v}
        p_lo  = predict_proba(model, build_row(s_lo, month, wind_dir, rain_today, 0))
        p_hi  = predict_proba(model, build_row(s_hi, month, wind_dir, rain_today, 0))

        rows.append({
            "feature": f"{meta['icon']} {feat}",
            "p_low": p_lo, "p_high": p_hi,
            "swing": abs(p_hi - p_lo),
        })

    df = pd.DataFrame(rows).sort_values("swing", ascending=True).tail(n)

    fig = go.Figure()
    for r in df.itertuples():
        lo_p  = min(r.p_low, r.p_high)
        hi_p  = max(r.p_low, r.p_high)
        color = "#2980b9" if r.p_high >= r.p_low else "#e74c3c"
        fig.add_trace(go.Bar(
            x=[hi_p - lo_p], y=[r.feature], base=lo_p,
            orientation="h", marker_color=color, marker_opacity=0.85,
            showlegend=False,
            hovertemplate=(f"<b>{r.feature}</b><br>"
                           f"Bas: {lo_p:.1%}<br>Haut: {hi_p:.1%}<extra></extra>"),
        ))

    fig.add_vline(x=base_prob, line_dash="dash", line_color="#888", line_width=1.5,
                  annotation_text=f"Actuel {base_prob:.0%}",
                  annotation_position="top right")
    fig.update_layout(
        xaxis=dict(tickformat=".0%", range=[0,1], title="P(pluie)"),
        yaxis_title=None, height=400,
        margin=dict(l=10,r=10,t=30,b=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig, base_prob


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]   { background:#1a1f2e; }
[data-testid="stSidebar"] * { color:#e8eaf0 !important; }
[data-testid="stSidebar"] .stRadio label { color:#e8eaf0 !important; }
.metric-card{
  background:linear-gradient(135deg,#1e3a5f,#2980b9);
  border-radius:12px;padding:16px 20px;text-align:center;
  box-shadow:0 4px 12px rgba(0,0,0,.2);margin-bottom:6px;
}
.metric-card .lbl{font-size:11px;color:#a8c8e8;text-transform:uppercase;letter-spacing:.05em}
.metric-card .val{font-size:2rem;font-weight:700;color:#fff;line-height:1.2}
.metric-card .sub{font-size:12px;color:#c8ddf0;margin-top:4px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌏 Rain Predictor\n*Australia*")
    st.markdown("---")

    # ── Modèle ────────────────────────────────────────────────────────────────
    st.markdown("### 🤖 Modèle")
    model_name = st.radio("", list(MODEL_PATHS.keys()))

    st.markdown("---")

    # ── Contexte (features catégorielles / temporelles) ───────────────────────
    st.markdown("### 📅 Contexte")

    month_label  = st.selectbox("📆 Mois", list(MONTHS.keys()),
                                index=list(MONTHS.keys()).index("Juin"))
    month_val    = MONTHS[month_label]

    wind_dir_val = st.selectbox("🧭 Direction vent 3pm", list(WIND_DIRS.keys()))

    rain_today   = st.radio("🌦️ A-t-il plu aujourd'hui ?",
                            ["Non ☀️", "Oui 🌧️"], horizontal=True)
    rain_today_val = 1 if rain_today.startswith("Oui") else 0

    st.markdown("---")

    # ── Sliders météo ─────────────────────────────────────────────────────────
    st.markdown("### 🎚️ Paramètres météo")
    st.caption("Tous les autres paramètres (Evaporation, Sunshine, Cloud…)\n"
               "ont été supprimés lors du cleaning.")

    sliders = {}
    for group_label, feats in SLIDER_GROUPS.items():
        with st.expander(group_label,
                         expanded=(group_label == "🌡️ Températures")):
            for feat in feats:
                m = SLIDER_META[feat]
                sliders[feat] = st.slider(
                    f"{m['icon']} {feat} ({m['unit']})",
                    min_value=float(m["min"]),
                    max_value=float(m["max"]),
                    value=float(BASELINE[feat]),
                    step=float(m["step"]),
                    key=f"sl_{feat}",
                )

    st.markdown("---")
    if st.button("↩️ Réinitialiser les sliders", use_container_width=True):
        for feat in SLIDER_META:
            st.session_state[f"sl_{feat}"] = float(BASELINE[feat])
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌧️ Australian Rain Prediction Dashboard")
st.caption("Modifiez les paramètres dans la barre latérale → la carte se met à jour en temps réel.")

# Chargement modèle
model = load_model(model_name)
if model is None:
    st.error(
        f"❌ Modèle introuvable : `{MODEL_PATHS[model_name]}`\n\n"
        "Assurez-vous que vos fichiers `.joblib` sont dans le dossier `saved_models/`."
    )
    st.stop()

# ── Calculs ───────────────────────────────────────────────────────────────────
df_cities = predict_all_cities(model, sliders, month_val, wind_dir_val, rain_today_val)

# Prédiction "générique" (City_Encoded=0, i.e. Adelaide comme référence)
ref_df    = build_row(sliders, month_val, wind_dir_val, rain_today_val, 0)
base_prob = predict_proba(model, ref_df)
is_rain   = base_prob >= 0.5

n_rain   = (df_cities["prob_pluie"] >= 0.5).sum()
avg_prob = df_cities["prob_pluie"].mean()
max_row  = df_cities.loc[df_cities["prob_pluie"].idxmax()]
min_row  = df_cities.loc[df_cities["prob_pluie"].idxmin()]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
rain_color = "#2980b9" if is_rain else "#27ae60"

with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="lbl">Modèle actif</div>
      <div class="val" style="font-size:1.1rem">{model_name}</div>
      <div class="sub">{month_label} · Vent {wind_dir_val}<br>
           Pluie aujourd'hui : {'Oui' if rain_today_val else 'Non'}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    emoji = "🌧️" if is_rain else "☀️"
    st.markdown(f"""<div class="metric-card"
        style="background:linear-gradient(135deg,#1e3a5f,{rain_color})">
      <div class="lbl">Prédiction (paramètres actuels)</div>
      <div class="val">{base_prob:.0%} {emoji}</div>
      <div class="sub">{'Pluie probable demain' if is_rain else 'Temps sec probable demain'}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="lbl">Villes avec pluie prévue</div>
      <div class="val">{n_rain} / {len(df_cities)}</div>
      <div class="sub">Moyenne nationale : {avg_prob:.0%}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="metric-card">
      <div class="lbl">Extrêmes</div>
      <div class="val" style="font-size:1rem">
        🌧️ {max_row['ville']} {max_row['prob_pluie']:.0%}<br>
        ☀️ {min_row['ville']} {min_row['prob_pluie']:.0%}
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Carte + Tornado ───────────────────────────────────────────────────────────
col_map, col_tor = st.columns([3, 2], gap="large")

with col_map:
    st.markdown("#### 🗺️ Carte des probabilités de pluie")
    st.caption("Vert = sec · Jaune/orange = risque modéré · Bleu = pluie probable")
    st.plotly_chart(make_map(df_cities), use_container_width=True)

with col_tor:
    st.markdown("#### 🌪️ Sensibilité des paramètres (±1 σ)")
    st.caption("Quelle feature fait le plus varier P(pluie) si on la modifie ?")
    fig_tor, _ = make_tornado(model, sliders, month_val, wind_dir_val, rain_today_val)
    st.plotly_chart(fig_tor, use_container_width=True)

# ── Debug : features envoyées au modèle ───────────────────────────────────────
with st.expander("🔍 Vecteur de features envoyé au modèle (debug)"):
    debug_df = align_to_model(
        build_row(sliders, month_val, wind_dir_val, rain_today_val, 0), model
    )
    st.dataframe(
        debug_df.T.rename(columns={0:"Valeur"}).style.format("{:.4f}"),
        use_container_width=True,
    )

# ── Tableau filtrable ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 Résultats par ville")

fa, fb = st.columns([2, 1])
with fa:
    search = st.text_input("🔍 Rechercher une ville", placeholder="Sydney, Darwin…")
with fb:
    filtre = st.selectbox("Afficher", ["Toutes les villes", "🌧️ Pluie uniquement", "☀️ Sec uniquement"])

df_show = df_cities.copy()
if search:
    df_show = df_show[df_show["ville"].str.contains(search, case=False)]
if filtre == "🌧️ Pluie uniquement":
    df_show = df_show[df_show["prob_pluie"] >= 0.5]
elif filtre == "☀️ Sec uniquement":
    df_show = df_show[df_show["prob_pluie"] < 0.5]

df_show = df_show.sort_values("prob_pluie", ascending=False).reset_index(drop=True)
df_show["P(pluie)"] = df_show["prob_pluie"].apply(lambda x: f"{x:.1%}")

st.dataframe(
    df_show[["ville","P(pluie)","pluie_prevue"]].rename(
        columns={"ville":"Ville","pluie_prevue":"Verdict"}),
    use_container_width=True, height=300, hide_index=True,
)

st.caption(
    f"**{model_name}** · {len(df_cities)} villes · "
    "Features : 13 raw + TempRange + Humidity_Avg + Pressure_Diff + "
    "Month_sin/cos + WindDir3pm_sin/cos + City_Encoded = **21 features**"
)
