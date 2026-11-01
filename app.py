"""
ChurnSense AI - Customer Churn Prediction web app (Streamlit).
Run locally:  streamlit run app.py

Team: Aditini Tiwari & Ankit Singh Negi  |  DIT University  |  Artificial Intelligence project
"""
import os
import subprocess
import sys

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ChurnSense AI | Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)


def html(s: str):
    """Render an HTML snippet (lines are stripped so Markdown never treats them as code blocks)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model + data loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="First start: training the model (takes about 1 minute)...")
def load_model():
    """Load the saved model; if it is missing or incompatible, train it from data/telco_churn.csv."""
    path = "models/churn_model.joblib"
    try:
        return joblib.load(path)
    except Exception:
        subprocess.run([sys.executable, "train.py"], check=True)
        return joblib.load(path)


@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churned"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_data
def load_metrics():
    try:
        return pd.read_csv("outputs/model_comparison.csv", index_col=0)
    except Exception:
        return None


model = load_model()
EXPECTED = list(model.named_steps["prep"].feature_names_in_)
df = load_data()
metrics = load_metrics()

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(900px 480px at 88% -6%, rgba(214,32,110,.30), transparent 60%),radial-gradient(800px 500px at 0% 8%, rgba(110,40,200,.18), transparent 60%),#07060b;color:#e9e6f2;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stToolbar"]{visibility:hidden;}
.block-container{max-width:1200px;padding-top:1rem;padding-bottom:3rem;}
.nav{display:flex;align-items:center;justify-content:space-between;padding:14px 4px 18px;border-bottom:1px solid #1d1a29;margin-bottom:34px;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:10px;font-size:27px;font-weight:700;color:#fff;letter-spacing:-.6px}
.logo{width:34px;height:34px;background:linear-gradient(135deg,#ff3d8b,#b3125a);clip-path:polygon(0 0,100% 0,62% 100%,36% 56%,0 40%)}
.links a{color:#d8d4e6 !important;text-decoration:none !important;margin:0 14px;font-weight:500;font-size:15px}
.links a:hover{color:#ff4fa3 !important}
.pill{display:inline-block;border:1.5px solid #fff;color:#fff;border-radius:999px;padding:9px 20px;font-weight:600;font-size:14px;margin-left:8px}
.pill.fill{background:#e0286f;border-color:#e0286f}
.eyebrow{color:#9a95ad;letter-spacing:.14em;font-size:13px;font-weight:600;text-transform:uppercase;margin:34px 0 18px}
.h1{font-size:66px;line-height:1.03;font-weight:700;letter-spacing:-2.2px;color:#fff;margin:0 0 24px}
.grad{background:linear-gradient(90deg,#c2185b,#ff4fa3 55%,#ff9bd0);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:#9a95ad;font-size:20px;line-height:1.55;max-width:560px;margin-bottom:26px}
.credit{display:inline-flex;gap:10px;align-items:center;background:#120f1b;border:1px solid #262134;border-radius:14px;padding:12px 16px;color:#cfcadd;font-size:14px}
.credit b{color:#fff}
.preview{border:1px solid #2a2538;border-radius:30px;padding:14px;background:linear-gradient(160deg,#4a0c2c,#1b0c30 60%,#0b0912)}
.pv-in{background:#0f0d17;border-radius:20px;padding:24px}
.pv-top{display:flex;justify-content:space-between;color:#9a95ad;font-size:13px;margin-bottom:16px}
.pv-row{display:flex;align-items:center;gap:22px}
.bar{height:8px;border-radius:99px;background:#241f33;margin:7px 0 14px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#c2185b,#ff6fb0)}
.lbl{color:#cfcadd;font-size:13px;display:flex;justify-content:space-between}
.ring{width:150px;height:150px;border-radius:50%;background:conic-gradient(var(--c) calc(var(--p)*1%), #231f33 0);display:grid;place-items:center;flex-shrink:0}
.ring-in{width:118px;height:118px;border-radius:50%;background:#0f0d17;display:grid;place-content:center;text-align:center}
.ring-in span{font-size:30px;font-weight:700;color:#fff;line-height:1}
.ring-in small{color:#9a95ad;font-size:12px;margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:44px}
.kpi{background:#0f0d17;border:1px solid #221e30;border-radius:18px;padding:20px 22px}
.kpi span{color:#9a95ad;font-size:13px;font-weight:500}
.kpi b{display:block;font-size:34px;color:#fff;margin-top:6px;letter-spacing:-1px}
.kpi small{color:#ff6fb0;font-size:12px}
.sec{font-size:40px;font-weight:700;color:#fff;letter-spacing:-1.2px;margin:70px 0 6px}
.sec-sub{color:#9a95ad;font-size:17px;margin-bottom:24px}
label p, .stSelectbox label p, .stSlider label p, .stNumberInput label p{color:#cfcadd !important;font-weight:500}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div{background:#13101c !important;border-color:#2a2538 !important;border-radius:12px !important}
.stButton>button{background:#e0286f;color:#fff;border:none;border-radius:999px;padding:.7rem 2.2rem;font-weight:600;font-size:16px}
.stButton>button:hover{background:#ff3d8b;color:#fff}
.res{display:flex;gap:30px;align-items:center;background:#0f0d17;border:1px solid #221e30;border-radius:22px;padding:28px;margin-top:18px;flex-wrap:wrap}
.tag{display:inline-block;padding:5px 14px;border-radius:999px;font-size:13px;font-weight:600;margin-bottom:10px}
.res h3{color:#fff;margin:4px 0 8px;font-size:26px}
.res p{color:#a9a4bb;margin:0 0 12px}
.chip{display:inline-block;padding:5px 12px;border-radius:999px;font-size:13px;font-weight:500;margin:0 8px 8px 0}
.chip.bad{background:rgba(255,77,109,.14);color:#ff7a93}
.chip.good{background:rgba(52,211,153,.14);color:#4ade9f}
.tbl{width:100%;border-collapse:collapse;background:#0f0d17;border:1px solid #221e30;border-radius:16px;overflow:hidden}
.tbl th{color:#9a95ad;font-weight:500;font-size:13px;text-align:left;padding:14px 16px;border-bottom:1px solid #221e30}
.tbl td{color:#e9e6f2;padding:14px 16px;border-bottom:1px solid #1a1727;font-size:15px}
.tbl tr.best td{background:rgba(224,40,111,.10);color:#fff;font-weight:600}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.step{background:#0f0d17;border:1px solid #221e30;border-radius:18px;padding:24px}
.step i{font-style:normal;display:inline-grid;place-items:center;width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#ff3d8b,#b3125a);color:#fff;font-weight:700;margin-bottom:14px}
.step h4{color:#fff;margin:0 0 8px;font-size:18px}
.step p{color:#9a95ad;margin:0;font-size:14.5px;line-height:1.55}
.team{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;max-width:760px}
.member{background:linear-gradient(160deg,#1c0d27,#0f0d17);border:1px solid #2a2538;border-radius:22px;padding:28px;display:flex;gap:18px;align-items:center}
.av{width:68px;height:68px;border-radius:50%;background:linear-gradient(135deg,#ff3d8b,#7c2bd9);display:grid;place-items:center;color:#fff;font-size:24px;font-weight:700;flex-shrink:0}
.member h4{color:#fff;margin:0 0 4px;font-size:20px}
.member p{color:#9a95ad;margin:0;font-size:14px}
.college{margin-top:22px;color:#cfcadd;font-size:16px}
.college b{color:#ff6fb0}
.foot{margin-top:70px;padding-top:22px;border-top:1px solid #1d1a29;color:#756f88;font-size:14px;text-align:center}
.foot b{color:#cfcadd}
@media (max-width:900px){.h1{font-size:42px}.kpis,.steps{grid-template-columns:repeat(2,1fr)}.team{grid-template-columns:1fr}.links{display:none}}
</style>
""")

# ----------------------------------------------------------------------------
# Navbar + hero
# ----------------------------------------------------------------------------
html("""
<div class="nav">
  <div class="brand"><div class="logo"></div>churnsense</div>
  <div class="links">
    <a href="#predict" target="_self">Predict</a>
    <a href="#insights" target="_self">Insights</a>
    <a href="#how" target="_self">How it works</a>
    <a href="#team" target="_self">Team</a>
  </div>
  <div><span class="pill">DIT University</span><span class="pill fill">AI Project</span></div>
</div>
""")

left, right = st.columns([1.1, 1], gap="large")
with left:
    html("""
    <div class="eyebrow">— CHURNSENSE AI</div>
    <div class="h1"><span class="grad">AI churn predictions,</span> before it's too late.</div>
    <div class="sub">Turn customer data into churn risk scores your team can act on. Enter a customer's details and get an instant probability, the factors behind it and what to do next.</div>
    <div class="credit">Built by <b>Aditini Tiwari</b> &amp; <b>Ankit Singh Negi</b> · DIT University</div>
    """)
with right:
    html("""
    <div class="preview"><div class="pv-in">
      <div class="pv-top"><span>Sample output · Customer C-48213</span><span>Live model</span></div>
      <div class="pv-row">
        <div class="ring" style="--p:82;--c:#ff4fa3"><div class="ring-in"><span>82%</span><small>churn risk</small></div></div>
        <div style="flex:1">
          <div class="lbl"><span>Month-to-month contract</span><span>High</span></div><div class="bar"><i style="width:90%"></i></div>
          <div class="lbl"><span>Short tenure (2 months)</span><span>High</span></div><div class="bar"><i style="width:80%"></i></div>
          <div class="lbl"><span>Electronic check payment</span><span>Medium</span></div><div class="bar"><i style="width:55%"></i></div>
        </div>
      </div>
    </div></div>
    """)

# ----------------------------------------------------------------------------
# KPI strip (computed from the real dataset + trained model results)
# ----------------------------------------------------------------------------
best_name, best_auc, best_acc = "-", "-", "-"
if metrics is not None and len(metrics):
    best_name = metrics["roc_auc"].idxmax()
    best_auc = f"{metrics.loc[best_name, 'roc_auc']:.3f}"
    best_acc = f"{metrics.loc[best_name, 'accuracy']:.1%}"

html(f"""
<div class="kpis">
  <div class="kpi"><span>Customers analysed</span><b>{len(df):,}</b><small>training + test data</small></div>
  <div class="kpi"><span>Overall churn rate</span><b>{df['Churned'].mean():.1%}</b><small>customers who left</small></div>
  <div class="kpi"><span>Best model</span><b style="font-size:24px;padding-top:9px">{best_name}</b><small>accuracy {best_acc}</small></div>
  <div class="kpi"><span>ROC-AUC score</span><b>{best_auc}</b><small>higher is better (max 1.0)</small></div>
</div>
""")

# ----------------------------------------------------------------------------
# Predict section
# ----------------------------------------------------------------------------
html('<div id="predict"></div><div class="sec">Predict churn risk</div>'
     '<div class="sec-sub">Fill in the customer details below. A retention or support team member would enter this information.</div>')

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
with c2:
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multi = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
with c3:
    stv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 10.0, 150.0, 70.0)
total = st.number_input("Total charges", 0.0, 12000.0, float(min(monthly * max(tenure, 1), 12000.0)))

if st.button("Predict churn risk"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": online_sec, "TechSupport": tech, "StreamingTV": stv, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total}])
    for col in EXPECTED:                      # columns that only exist in the real Kaggle data
        if col not in row.columns:
            row[col] = "No"
    p = float(model.predict_proba(row[EXPECTED])[0, 1])
    pct = round(p * 100)

    if p >= 0.5:
        level, color, tag_bg, msg = "High risk", "#ff4d6d", "rgba(255,77,109,.16)", "This customer is likely to leave. Act now."
    elif p >= 0.3:
        level, color, tag_bg, msg = "Medium risk", "#ffb020", "rgba(255,176,32,.16)", "This customer shows warning signs. Keep an eye on them."
    else:
        level, color, tag_bg, msg = "Low risk", "#34d399", "rgba(52,211,153,.16)", "This customer is likely to stay."

    bad, good = [], []
    if contract == "Month-to-month":
        bad.append("Month-to-month contract")
    else:
        good.append(f"{contract} contract")
    if tenure < 12:
        bad.append(f"Short tenure ({tenure} months)")
    elif tenure >= 36:
        good.append(f"Long tenure ({tenure} months)")
    if internet == "Fiber optic":
        bad.append("Fiber optic plan")
    if payment == "Electronic check":
        bad.append("Pays by electronic check")
    elif "automatic" in payment:
        good.append("Automatic payments")
    if online_sec == "No" and internet != "No":
        bad.append("No online security")
    if tech == "No" and internet != "No":
        bad.append("No tech support")
    if tech == "Yes" or online_sec == "Yes":
        good.append("Uses support / security add-ons")
    if monthly > 80:
        bad.append(f"High monthly bill (${monthly:.0f})")

    chips = "".join(f'<span class="chip bad">{b}</span>' for b in bad) or '<span class="chip good">No major risk factors</span>'
    chips_good = "".join(f'<span class="chip good">{g}</span>' for g in good)
    action = ("Offer a discount or contract upgrade and call the customer this week."
              if p >= 0.5 else
              "Send a satisfaction survey or a loyalty offer." if p >= 0.3 else
              "No action needed. Keep the service quality high.")

    html(f"""
    <div class="res">
      <div class="ring" style="--p:{pct};--c:{color}"><div class="ring-in"><span>{pct}%</span><small>churn risk</small></div></div>
      <div style="flex:1;min-width:280px">
        <span class="tag" style="background:{tag_bg};color:{color}">{level}</span>
        <h3>{msg}</h3>
        <p><b style="color:#fff">Suggested action:</b> {action}</p>
        <div>{chips}{chips_good}</div>
      </div>
    </div>
    """)

# ----------------------------------------------------------------------------
# Insights section
# ----------------------------------------------------------------------------
html('<div id="insights"></div><div class="sec">What drives churn?</div>'
     '<div class="sec-sub">Patterns found in 7,043 real telecom customers, and how the three models performed.</div>')


def churn_chart(frame, col, title, sort_x=None):
    g = frame.groupby(col, observed=True)["Churned"].mean().reset_index()
    g[col] = g[col].astype(str)
    chart = (
        alt.Chart(g)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, color="#ff4fa3")
        .encode(
            x=alt.X(f"{col}:N", sort=sort_x or "-y", title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Churned:Q", title="Churn rate", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("Churned:Q", format=".1%", title="Churn rate")],
        )
        .properties(height=280, title=title)
        .configure(background="transparent")
        .configure_axis(labelColor="#b9b4c9", titleColor="#b9b4c9", gridColor="#241f33", domainColor="#241f33")
        .configure_title(color="#ffffff", anchor="start", fontSize=16, fontWeight=600)
        .configure_view(strokeWidth=0)
    )
    return chart


d2 = df.copy()
d2["Tenure group"] = pd.cut(d2["tenure"], [-1, 12, 24, 48, 72], labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
a, b = st.columns(2, gap="large")
with a:
    st.altair_chart(churn_chart(d2, "Contract", "Churn rate by contract type"))
with b:
    st.altair_chart(churn_chart(d2, "Tenure group", "Churn rate by tenure",
                                sort_x=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]))
a, b = st.columns(2, gap="large")
with a:
    st.altair_chart(churn_chart(d2, "InternetService", "Churn rate by internet service"))
with b:
    st.altair_chart(churn_chart(d2, "PaymentMethod", "Churn rate by payment method"))

if metrics is not None and len(metrics):
    rows = ""
    for name, r in metrics.iterrows():
        cls = ' class="best"' if name == best_name else ""
        rows += (f"<tr{cls}><td>{name}</td><td>{r['accuracy']:.1%}</td><td>{r['precision']:.2f}</td>"
                 f"<td>{r['recall']:.2f}</td><td>{r['f1']:.2f}</td><td>{r['roc_auc']:.3f}</td></tr>")
    html(f"""
    <div style="color:#fff;font-size:20px;font-weight:600;margin:34px 0 14px">Model comparison</div>
    <table class="tbl"><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>{rows}</table>
    """)

if os.path.exists("outputs/feature_importance.png"):
    html('<div style="color:#fff;font-size:20px;font-weight:600;margin:34px 0 14px">Top churn drivers (from the best model)</div>')
    st.image("outputs/feature_importance.png")

# ----------------------------------------------------------------------------
# How it works
# ----------------------------------------------------------------------------
html("""
<div id="how"></div><div class="sec">How it works</div>
<div class="sec-sub">From raw customer records to a risk score in four steps.</div>
<div class="steps">
  <div class="step"><i>1</i><h4>Collect data</h4><p>7,043 telecom customers with 21 columns: tenure, contract, services, billing and whether they left.</p></div>
  <div class="step"><i>2</i><h4>Clean &amp; prepare</h4><p>Fix missing values, scale numbers and one-hot encode text columns so models can read them.</p></div>
  <div class="step"><i>3</i><h4>Train models</h4><p>Logistic Regression, Random Forest and Gradient Boosting learn from 80% of the data and are tested on 20%.</p></div>
  <div class="step"><i>4</i><h4>Predict &amp; act</h4><p>The best model scores each customer. High-risk customers get retention offers before they leave.</p></div>
</div>
""")

# ----------------------------------------------------------------------------
# Team + footer
# ----------------------------------------------------------------------------
html("""
<div id="team"></div><div class="sec">Meet the team</div>
<div class="sec-sub">Artificial Intelligence project: Customer Churn Prediction.</div>
<div class="team">
  <div class="member"><div class="av">AT</div><div><h4>Aditini Tiwari</h4><p>Project team member<br>DIT University</p></div></div>
  <div class="member"><div class="av">AN</div><div><h4>Ankit Singh Negi</h4><p>Project team member<br>DIT University</p></div></div>
</div>
<div class="college">College: <b>DIT University</b></div>
<div class="foot">© 2026 <b>ChurnSense AI</b> · Built by Aditini Tiwari &amp; Ankit Singh Negi · DIT University · Dataset: Telco Customer Churn (Kaggle)</div>
""")
