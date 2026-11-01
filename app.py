"""
ChurnSense AI - Customer Churn Prediction web app (Streamlit).
Run locally:  streamlit run app.py
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
# Styling (teal + blue on near-black, with one light section)
# ----------------------------------------------------------------------------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(900px 500px at 90% -8%, rgba(20,184,166,.26), transparent 60%),radial-gradient(800px 520px at -5% 12%, rgba(59,130,246,.16), transparent 60%),#05080b;color:#e6ecee;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stToolbar"]{visibility:hidden;}
.block-container{max-width:1200px;padding-top:0;padding-bottom:3rem;}
[data-testid="stImage"] img{border-radius:18px;}
.banner{display:flex;justify-content:center;align-items:center;gap:12px;background:linear-gradient(90deg,#0f766e,#0e7490 50%,#1d4ed8);color:#fff;padding:12px 18px;font-size:15px;font-weight:500;margin:0 -2rem;text-align:center;flex-wrap:wrap}
.banner .new{background:rgba(255,255,255,.22);border-radius:999px;padding:3px 12px;font-size:12px;font-weight:700;letter-spacing:.04em}
.nav{display:flex;align-items:center;justify-content:space-between;padding:20px 4px 20px;border-bottom:1px solid #162126;margin-bottom:26px;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:11px;font-size:28px;font-weight:600;color:#fff;letter-spacing:-.8px}
.logo{width:32px;height:32px;border-radius:50%;background:conic-gradient(#2dd4bf 0 72%,#2563eb 0);box-shadow:inset 0 0 0 8px #05080b}
.links a{color:#d3dde0 !important;text-decoration:none !important;margin:0 15px;font-weight:500;font-size:15px}
.links a:hover{color:#2dd4bf !important}
.pill{display:inline-block;border:1.5px solid #fff;color:#fff !important;text-decoration:none !important;border-radius:999px;padding:10px 22px;font-weight:600;font-size:14px;margin-left:10px}
.pill.fill{background:#14b8a6;border-color:#14b8a6;color:#03110f !important}
.eyebrow{color:#8a9a9e;letter-spacing:.14em;font-size:13px;font-weight:600;text-transform:uppercase;margin:34px 0 18px}
.eyebrow:before{content:"";display:inline-block;width:18px;height:2px;background:#2dd4bf;vertical-align:middle;margin-right:12px}
.h1{font-size:68px;line-height:1.03;font-weight:600;letter-spacing:-2.4px;color:#fff;margin:0 0 24px}
.grad{background:linear-gradient(90deg,#14b8a6,#22d3ee 55%,#60a5fa);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:#8a9a9e;font-size:20px;line-height:1.55;max-width:560px;margin-bottom:10px;font-weight:300}
.preview{border:1px solid #1f2d33;border-radius:32px;padding:14px;background:linear-gradient(160deg,#0f766e,#1e3a8a 58%,#070b10)}
.pv-in{background:#0a1014;border-radius:22px;padding:24px}
.pv-top{display:flex;justify-content:space-between;color:#8a9a9e;font-size:13px;margin-bottom:16px}
.pv-row{display:flex;align-items:center;gap:22px}
.bar{height:8px;border-radius:99px;background:#18242a;margin:7px 0 14px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#14b8a6,#60a5fa)}
.lbl{color:#cfdadd;font-size:13px;display:flex;justify-content:space-between}
.ring{width:150px;height:150px;border-radius:50%;background:conic-gradient(var(--c) calc(var(--p)*1%), #18242a 0);display:grid;place-items:center;flex-shrink:0}
.ring-in{width:118px;height:118px;border-radius:50%;background:#0a1014;display:grid;place-content:center;text-align:center}
.ring-in span{font-size:30px;font-weight:700;color:#fff;line-height:1}
.ring-in small{color:#8a9a9e;font-size:12px;margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:46px}
.kpi{background:#0a1014;border:1px solid #162126;border-radius:18px;padding:20px 22px}
.kpi span{color:#8a9a9e;font-size:13px;font-weight:500}
.kpi b{display:block;font-size:34px;color:#fff;margin-top:6px;letter-spacing:-1px;font-weight:600}
.kpi small{color:#2dd4bf;font-size:12px}
.feat{display:grid;grid-template-columns:1fr 1.7fr;gap:70px;margin-top:100px}
.h2{font-size:44px;line-height:1.12;font-weight:400;color:#fff;letter-spacing:-1.2px;margin:0 0 20px}
.feat p.lead{color:#8a9a9e;font-size:18px;line-height:1.65;font-weight:300}
.fi{display:flex;gap:34px;padding:34px 0;border-bottom:1px solid #162126}
.fi:first-child{padding-top:0}
.num{font-size:58px;font-weight:200;color:#0f766e;min-width:92px;line-height:1}
.fi h4{color:#fff;font-weight:400;font-size:28px;margin:0 0 10px;letter-spacing:-.5px}
.fi p{color:#8a9a9e;font-size:17px;line-height:1.6;margin:0;font-weight:300}
.quote{font-size:42px;font-weight:300;line-height:1.28;color:#fff;max-width:980px;margin:110px 0 18px;letter-spacing:-1px}
.qcap{font-weight:600;color:#2dd4bf;font-size:15px;letter-spacing:.08em;text-transform:uppercase}
.sec{font-size:42px;font-weight:500;color:#fff;letter-spacing:-1.3px;margin:100px 0 6px}
.sec-sub{color:#8a9a9e;font-size:17px;margin-bottom:24px;font-weight:300}
label p, .stSelectbox label p, .stSlider label p, .stNumberInput label p{color:#cbd7da !important;font-weight:500}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div{background:#0c1317 !important;border-color:#1f2d33 !important;border-radius:12px !important}
.stButton>button{background:#14b8a6;color:#03110f;border:none;border-radius:999px;padding:.75rem 2.4rem;font-weight:700;font-size:16px}
.stButton>button:hover{background:#2dd4bf;color:#03110f}
.res{display:flex;gap:30px;align-items:center;background:#0a1014;border:1px solid #162126;border-radius:24px;padding:28px;margin-top:18px;flex-wrap:wrap}
.tag{display:inline-block;padding:5px 14px;border-radius:999px;font-size:13px;font-weight:600;margin-bottom:10px}
.res h3{color:#fff;margin:4px 0 8px;font-size:26px;font-weight:500}
.res p{color:#9fb0b4;margin:0 0 12px}
.chip{display:inline-block;padding:5px 12px;border-radius:999px;font-size:13px;font-weight:500;margin:0 8px 8px 0}
.chip.bad{background:rgba(255,93,115,.14);color:#ff8497}
.chip.good{background:rgba(45,212,191,.14);color:#4fe3cf}
.tbl{width:100%;border-collapse:collapse;background:#0a1014;border:1px solid #162126;border-radius:16px;overflow:hidden}
.tbl th{color:#8a9a9e;font-weight:500;font-size:13px;text-align:left;padding:14px 16px;border-bottom:1px solid #162126}
.tbl td{color:#e6ecee;padding:14px 16px;border-bottom:1px solid #111b20;font-size:15px}
.tbl tr.best td{background:rgba(20,184,166,.12);color:#fff;font-weight:600}
.light{background:#f1f6f5;border-radius:34px;padding:60px 56px;margin-top:110px}
.light .eyebrow{color:#5e7370;margin-top:0}
.lgrid{display:grid;grid-template-columns:1fr 1.5fr;gap:56px;align-items:start}
.light .h2{color:#0a1412;font-weight:400}
.light .acc{color:#0d9488}
.light p.lead{color:#5b6b68;font-size:17px;line-height:1.7;font-weight:400}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.card{background:#fff;border:1px solid #dfeae8;border-radius:22px;padding:24px;box-shadow:0 8px 24px rgba(15,60,55,.06)}
.card i{font-style:normal;display:inline-grid;place-items:center;width:38px;height:38px;border-radius:50%;background:rgba(20,184,166,.12);border:1px solid #8fe0d5;color:#0f766e;font-weight:700;margin-bottom:16px;font-size:14px}
.card h4{color:#0a1412;margin:0 0 8px;font-size:21px;font-weight:500;letter-spacing:-.4px}
.card p{color:#5b6b68;margin:0;font-size:15px;line-height:1.6}
.flow{background:linear-gradient(135deg,#0d9488,#2563eb);border-radius:22px;padding:26px;display:flex;align-items:center;justify-content:space-around;gap:10px;flex-wrap:wrap;margin-top:16px}
.fb{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.32);color:#fff;border-radius:14px;padding:14px 20px;font-weight:600;font-size:14px;text-align:center}
.arr{color:#fff;font-size:24px}
.foot{margin-top:90px;padding-top:24px;border-top:1px solid #162126;color:#6b7c80;font-size:14px;text-align:center}
.foot b{color:#cbd7da}
@media (max-width:900px){.h1{font-size:42px}.h2{font-size:32px}.quote{font-size:28px}.kpis{grid-template-columns:repeat(2,1fr)}.feat,.lgrid{grid-template-columns:1fr;gap:30px}.cards{grid-template-columns:1fr}.links{display:none}.light{padding:34px 22px}.num{font-size:40px;min-width:64px}}
</style>
""")

# ----------------------------------------------------------------------------
# Announcement banner (the only place the project authors are named) + navbar
# ----------------------------------------------------------------------------
html("""
<div class="banner"><span class="new">NEW</span><span>AI customer churn prediction project by Aditini Tiwari &amp; Ankit Singh Negi</span></div>
<div class="nav">
  <div class="brand"><div class="logo"></div>churnsense</div>
  <div class="links">
    <a href="#why" target="_self">Product</a>
    <a href="#predict" target="_self">Predict</a>
    <a href="#insights" target="_self">Insights</a>
    <a href="#how" target="_self">How it works</a>
  </div>
  <div><a class="pill" href="#predict" target="_self">Try the demo</a><a class="pill fill" href="#insights" target="_self">See insights</a></div>
</div>
""")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large")
with left:
    html("""
    <div class="eyebrow">CHURNSENSE PREDICT</div>
    <div class="h1"><span class="grad">Spot churn early.</span> Act before customers leave.</div>
    <div class="sub">Turn everyday customer data into a clear risk score, the reasons behind it and the next best step, in seconds.</div>
    """)
with right:
    html("""
    <div style="height:26px"></div>
    <div class="preview"><div class="pv-in">
      <div class="pv-top"><span>Sample output · Customer C-48213</span><span>Live model</span></div>
      <div class="pv-row">
        <div class="ring" style="--p:82;--c:#22d3ee"><div class="ring-in"><span>82%</span><small>churn risk</small></div></div>
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
# Numbered feature list + big statement
# ----------------------------------------------------------------------------
html("""
<div id="why" class="feat">
  <div>
    <div class="eyebrow">WHY CHURNSENSE</div>
    <div class="h2">Your early-warning system for customer churn</div>
    <p class="lead">ChurnSense learns from past customers who stayed and who left, then flags the ones who look likely to leave next, so teams can step in while there is still time.</p>
  </div>
  <div>
    <div class="fi"><div class="num">01</div><div><h4>Catch risk early</h4><p>Every customer gets a churn probability from 0 to 100%, so the riskiest accounts rise to the top of the list.</p></div></div>
    <div class="fi"><div class="num">02</div><div><h4>Explain the 'why'</h4><p>Each score comes with plain-language risk factors such as contract type, tenure and billing, not just a number.</p></div></div>
    <div class="fi"><div class="num">03</div><div><h4>Suggest the next step</h4><p>High, medium and low risk each trigger a different recommended action, from a retention call to a loyalty offer.</p></div></div>
  </div>
</div>
<div class="quote">Keeping an existing customer is usually far cheaper than winning a new one.</div>
<div class="qcap">The business case for churn prediction</div>
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
        level, color, tag_bg, msg = "High risk", "#ff5d73", "rgba(255,93,115,.16)", "This customer is likely to leave. Act now."
    elif p >= 0.3:
        level, color, tag_bg, msg = "Medium risk", "#ffb020", "rgba(255,176,32,.16)", "This customer shows warning signs. Keep an eye on them."
    else:
        level, color, tag_bg, msg = "Low risk", "#2dd4bf", "rgba(45,212,191,.16)", "This customer is likely to stay."

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
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, color="#2dd4bf")
        .encode(
            x=alt.X(f"{col}:N", sort=sort_x or "-y", title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Churned:Q", title="Churn rate", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("Churned:Q", format=".1%", title="Churn rate")],
        )
        .properties(height=280, title=title)
        .configure(background="transparent")
        .configure_axis(labelColor="#a9b8bc", titleColor="#a9b8bc", gridColor="#17232a", domainColor="#17232a")
        .configure_title(color="#ffffff", anchor="start", fontSize=16, fontWeight=500)
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
    <div style="color:#fff;font-size:20px;font-weight:500;margin:34px 0 14px">Model comparison</div>
    <table class="tbl"><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>{rows}</table>
    """)

if os.path.exists("outputs/feature_importance.png"):
    html('<div style="color:#fff;font-size:20px;font-weight:500;margin:34px 0 14px">Top churn drivers (from the best model)</div>')
    st.image("outputs/feature_importance.png")

# ----------------------------------------------------------------------------
# How it works (light section)
# ----------------------------------------------------------------------------
html("""
<div id="how" class="light">
  <div class="lgrid">
    <div>
      <div class="eyebrow">HOW IT WORKS</div>
      <div class="h2">AI churn prediction that tells you <span class="acc">who to call first.</span></div>
      <p class="lead">From raw customer records to a risk score in four steps, built on a public telecom dataset of 7,043 customers.</p>
    </div>
    <div>
      <div class="cards">
        <div class="card"><i>01</i><h4>Collect data</h4><p>21 columns per customer: tenure, contract, services, billing and whether they left.</p></div>
        <div class="card"><i>02</i><h4>Clean &amp; prepare</h4><p>Fix missing values, scale numbers and one-hot encode text so models can read them.</p></div>
        <div class="card"><i>03</i><h4>Train models</h4><p>Logistic Regression, Random Forest and Gradient Boosting learn from 80% of the data and are tested on 20%.</p></div>
        <div class="card"><i>04</i><h4>Predict &amp; act</h4><p>The best model scores each customer so retention offers go to those most likely to leave.</p></div>
      </div>
      <div class="flow"><div class="fb">Customer data</div><div class="arr">→</div><div class="fb">Trained ML model</div><div class="arr">→</div><div class="fb">Risk score + reasons</div></div>
    </div>
  </div>
</div>
<div class="foot"><b>ChurnSense AI</b> · Customer churn prediction with machine learning · Dataset: Telco Customer Churn (Kaggle)</div>
""")
