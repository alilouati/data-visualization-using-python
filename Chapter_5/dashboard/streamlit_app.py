"""Al-Kharj Road Accidents Dashboard (2022).

The same app students build in the Chapter 5 Colab notebook, ready to run online.
Run locally:   streamlit run Chapter_5/dashboard/streamlit_app.py
Deploy:        share.streamlit.io -> Create app -> this repo, branch main,
               main file path Chapter_5/dashboard/streamlit_app.py
"""
from pathlib import Path
import numpy as np, pandas as pd, plotly.express as px, streamlit as st, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Al-Kharj Road Accidents 2022", layout="wide")
WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")          # the file starts with a byte-order mark
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

DATA = Path(__file__).resolve().parent.parent / "2022.csv"   # Chapter_5/2022.csv
df = load_csv(DATA)
num = df.select_dtypes("number").columns.tolist()
if len(num) < 2:
    st.error("The dataset must contain at least two numeric columns."); st.stop()

st.title("Al-Kharj Road Accidents Dashboard (2022)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Days", len(df))
k2.metric("Car accidents (within city)", int(df.get("CarAccidentsWithinCity", df[num[0]]).sum()))
k3.metric("Car accidents (outside city)", int(df.get("CarAccidentsOutsideCity", df[num[0]]).sum()))
dead = [c for c in num if c.startswith("Deceased")]
k4.metric("Deceased", int(df[dead].sum().sum()) if dead else "–")

tab1, tab2, tab3 = st.tabs(["📈 Explore", "🌳 Decision tree", "🔵 K-Means"])

with tab1:
    feature = st.selectbox("Numeric column", num, key="explore_col")
    window = st.slider("Rolling mean (days)", 1, 30, 7)
    ts = df[["Date", feature]].assign(rolling=df[feature].rolling(window, center=True).mean())
    st.plotly_chart(px.line(ts, x="Date", y=[feature, "rolling"], title=f"{feature} per day and {window}-day mean"), width="stretch")
    wd = df.assign(weekday=df.Date.dt.day_name())
    st.plotly_chart(px.box(wd, x="weekday", y=feature, category_orders={"weekday": WEEK}, title=f"{feature} by weekday"), width="stretch")
    st.dataframe(df.describe().T.round(2))

with tab2:
    default_t = num.index("CarAccidentsWithinCity") if "CarAccidentsWithinCity" in num else 0
    target_col = st.selectbox("Column that defines 'Severe' (above its median)", num, index=default_t, key="target_col")
    depth = st.slider("Max tree depth", 1, 8, 3)
    data = df.copy()
    data["Situation"] = np.where(data[target_col] > data[target_col].median(), "Severe", "Not Severe")
    counts = data["Situation"].value_counts()
    st.caption(f"Class balance: {counts.to_dict()}")
    if len(counts) < 2 or counts.min() < 10:
        st.error("This column is almost always the same value (e.g. zero), so the two classes are too unbalanced to train on. Pick another column.")
        st.stop()

    candidates = [c for c in num if c != target_col]
    leak = data[candidates].corrwith(data[target_col]).abs().sort_values(ascending=False)
    st.write("**Leakage check**: |correlation| of each predictor with the labelling column")
    st.dataframe(leak.rename("abs_corr").round(3).to_frame().T)
    flagged = leak[leak > 0.7].index.tolist()
    if flagged:
        st.warning(f"{flagged} are near-copies of '{target_col}'. Accuracy using them would be trivial; they are excluded by default.")
    X_cols = st.multiselect("Predictors", candidates, default=[c for c in candidates if c not in flagged])
    if st.checkbox("Add calendar features (weekday, month)", value=True):
        data["weekday_num"] = data.Date.dt.dayofweek; data["month"] = data.Date.dt.month
        X_cols = X_cols + ["weekday_num", "month"]
    if not X_cols:
        st.info("Choose at least one predictor."); st.stop()

    X_tr, X_te, y_tr, y_te = train_test_split(data[X_cols], data["Situation"], test_size=0.3, stratify=data["Situation"], random_state=42)
    model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_tr, y_tr)
    acc_tr, acc_te = accuracy_score(y_tr, model.predict(X_tr)), accuracy_score(y_te, model.predict(X_te))
    baseline = y_te.value_counts(normalize=True).max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train accuracy", f"{acc_tr:.3f}"); c2.metric("Test accuracy", f"{acc_te:.3f}")
    c3.metric("Baseline (majority class)", f"{baseline:.3f}"); c4.metric("Generalisation gap", f"{acc_tr - acc_te:+.3f}")
    if acc_te <= baseline + 0.02:
        st.info("The model barely beats guessing the majority class: these predictors carry little information about the label.")

    cm = confusion_matrix(y_te, model.predict(X_te), labels=model.classes_)
    l, r = st.columns(2)
    l.plotly_chart(px.imshow(cm, x=list(model.classes_), y=list(model.classes_), text_auto=True, color_continuous_scale="Blues",
                             labels=dict(x="predicted", y="actual"), title="Confusion matrix (test set)"), width="stretch")
    imp = pd.Series(model.feature_importances_, index=X_cols).sort_values()
    r.plotly_chart(px.bar(imp, orientation="h", title="Feature importance"), width="stretch")

    fig_tree, ax = plt.subplots(figsize=(min(4 + 3 * depth, 24), 2 + 1.5 * depth))
    plot_tree(model, feature_names=X_cols, class_names=list(model.classes_), filled=True, rounded=True, fontsize=8, ax=ax)
    st.pyplot(fig_tree)
    with st.expander("Text rules"):
        st.code(export_text(model, feature_names=X_cols))

with tab3:
    cols = st.multiselect("Features for clustering", num, default=num)
    if len(cols) < 2:
        st.info("Choose at least two features."); st.stop()
    Xs = StandardScaler().fit_transform(df[cols])
    ks = list(range(2, 9))
    sil = [silhouette_score(Xs, KMeans(k, n_init=10, random_state=42).fit_predict(Xs)) for k in ks]
    best = ks[int(np.argmax(sil))]
    st.plotly_chart(px.line(x=ks, y=sil, markers=True, labels=dict(x="k", y="silhouette"), title=f"Silhouette by k (best k = {best})"), width="stretch")
    k = st.slider("Number of clusters", 2, 8, best)
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(Xs)
    pcs = PCA(2).fit_transform(Xs)
    st.plotly_chart(px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=labels.astype(str), labels=dict(x="PC1", y="PC2", color="cluster"),
                               hover_name=df.Date.dt.strftime("%d %b"), title="Days clustered, shown in PCA space", opacity=0.7), width="stretch")
    st.write("Cluster profiles (mean per day):")
    st.dataframe(df[cols].assign(cluster=labels).groupby("cluster").mean().round(2).assign(days=np.bincount(labels)))
    st.write("Do clusters follow the calendar? (days per cluster × weekday)")
    st.dataframe(pd.crosstab(labels, df.Date.dt.day_name()).reindex(columns=WEEK, fill_value=0))