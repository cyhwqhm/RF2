
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="RF Predictor with SHAP",
    page_icon="🌲",
    layout="centered",
)

st.title("Postoperative Residual PitNETs Predictor")
st.caption("Based on random forest model")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model("best_rf_model.pkl")
except Exception as e:
    st.error(f"无法加载模型: {e}")
    st.stop()

# ---------------------------
# Inputs
# ---------------------------
st.subheader("Please enter the model feature values")
FEATURES = [
    "Age2",
    "Maximum_coronal_length",
    "Maximum_sagittal_length",
    "Retrosellar",
    "ICA",
    "KNOSP",
    "Sphenoid_sinus_pneumatization",
    "Tumor_texture"
]

inputs = {}
cols = st.columns(2)
for i, name in enumerate(FEATURES):
    with cols[i % 2]:
        # 若某些为类目变量，可用数字编码（如 0/1/2）
        inputs[name] = st.number_input(name, value=0.0, format="%.2f")

# ---------------------------
# Predict + SHAP
# ---------------------------
def predict_probability(mdl, X_df):
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X_df)
        # 尝试取二分类的“第二列”为阳性概率；若不是二分类则取预测类的概率
        if proba.ndim == 2 and proba.shape[1] >= 2:
            pos_idx = 1
            p = float(proba[0, pos_idx])
        else:
            p = float(proba[0, np.argmax(proba[0])])
        return p
    else:
        # 回归模型：直接返回预测值（按概率显示）
        return float(mdl.predict(X_df)[0])

def render_shap_force(mdl, X_df):
    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X_df)
    # 分类任务（list）/ 回归（ndarray）兼容
    if isinstance(shap_values, list):
        sv = shap_values[1]  # 取第 1 类（正类）
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values
        base_value = explainer.expected_value

    force = shap.force_plot(
        base_value,
        sv[0, :],
        X_df.iloc[0, :],
        matplotlib=False
    )
    # 通过注入 JS + HTML 的方式嵌入（避免 shap.save_html(None, ...) 的写入错误）
    html = f"""
    <head>{shap.getjs()}</head>
    <body>{force.html()}</body>
    """
    components.html(html, height=320)

if st.button("Predict"):
    X = pd.DataFrame([inputs])
    try:
        p = predict_probability(model, X)
        st.markdown(f"**Based on feature values, predicted possibility is {p*100:.2f}%**")
        st.subheader("SHAP force plot")
        render_shap_force(model, X)
    except Exception as e:
        st.error(f"预测或生成 SHAP 力图时出错：{e}")
