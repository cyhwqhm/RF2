import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
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
# Fixed SHAP base value for display
# ---------------------------
TARGET_BASE_VALUE = 0.27083

# ---------------------------
# Inputs
# ---------------------------
st.subheader("Please enter the model feature values")

FEATURES = [
    "Age(0-99)",
    "Maximum_coronal_length(mm)",
    "Maximum_sagittal_length(mm)",
    "Retrosellar(No=0,Yes=1)",
    "ICA(None=0,Left=1,Right=2,Both=3)",
    "KNOSP(Grade=1,2,3,4)",
    "Sphenoid_sinus_pneumatization(Grade=1,2,3,4,5,6)",
    "Tumor_texture(Soft=0,Firm=1)"
]

inputs = {}
cols = st.columns(2)

for i, name in enumerate(FEATURES):
    with cols[i % 2]:
        inputs[name] = st.number_input(
            name,
            value=0.0,
            format="%.2f"
        )

# ---------------------------
# Predict probability
# ---------------------------
def get_positive_class_index(mdl):
    """
    获取二分类模型中阳性类别 1 对应的列索引。
    如果模型没有 classes_ 属性，则默认取第 2 列。
    """
    if hasattr(mdl, "classes_"):
        classes = list(mdl.classes_)
        if 1 in classes:
            return classes.index(1)
        else:
            return 1
    else:
        return 1


def predict_probability(mdl, X_df):
    """
    返回模型预测的阳性概率。
    """
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X_df)
        pos_idx = get_positive_class_index(mdl)

        if proba.ndim == 2 and proba.shape[1] > pos_idx:
            return float(proba[0, pos_idx])
        else:
            return float(proba[0, np.argmax(proba[0])])
    else:
        return float(mdl.predict(X_df)[0])

# ---------------------------
# SHAP helper
# ---------------------------
def extract_positive_class_shap_values(explainer, shap_values, pos_idx=1):
    """
    兼容不同 SHAP 版本的输出格式。

    常见情况：
    1. 旧版 SHAP:
       shap_values 是 list:
       shap_values[0] = 阴性类 SHAP
       shap_values[1] = 阳性类 SHAP

    2. 新版 SHAP:
       shap_values 是 ndarray:
       shape = (n_samples, n_features, n_classes)

    3. 回归或特殊情况:
       shap_values 是 ndarray:
       shape = (n_samples, n_features)
    """

    # 情况 1：旧版 SHAP，二分类 list 输出
    if isinstance(shap_values, list):
        sv = shap_values[pos_idx]

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[pos_idx]
        else:
            base_value = explainer.expected_value

        return np.array(sv), float(base_value)

    # 情况 2：新版 SHAP，三维数组输出
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, pos_idx]

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[pos_idx]
        else:
            base_value = explainer.expected_value

        return np.array(sv), float(base_value)

    # 情况 3：普通二维数组
    sv = np.array(shap_values)

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        if len(explainer.expected_value) > pos_idx:
            base_value = explainer.expected_value[pos_idx]
        else:
            base_value = explainer.expected_value[0]
    else:
        base_value = explainer.expected_value

    return sv, float(base_value)


def adjust_shap_to_fixed_base_value(sv, original_base_value, target_base_value):
    """
    将 SHAP force plot 的 base_value 固定为 target_base_value，
    并平移每个特征的 SHAP value，保证加和关系不变。

    原始关系：
        original_base_value + sum(original_shap_values) = prediction

    调整后：
        target_base_value + sum(adjusted_shap_values) = prediction
    """

    sv = np.array(sv, dtype=float)

    n_features = sv.shape[1]

    # 每个特征均匀分摊 base_value 的差值
    shift_per_feature = (original_base_value - target_base_value) / n_features

    sv_adjusted = sv + shift_per_feature

    return sv_adjusted, shift_per_feature


def render_shap_force(mdl, X_df):
    """
    生成 SHAP force plot，并将显示的 base_value 固定为 0.27083。
    """

    # 建议保留默认 TreeExplainer，适合当前只加载 pkl、不加载 background 的 Streamlit 部署方式
    explainer = shap.TreeExplainer(mdl)

    shap_values = explainer.shap_values(X_df)

    pos_idx = get_positive_class_index(mdl)

    sv, original_base_value = extract_positive_class_shap_values(
        explainer=explainer,
        shap_values=shap_values,
        pos_idx=pos_idx
    )

    # 固定 base_value，并平移 SHAP values
    sv_adjusted, shift_per_feature = adjust_shap_to_fixed_base_value(
        sv=sv,
        original_base_value=original_base_value,
        target_base_value=TARGET_BASE_VALUE
    )

    # 计算模型真实预测概率
    model_prediction = predict_probability(mdl, X_df)

    # SHAP 原始加和
    shap_original_prediction = original_base_value + np.sum(sv[0, :])

    # SHAP 调整后加和
    shap_adjusted_prediction = TARGET_BASE_VALUE + np.sum(sv_adjusted[0, :])

    st.caption(
        f"Original SHAP base value: {original_base_value:.5f} | "
        f"Displayed SHAP base value: {TARGET_BASE_VALUE:.5f}"
    )

    st.caption(
        f"Model predicted probability: {model_prediction:.5f} | "
        f"Adjusted SHAP reconstructed probability: {shap_adjusted_prediction:.5f}"
    )

    # force plot
    force = shap.force_plot(
        TARGET_BASE_VALUE,
        sv_adjusted[0, :],
        X_df.iloc[0, :],
        matplotlib=False
    )

    html = f"""
    <head>{shap.getjs()}</head>
    <body>{force.html()}</body>
    """

    components.html(html, height=340)


# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict"):
    X = pd.DataFrame([inputs])

    # 保证特征顺序和训练时一致
    X = X[FEATURES]

    try:
        p = predict_probability(model, X)

        st.markdown(
            f"**Based on feature values, predicted possibility is {p * 100:.2f}%**"
        )

        st.subheader("SHAP force plot")
        render_shap_force(model, X)

    except Exception as e:
        st.error(f"预测或生成 SHAP 力图时出错：{e}")
