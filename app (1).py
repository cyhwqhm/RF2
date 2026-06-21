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
# Fixed SHAP base value
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
# Helper functions
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


def extract_positive_class_shap_values(explainer, shap_values, pos_idx=1):
    """
    兼容不同 SHAP 版本的输出格式。
    """

    # 旧版 SHAP: list[class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[pos_idx]

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[pos_idx]
        else:
            base_value = explainer.expected_value

        return np.array(sv), float(base_value)

    # 新版 SHAP: shape = (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, pos_idx]

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[pos_idx]
        else:
            base_value = explainer.expected_value

        return np.array(sv), float(base_value)

    # 其他情况: shape = (n_samples, n_features)
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
    固定 force plot 显示的 base_value，
    并平移 SHAP value，保持加和关系不变。

    原始关系：
        original_base_value + sum(original_shap_values) = prediction

    调整后：
        target_base_value + sum(adjusted_shap_values) = prediction
    """

    sv = np.array(sv, dtype=float)
    n_features = sv.shape[1]

    shift_per_feature = (original_base_value - target_base_value) / n_features
    sv_adjusted = sv + shift_per_feature

    return sv_adjusted


def render_shap_force(mdl, X_df):
    """
    生成 SHAP force plot。
    仅显示力图，不显示下方特征值小字。
    """

    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X_df)

    pos_idx = get_positive_class_index(mdl)

    sv, original_base_value = extract_positive_class_shap_values(
        explainer=explainer,
        shap_values=shap_values,
        pos_idx=pos_idx
    )

    sv_adjusted = adjust_shap_to_fixed_base_value(
        sv=sv,
        original_base_value=original_base_value,
        target_base_value=TARGET_BASE_VALUE
    )

    # 关键修改：
    # 不传入 X_df.iloc[0, :]，即可去掉 force plot 下方的特征小字
    force = shap.force_plot(
        TARGET_BASE_VALUE,
        sv_adjusted[0, :],
        matplotlib=False
    )

    html = f"""
    <head>{shap.getjs()}</head>
    <body>{force.html()}</body>
    """

    components.html(html, height=260)


# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict"):
    X = pd.DataFrame([inputs])

    # 保证特征顺序与训练时一致
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
