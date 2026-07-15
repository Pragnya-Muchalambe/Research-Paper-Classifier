"""Streamlit UI for the Research Paper Acceptance Classifier."""

import streamlit as st

from research_paper_acceptance_classifier import (
    build_input_row,
    generate_dataset,
    predict_with_models,
    train_models,
)


st.set_page_config(page_title="Research Paper Acceptance Classifier", layout="wide")


@st.cache_resource
def get_training_bundle(sample_size: int, seed: int):
    df = generate_dataset(n_samples=sample_size, seed=seed)
    bundle = train_models(df, seed=seed)
    return df, bundle


st.title("Research Paper Acceptance Classifier")
st.caption("Offline demo using Naive Bayes, Logistic Regression, and Decision Tree")

with st.sidebar:
    st.header("Settings")
    sample_size = st.slider("Synthetic sample size", min_value=200, max_value=2000, value=700, step=100)
    random_seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    st.info(
        "The model is cached for speed. It is retrained automatically when settings change."
    )
    st.markdown("### About")
    st.write(
        "This app predicts whether a paper is likely to be accepted based on abstract text "
        "and simple paper metadata."
    )

df, bundle = get_training_bundle(sample_size=sample_size, seed=int(random_seed))

col_top_1, col_top_2, col_top_3 = st.columns(3)
with col_top_1:
    st.metric("Dataset Size", len(df))
with col_top_2:
    st.metric("Accepted", int(df["label"].sum()))
with col_top_3:
    st.metric("Rejected", int((df["label"] == 0).sum()))

st.subheader("Paper Input")
abstract = st.text_area(
    "Abstract",
    height=180,
    placeholder="Paste your abstract here...",
)

col1, col2, col3 = st.columns(3)
with col1:
    author_count = st.slider("Author count", 1, 15, 4)
with col2:
    title_length = st.slider("Title length (words)", 3, 30, 12)
with col3:
    citation_density = st.slider("Citation density", 0.00, 0.20, 0.08, 0.01)

show_features = st.checkbox("Show extracted features", value=False)
predict_clicked = st.button("Predict", use_container_width=True)

if predict_clicked:
    if not abstract.strip():
        st.warning("Please enter an abstract before prediction.")
    else:
        predictions_df, input_df = predict_with_models(
            abstract=abstract,
            author_count=author_count,
            title_length=title_length,
            citation_density=citation_density,
            models=bundle.models,
        )

        st.subheader("Predictions")
        for _, row in predictions_df.iterrows():
            msg = f"{row['Model']}: {row['Prediction']} ({row['Confidence (%)']:.2f}%)"
            if row["Prediction"] == "Accepted":
                st.success(msg)
            else:
                st.error(msg)

        st.dataframe(predictions_df, use_container_width=True)

        if show_features:
            st.subheader("Extracted Features")
            st.dataframe(input_df, use_container_width=True)

st.subheader("Model Performance")
metric_rows = []
for model_name, values in bundle.metrics.items():
    metric_rows.append(
        {
            "Model": model_name,
            "Accuracy": round(values["Accuracy"], 4),
            "Precision": round(values["Precision"], 4),
            "Recall": round(values["Recall"], 4),
            "F1 Score": round(values["F1 Score"], 4),
        }
    )

st.dataframe(metric_rows, use_container_width=True)

with st.expander("Classification Reports"):
    for model_name, values in bundle.metrics.items():
        st.markdown(f"**{model_name}**")
        st.code(values["Classification Report"])