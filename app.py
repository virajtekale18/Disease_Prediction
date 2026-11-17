"""
app.py - Stage 9: simple web interface (Streamlit)

Start it from the project folder:
    streamlit run app.py
Your browser opens at http://localhost:8501.

The app only loads the saved model (via prediction.py) - it never retrains.
EDUCATIONAL PROTOTYPE ONLY - NOT A MEDICAL DIAGNOSIS.
"""
import pandas as pd
import streamlit as st

from prediction import MIN_RECOMMENDED_SYMPTOMS, load_bundle, predict, readable

st.set_page_config(page_title="Disease Prediction (educational)", page_icon="🩺")
st.title("Symptom-based Disease Prediction")

# ---- Load model -----------------------------------------------------------
try:
    bundle = load_bundle()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

features = bundle["feature_columns"]
label_to_name = {readable(f): f for f in features}     # 'Skin rash' -> 'skin_rash'

# ---- Input ------------------------------------------------------------------
selected = st.multiselect(
    "Select the symptoms you observe (type to search):",
    options=sorted(label_to_name),
    placeholder="e.g. Itching, Skin rash, Chills",
)
top_k = st.slider("How many possible diseases to show", 3, 10, 5)

if len(selected) and len(selected) < MIN_RECOMMENDED_SYMPTOMS:
    st.info(f"Tip: results are more reliable with at least {MIN_RECOMMENDED_SYMPTOMS} symptoms.")

# ---- Predict ----------------------------------------------------------------
if st.button("Predict", type="primary", disabled=not selected):
    try:
        result = predict([label_to_name[s] for s in selected], top_k=top_k)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader(f"Most likely: {result['predicted_disease']}")
    for w in result["warnings"]:
        st.info(w)

    if result["score_type"] == "probability":
        col, caption = "Probability (%)", "Model-estimated probability for each disease."
    else:
        col = "Similarity (%)"
        caption = (f"The {result['model_name']} model predicts by finding the most similar known "
                   "case, so it has no meaningful probabilities. Instead, each score shows how "
                   "similar your symptoms are to the closest known case of that disease "
                   "(100% = identical pattern). These are NOT probabilities.")

    table = pd.DataFrame(
        [(d, round(s * 100, 1)) for d, s in result["ranking"]], columns=["Disease", col])
    st.bar_chart(table.set_index("Disease"), horizontal=True)
    st.dataframe(table, hide_index=True)
    st.caption(caption)

with st.expander("About this model"):
    cv = bundle.get("cv_scores", {})
    st.write(f"**Model:** {bundle.get('model_name')} {bundle.get('best_params', '')}")
    st.write(f"**Trained on:** {bundle.get('trained_on')} covering "
             f"{len(bundle['label_encoder'].classes_)} diseases and {len(features)} symptoms.")
    if cv:
        st.write(f"**Cross-validated macro F1:** {cv.get('clean_f1_macro', 0):.3f} with all "
                 f"symptoms, {cv.get('hidden_f1_macro', 0):.3f} with 2 symptoms hidden.")
    st.write("The training data is synthetic and very regular, so these scores overstate "
             "how well it would work on real patients.")
