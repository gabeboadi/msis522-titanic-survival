
import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

ART = "artifacts"
FIG_DIR = os.path.join(ART, "figures")
MODEL_DIR = os.path.join(ART, "models")

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    results_df = pd.read_csv(os.path.join(MODEL_DIR, "model_results.csv"))

    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.joblib")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib")),
        "MLP Neural Network": tf.keras.models.load_model(os.path.join(MODEL_DIR, "mlp_model.keras"))
    }

    with open(os.path.join(MODEL_DIR, "best_params.json"), "r") as f:
        best_params = json.load(f)

    with open(os.path.join(MODEL_DIR, "shap_info.json"), "r") as f:
        shap_info = json.load(f)

    shap_background = np.load(os.path.join(MODEL_DIR, "shap_background.npy"))

    return preprocessor, results_df, models, best_params, shap_info, shap_background

preprocessor, results_df, models, best_params, shap_info, shap_background = load_artifacts()

st.title("Titanic Survival Prediction: End-to-End Data Science Workflow")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

with tab1:
    st.header("Executive Summary")

    st.write(
        "This project uses the Titanic dataset to predict whether a passenger survived or did not survive. "
        "The target variable is survived, where 1 means the passenger survived and 0 means the passenger did not survive. "
        "The features used in this analysis include passenger class, sex, age, fare, family information, and port of embarkation."
    )

    st.write(
        "This problem matters because it is a simple but useful example of how machine learning can uncover patterns in real-world outcomes. "
        "In this case, the models help show how demographic and travel-related features relate to survival. "
        "The same workflow can be applied to more serious problems such as customer churn, fraud detection, and health risk prediction."
    )

    st.write(
        "I first explored the data visually to understand how survival changed across key features such as sex, passenger class, age, fare, and family size. "
        "I then trained and compared five classification models: Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Neural Network."
    )

    best_model_name = results_df.sort_values(by="f1", ascending=False).iloc[0]["model"]
    best_model_f1 = results_df.sort_values(by="f1", ascending=False).iloc[0]["f1"]
    best_model_auc = results_df.sort_values(by="roc_auc", ascending=False).iloc[0]["roc_auc"]

    st.write(
        f"After comparing all models using accuracy, precision, recall, F1 score, and ROC-AUC, the strongest model was {best_model_name}. "
        f"It achieved an F1 score of {best_model_f1:.3f} and a ROC-AUC of {best_model_auc:.3f} on the test set. "
        "I then used SHAP to explain the best tree-based model and built this app so a user can interact with the results directly."
    )

with tab2:
    st.header("Descriptive Analytics")

    plots_and_captions = [
        ("target_distribution.png",
         "This plot shows the number of passengers who survived and did not survive. "
         "It shows that the classes are not perfectly balanced, so later model evaluation should not rely only on accuracy."),
        ("survival_by_sex.png",
         "This plot shows that female passengers had a much higher survival rate than male passengers. "
         "This suggests that sex is one of the strongest predictors of survival in the dataset."),
        ("survival_by_pclass.png",
         "This plot shows that first-class passengers had the highest survival rate and third-class passengers had the lowest. "
         "This suggests that passenger class strongly influenced survival outcomes."),
        ("age_by_survival.png",
         "This plot compares the age distribution of survivors and non-survivors. "
         "The two groups overlap, but age still adds useful predictive information."),
        ("fare_by_survival.png",
         "This plot shows that passengers who survived often had higher ticket fares. "
         "Because fare is related to class and status, it likely captures an important survival pattern."),
        ("correlation_heatmap.png",
         "This heatmap summarizes the relationships among the numeric and encoded features. "
         "It supports the earlier findings that class, sex, and fare are strongly related to survival.")
    ]

    for plot_file, caption in plots_and_captions:
        st.image(os.path.join(FIG_DIR, plot_file), use_container_width=True)
        st.caption(caption)

with tab3:
    st.header("Model Performance")

    st.subheader("Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Model Comparison by F1 Score")
    st.image(os.path.join(FIG_DIR, "model_comparison_f1.png"), use_container_width=True)

    st.subheader("ROC Curves")
    roc_files = [
        "roc_logistic_regression.png",
        "roc_decision_tree.png",
        "roc_random_forest.png",
        "roc_xgboost.png",
        "roc_mlp.png"
    ]

    for roc_file in roc_files:
        path = os.path.join(FIG_DIR, roc_file)
        if os.path.exists(path):
            st.image(path, use_container_width=True)

    st.subheader("Best Hyperparameters")
    st.json(best_params)

    st.subheader("Neural Network Training History")
    st.image(os.path.join(FIG_DIR, "mlp_loss_curve.png"), use_container_width=True)
    st.image(os.path.join(FIG_DIR, "mlp_accuracy_curve.png"), use_container_width=True)

with tab4:
    st.header("Explainability & Interactive Prediction")

    st.subheader("SHAP Global Explainability")
    st.image(os.path.join(FIG_DIR, "shap_beeswarm.png"), use_container_width=True)
    st.caption("The SHAP beeswarm plot shows which features had the largest overall effect on predictions and whether they pushed the prediction up or down.")

    st.image(os.path.join(FIG_DIR, "shap_bar.png"), use_container_width=True)
    st.caption("The SHAP bar plot ranks features by average importance, which helps identify the strongest drivers of model behavior.")

    st.subheader("Interactive Prediction")

    selected_model = st.selectbox(
        "Choose a model for prediction",
        list(models.keys()),
        index=3
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.selectbox("Sex", ["male", "female"])
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])
        pclass = st.selectbox("Passenger Class", [1, 2, 3])

    with col2:
        age = st.slider("Age", 0, 80, 30)
        sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
        parch = st.slider("Parents/Children Aboard", 0, 6, 0)

    with col3:
        fare = st.slider("Fare", 0.0, 600.0, 32.0, 1.0)

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    input_df = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked,
        "family_size": family_size,
        "is_alone": is_alone
    }])

    X_input_processed = preprocessor.transform(input_df)
    X_input_dense = X_input_processed.toarray() if hasattr(X_input_processed, "toarray") else X_input_processed

    if selected_model == "MLP Neural Network":
        pred_prob = float(models[selected_model].predict(X_input_dense, verbose=0).ravel()[0])
    else:
        pred_prob = float(models[selected_model].predict_proba(X_input_processed)[:, 1][0])

    pred_class = 1 if pred_prob >= 0.5 else 0

    st.metric("Predicted Class", pred_class)
    st.metric("Predicted Survival Probability", f"{pred_prob:.2%}")

    st.subheader("SHAP Waterfall for Custom Input")

    shap_model_name = shap_info["best_tree_model_for_shap"]
    shap_model = models[shap_model_name]
    explainer = shap.TreeExplainer(shap_model)

    raw_shap_values = explainer(X_input_dense)

    if len(raw_shap_values.values.shape) == 3:
        one_shap = shap.Explanation(
            values=raw_shap_values.values[0, :, 1],
            base_values=raw_shap_values.base_values[0, 1] if len(np.array(raw_shap_values.base_values).shape) > 1 else raw_shap_values.base_values[0],
            data=raw_shap_values.data[0],
            feature_names=preprocessor.get_feature_names_out().tolist()
        )
    else:
        one_shap = shap.Explanation(
            values=raw_shap_values.values[0],
            base_values=raw_shap_values.base_values[0] if len(np.array(raw_shap_values.base_values).shape) > 0 else raw_shap_values.base_values,
            data=raw_shap_values.data[0],
            feature_names=preprocessor.get_feature_names_out().tolist()
        )

    fig = plt.figure()
    shap.plots.waterfall(one_shap, max_display=15, show=False)
    st.pyplot(fig, clear_figure=True)

    st.caption(f"This SHAP waterfall plot uses the best tree-based model: {shap_model_name}.")
