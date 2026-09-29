# 7-Day Learning Engagement Decline Risk Prediction

English Streamlit interface for a research-use XGBoost model with 13 learner activity features. It displays predicted risk and local SHAP contributions. The bundled model and metadata load from `model/`; no credentials or external database are needed.

## Deploy on Streamlit Community Cloud

1. Upload the contents of this folder to the root of `xdh001/learning-engagement-risk` on the `main` branch. Keep the two JSON files inside `model/`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub and authorize repository access if requested.
3. Select repository `xdh001/learning-engagement-risk`, branch `main`, and main file path `app.py`; select **Deploy**.
4. Wait for dependencies and the model to load. Enter plausible values for all 13 fields and click **Predict risk**. Confirm the probability, threshold comparison, SHAP chart, and CSV download display.
5. Copy the actual URL shown by Streamlit after the app is live. Do not cite a proposed URL before testing it.

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The bundled threshold is the Development OOF Youden cut-point (`0.1676448434591293`); the interface describes it as a reference threshold. Input features must be derived consistently with the study's original feature definitions. This is a research demonstration, not an intervention or clinical decision system. Do not upload identifiable student records to a public app.
