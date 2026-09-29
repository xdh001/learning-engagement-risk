# ============================================================
# 7-Day Learning Engagement Decline Risk Prediction
# Final publication-ready Streamlit application
# ============================================================

from pathlib import Path
import json
import io

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import altair as alt


# ============================================================
# 0. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="7-Day Learning Engagement Decline Risk Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 1. PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

MODEL_PATH = MODEL_DIR / "xgb_final.json"
META_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# 2. EXACT FINAL MODEL FEATURES
# ============================================================

FEATURES = [
    "active_days_7d",
    "active_days_14d",
    "active_days_total_to_date",
    "days_since_last_activity",
    "days_since_first_activity",
    "max_inactivity_gap_28d",
    "mean_activity_gap_28d",
    "sd_activity_gap_28d",
    "active_day_density_28d",
    "active_weeks_in_28d",
    "wp_active_day_ratio_7v21",
    "wp_active_day_ratio_14v14",
    "wp_active_day_diff_7v21",
]


# ============================================================
# 3. ENGLISH LABELS
# ============================================================

LABELS = {
    "active_days_7d":
        "Active learning days in the previous 7 days",

    "active_days_14d":
        "Active learning days in the previous 14 days",

    "active_days_total_to_date":
        "Cumulative active learning days up to the prediction time",

    "days_since_last_activity":
        "Days since the most recent learning activity",

    "days_since_first_activity":
        "Days from the first learning activity to the prediction time",

    "max_inactivity_gap_28d":
        "Maximum learning-activity gap in the previous 28 days",

    "mean_activity_gap_28d":
        "Mean learning-activity gap in the previous 28 days",

    "sd_activity_gap_28d":
        "SD of learning-activity gaps in the previous 28 days",

    "active_day_density_28d":
        "Active-day density in the previous 28 days",

    "active_weeks_in_28d":
        "Number of active weeks in the previous 28 days",

    "wp_active_day_ratio_7v21":
        "Ratio of recent 7-day activity to prior 21-day weekly activity",

    "wp_active_day_ratio_14v14":
        "Ratio of recent 14-day activity to prior 14-day activity",

    "wp_active_day_diff_7v21":
        "Difference between recent 7-day activity and prior 21-day weekly activity",
}


# ============================================================
# 4. SHORT LABELS FOR SHAP FIGURE
# ============================================================

SHAP_LABELS = {
    "active_days_7d":
        "Active days in previous 7 days",

    "active_days_14d":
        "Active days in previous 14 days",

    "active_days_total_to_date":
        "Cumulative active days",

    "days_since_last_activity":
        "Days since last activity",

    "days_since_first_activity":
        "Days since first activity",

    "max_inactivity_gap_28d":
        "Maximum inactivity gap (28 d)",

    "mean_activity_gap_28d":
        "Mean activity gap (28 d)",

    "sd_activity_gap_28d":
        "SD of activity gaps (28 d)",

    "active_day_density_28d":
        "Active-day density (28 d)",

    "active_weeks_in_28d":
        "Active weeks (28 d)",

    "wp_active_day_ratio_7v21":
        "7 d / prior 21 d activity ratio",

    "wp_active_day_ratio_14v14":
        "14 d / prior 14 d activity ratio",

    "wp_active_day_diff_7v21":
        "7 d − prior 21 d activity difference",
}


# ============================================================
# 5. HELP TEXT
# ============================================================

HELP_TEXT = {
    "active_days_7d":
        "Number of calendar days with at least one valid learning-platform "
        "activity during the 7 days before the prediction time.",

    "active_days_14d":
        "Number of calendar days with at least one valid learning-platform "
        "activity during the 14 days before the prediction time.",

    "active_days_total_to_date":
        "Total number of active learning days from the first recorded "
        "learning activity to the current prediction time.",

    "days_since_last_activity":
        "Number of days between the most recent valid learning activity "
        "and the current prediction time.",

    "days_since_first_activity":
        "Number of days between the first recorded learning activity "
        "and the current prediction time.",

    "max_inactivity_gap_28d":
        "Maximum interval between consecutive learning activities "
        "during the previous 28 days.",

    "mean_activity_gap_28d":
        "Mean interval between consecutive learning activities "
        "during the previous 28 days.",

    "sd_activity_gap_28d":
        "Standard deviation of the intervals between consecutive "
        "learning activities during the previous 28 days.",

    "active_day_density_28d":
        "Proportion of active learning days during the previous 28 days. "
        "The value ranges from 0 to 1.",

    "active_weeks_in_28d":
        "Number of weeks with at least one valid learning activity "
        "during the previous 28 days. The value ranges from 0 to 4.",

    "wp_active_day_ratio_7v21":
        "Ratio of active learning days in the most recent 7 days "
        "to the average weekly number of active learning days "
        "during the preceding 21 days.",

    "wp_active_day_ratio_14v14":
        "Ratio of active learning days in the most recent 14 days "
        "to those during the preceding 14 days.",

    "wp_active_day_diff_7v21":
        "Difference between active learning days in the most recent "
        "7 days and the average weekly number of active learning days "
        "during the preceding 21 days.",
}


# ============================================================
# 6. FALLBACK DEFAULT VALUES
# ============================================================

DEFAULT_VALUES = {
    "active_days_7d": 3,
    "active_days_14d": 5,
    "active_days_total_to_date": 39,
    "days_since_last_activity": 1,
    "days_since_first_activity": 104,
    "max_inactivity_gap_28d": 6,
    "mean_activity_gap_28d": 2.25,
    "sd_activity_gap_28d": 1.58,
    "active_day_density_28d": 0.393,
    "active_weeks_in_28d": 4,
    "wp_active_day_ratio_7v21": 0.923,
    "wp_active_day_ratio_14v14": 1.000,
    "wp_active_day_diff_7v21": -0.333,
}


# ============================================================
# 7. LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return {}

    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


booster = load_model()
metadata = load_metadata()


# ============================================================
# 8. DEVELOPMENT-COHORT REFERENCE CUT-POINT
# ============================================================

def get_threshold(meta):

    possible_keys = [
        "threshold",
        "youden_threshold",
        "development_threshold",
        "reference_threshold",
        "cutoff",
        "cut_point",
    ]

    for key in possible_keys:

        if key in meta:

            try:
                return float(meta[key])

            except Exception:
                pass

    return 0.167645


THRESHOLD = get_threshold(metadata)


# ============================================================
# 9. DEFAULT / MEDIAN VALUES
# ============================================================

def get_default_value(feature):

    candidate_dict_names = [
        "medians",
        "median_values",
        "feature_medians",
        "defaults",
        "default_values",
        "imputation_values",
    ]

    for dict_name in candidate_dict_names:

        obj = metadata.get(dict_name)

        if isinstance(obj, dict) and feature in obj:

            try:
                return float(obj[feature])

            except Exception:
                pass

    return float(DEFAULT_VALUES[feature])


# ============================================================
# 10. PUBLICATION-STYLE CSS
# ============================================================

st.markdown(
    """
<style>

/* ----------------------------------------------------------
   Main page
---------------------------------------------------------- */

.block-container {
    max-width: 1180px;
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}


/* ----------------------------------------------------------
   Hide Streamlit development controls
---------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
    height: 0px;
}

[data-testid="stToolbar"] {
    display: none;
}

.stDeployButton {
    display: none;
}


/* ----------------------------------------------------------
   Header
---------------------------------------------------------- */

.publication-title {
    font-size: 2.55rem;
    font-weight: 760;
    line-height: 1.20;
    color: #172033;
    letter-spacing: -0.02em;
    margin-top: 0.2rem;
    margin-bottom: 0.55rem;
}

.publication-subtitle {
    font-size: 1.05rem;
    color: #667085;
    line-height: 1.55;
    margin-bottom: 1.9rem;
}


/* ----------------------------------------------------------
   Section headings
---------------------------------------------------------- */

.section-title {
    font-size: 1.55rem;
    font-weight: 720;
    color: #172033;
    margin-top: 1.2rem;
    margin-bottom: 0.9rem;
}

.group-title {
    font-size: 1.10rem;
    font-weight: 700;
    color: #344054;
    margin-top: 0.45rem;
    margin-bottom: 0.45rem;
}


/* ----------------------------------------------------------
   Cards
---------------------------------------------------------- */

.result-card {
    background: linear-gradient(
        145deg,
        #F8FAFC 0%,
        #F2F6FC 100%
    );
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.55rem 1.65rem;
    margin-bottom: 1rem;
}

.result-label {
    color: #667085;
    font-size: 0.95rem;
    line-height: 1.45;
    margin-bottom: 0.45rem;
}

.result-number {
    color: #1D4ED8;
    font-size: 3.35rem;
    font-weight: 760;
    line-height: 1.05;
    margin-top: 0.15rem;
    margin-bottom: 0.25rem;
}


/* ----------------------------------------------------------
   Reference cut-point card
---------------------------------------------------------- */

.status-card {
    border: 1px solid #D0D5DD;
    border-radius: 13px;
    background: #FFFFFF;
    padding: 1.10rem 1.20rem;
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
}

.status-title {
    color: #344054;
    font-size: 1.06rem;
    font-weight: 700;
    margin-bottom: 0.30rem;
}

.status-detail {
    color: #667085;
    font-size: 0.95rem;
}


/* ----------------------------------------------------------
   Research-use note
---------------------------------------------------------- */

.note-box {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 11px;
    background: #F8FAFC;
    border-left: 4px solid #94A3B8;
    color: #64748B;
    font-size: 0.92rem;
    line-height: 1.65;
}


/* ----------------------------------------------------------
   Model description
---------------------------------------------------------- */

.model-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.05rem 1.20rem;
    color: #475467;
    font-size: 0.94rem;
    line-height: 1.75;
}


/* ----------------------------------------------------------
   Inputs
---------------------------------------------------------- */

div[data-testid="stNumberInput"] input {
    background-color: #F8FAFC;
}


/* ----------------------------------------------------------
   Form submit button
---------------------------------------------------------- */

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    height: 3.25rem;
    border-radius: 10px;
    border: none;
    background: #1D4ED8;
    color: white;
    font-size: 1.05rem;
    font-weight: 700;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background: #1E40AF;
    color: white;
    border: none;
}


/* ----------------------------------------------------------
   Download button
---------------------------------------------------------- */

div[data-testid="stDownloadButton"] > button {
    border-radius: 9px;
    min-height: 2.8rem;
}


/* ----------------------------------------------------------
   Expander
---------------------------------------------------------- */

div[data-testid="stExpander"] {
    border-radius: 10px;
}


/* ----------------------------------------------------------
   Remove excessive horizontal padding on mobile
---------------------------------------------------------- */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .publication-title {
        font-size: 2rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 11. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="publication-title">'
    '7-Day Learning Engagement Decline Risk Prediction'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="publication-subtitle">'
    'An XGBoost-based research tool for estimating the probability '
    'of a decline in learning engagement during the subsequent 7 days.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 12. MODEL DESCRIPTION
# ============================================================

with st.expander(
    "About the prediction model",
    expanded=False,
):

    st.markdown(
        '<div class="model-box">'
        'This prediction tool was developed using an XGBoost model '
        'based on 13 cross-platform learning-behavior features. '
        'It estimates the probability of a decline in learning engagement '
        'during the 7 days following the prediction time.'
        '<br><br>'
        f'<b>Development-cohort reference cut-point:</b> '
        f'{THRESHOLD:.3f} ({THRESHOLD * 100:.1f}%).'
        '<br><br>'
        'The reference cut-point was derived from the development cohort '
        'using the Youden index. It is provided for research-oriented '
        'risk stratification and should not be interpreted as an absolute '
        'threshold for educational intervention.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 13. INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Learning Behavior Information Before the Prediction Time'
    '</div>',
    unsafe_allow_html=True,
)


with st.form("prediction_form"):

    # --------------------------------------------------------
    # GROUP 1
    # --------------------------------------------------------

    st.markdown(
        '<div class="group-title">'
        '1. Recent Learning Activity'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        2,
        gap="large",
    )


    with col1:

        active_days_7d = st.number_input(
            LABELS["active_days_7d"],
            min_value=0,
            max_value=7,
            value=int(
                round(
                    get_default_value(
                        "active_days_7d"
                    )
                )
            ),
            step=1,
            help=HELP_TEXT["active_days_7d"],
        )


        active_days_14d = st.number_input(
            LABELS["active_days_14d"],
            min_value=0,
            max_value=14,
            value=int(
                round(
                    get_default_value(
                        "active_days_14d"
                    )
                )
            ),
            step=1,
            help=HELP_TEXT["active_days_14d"],
        )


        days_since_last_activity = st.number_input(
            LABELS["days_since_last_activity"],
            min_value=0,
            value=int(
                round(
                    get_default_value(
                        "days_since_last_activity"
                    )
                )
            ),
            step=1,
            help=HELP_TEXT["days_since_last_activity"],
        )


    with col2:

        wp_active_day_ratio_7v21 = st.number_input(
            LABELS["wp_active_day_ratio_7v21"],
            min_value=0.0,
            value=float(
                get_default_value(
                    "wp_active_day_ratio_7v21"
                )
            ),
            step=0.01,
            format="%.3f",
            help=HELP_TEXT[
                "wp_active_day_ratio_7v21"
            ],
        )


        wp_active_day_ratio_14v14 = st.number_input(
            LABELS["wp_active_day_ratio_14v14"],
            min_value=0.0,
            value=float(
                get_default_value(
                    "wp_active_day_ratio_14v14"
                )
            ),
            step=0.01,
            format="%.3f",
            help=HELP_TEXT[
                "wp_active_day_ratio_14v14"
            ],
        )


        wp_active_day_diff_7v21 = st.number_input(
            LABELS["wp_active_day_diff_7v21"],
            value=float(
                get_default_value(
                    "wp_active_day_diff_7v21"
                )
            ),
            step=0.01,
            format="%.3f",
            help=HELP_TEXT[
                "wp_active_day_diff_7v21"
            ],
        )


    st.markdown("---")


    # --------------------------------------------------------
    # GROUP 2
    # --------------------------------------------------------

    st.markdown(
        '<div class="group-title">'
        '2. Long-Term Learning Engagement'
        '</div>',
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(
        2,
        gap="large",
    )


    with col3:

        active_days_total_to_date = st.number_input(
            LABELS["active_days_total_to_date"],
            min_value=0,
            value=int(
                round(
                    get_default_value(
                        "active_days_total_to_date"
                    )
                )
            ),
            step=1,
            help=HELP_TEXT[
                "active_days_total_to_date"
            ],
        )


        days_since_first_activity = st.number_input(
            LABELS["days_since_first_activity"],
            min_value=0,
            value=int(
                round(
                    get_default_value(
                        "days_since_first_activity"
                    )
                )
            ),
            step=1,
            help=HELP_TEXT[
                "days_since_first_activity"
            ],
        )


    with col4:

        active_day_density_28d = st.number_input(
            LABELS["active_day_density_28d"],
            min_value=0.0,
            max_value=1.0,
            value=float(
                get_default_value(
                    "active_day_density_28d"
                )
            ),
            step=0.01,
            format="%.3f",
            help=HELP_TEXT[
                "active_day_density_28d"
            ],
        )


        active_weeks_in_28d = st.number_input(
            LABELS["active_weeks_in_28d"],
            min_value=0,
            max_value=4,
            value=int(
                round(
                    get_default_value(
                        "active_weeks_in_28d"
                    )
                )
            ),
            step=1,
            help=HELP_TEXT[
                "active_weeks_in_28d"
            ],
        )


    st.markdown("---")


    # --------------------------------------------------------
    # GROUP 3
    # --------------------------------------------------------

    st.markdown(
        '<div class="group-title">'
        '3. Learning-Activity Regularity'
        '</div>',
        unsafe_allow_html=True,
    )

    col5, col6 = st.columns(
        2,
        gap="large",
    )


    with col5:

        max_inactivity_gap_28d = st.number_input(
            LABELS["max_inactivity_gap_28d"],
            min_value=0.0,
            value=float(
                get_default_value(
                    "max_inactivity_gap_28d"
                )
            ),
            step=0.5,
            format="%.2f",
            help=HELP_TEXT[
                "max_inactivity_gap_28d"
            ],
        )


        mean_activity_gap_28d = st.number_input(
            LABELS["mean_activity_gap_28d"],
            min_value=0.0,
            value=float(
                get_default_value(
                    "mean_activity_gap_28d"
                )
            ),
            step=0.1,
            format="%.2f",
            help=HELP_TEXT[
                "mean_activity_gap_28d"
            ],
        )


    with col6:

        sd_activity_gap_28d = st.number_input(
            LABELS["sd_activity_gap_28d"],
            min_value=0.0,
            value=float(
                get_default_value(
                    "sd_activity_gap_28d"
                )
            ),
            step=0.1,
            format="%.2f",
            help=HELP_TEXT[
                "sd_activity_gap_28d"
            ],
        )


    st.markdown("")

    submitted = st.form_submit_button(
        "Calculate Predicted Risk",
        width="stretch",
    )


# ============================================================
# 14. PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    validation_errors = []


    if active_days_7d > active_days_14d:

        validation_errors.append(
            "Active learning days in the previous 7 days "
            "cannot exceed active learning days in the "
            "previous 14 days."
        )


    if (
        active_days_total_to_date
        < active_days_14d
    ):

        validation_errors.append(
            "Cumulative active learning days cannot be "
            "lower than the number of active learning days "
            "in the previous 14 days."
        )


    if (
        days_since_last_activity
        > days_since_first_activity
    ):

        validation_errors.append(
            "Days since the most recent learning activity "
            "cannot exceed the time from the first learning "
            "activity to the prediction time."
        )


    if (
        mean_activity_gap_28d
        > max_inactivity_gap_28d
    ):

        validation_errors.append(
            "The mean learning-activity gap cannot exceed "
            "the maximum learning-activity gap."
        )


    if validation_errors:

        st.error(
            "Please review the input values:"
        )

        for message in validation_errors:

            st.warning(message)

        st.stop()


    # --------------------------------------------------------
    # Construct exact feature vector
    # --------------------------------------------------------

    input_values = {

        "active_days_7d":
            active_days_7d,

        "active_days_14d":
            active_days_14d,

        "active_days_total_to_date":
            active_days_total_to_date,

        "days_since_last_activity":
            days_since_last_activity,

        "days_since_first_activity":
            days_since_first_activity,

        "max_inactivity_gap_28d":
            max_inactivity_gap_28d,

        "mean_activity_gap_28d":
            mean_activity_gap_28d,

        "sd_activity_gap_28d":
            sd_activity_gap_28d,

        "active_day_density_28d":
            active_day_density_28d,

        "active_weeks_in_28d":
            active_weeks_in_28d,

        "wp_active_day_ratio_7v21":
            wp_active_day_ratio_7v21,

        "wp_active_day_ratio_14v14":
            wp_active_day_ratio_14v14,

        "wp_active_day_diff_7v21":
            wp_active_day_diff_7v21,
    }


    X = pd.DataFrame(
        [[input_values[f] for f in FEATURES]],
        columns=FEATURES,
    )


    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURES,
    )


    # --------------------------------------------------------
    # Predicted probability
    # --------------------------------------------------------

    probability = float(
        booster.predict(
            dmatrix
        )[0]
    )


    # --------------------------------------------------------
    # Native exact TreeSHAP
    # --------------------------------------------------------

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        approx_contribs=False,
    )[0]


    shap_values = contributions[:-1]


    # ========================================================
    # 15. RESULT SECTION
    # ========================================================

    st.markdown("---")


    st.markdown(
        '<div class="section-title">'
        'Prediction Results'
        '</div>',
        unsafe_allow_html=True,
    )


    left, right = st.columns(
        [0.88, 1.45],
        gap="large",
    )


    # --------------------------------------------------------
    # LEFT: predicted probability
    # --------------------------------------------------------

    with left:

        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">'
            f'Predicted probability of learning engagement '
            f'decline during the subsequent 7 days'
            f'</div>'
            f'<div class="result-number">'
            f'{probability * 100:.1f}%'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


        if probability >= THRESHOLD:

            status_title = (
                "Above the development-cohort "
                "reference cut-point"
            )

            status_detail = (
                f"Predicted probability: "
                f"{probability * 100:.1f}% "
                f"≥ reference cut-point: "
                f"{THRESHOLD * 100:.1f}%"
            )

        else:

            status_title = (
                "Below the development-cohort "
                "reference cut-point"
            )

            status_detail = (
                f"Predicted probability: "
                f"{probability * 100:.1f}% "
                f"< reference cut-point: "
                f"{THRESHOLD * 100:.1f}%"
            )


        st.markdown(
            f'<div class="status-card">'
            f'<div class="status-title">'
            f'{status_title}'
            f'</div>'
            f'<div class="status-detail">'
            f'{status_detail}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )


        st.markdown(
            '<div class="note-box">'
            'This prediction is intended to support research-oriented '
            'identification of students who may experience a decline '
            'in learning engagement. The result should be interpreted '
            'together with course progress, learning-task arrangements, '
            'and instructor observations, and should not be used as '
            'the sole basis for educational decisions.'
            '</div>',
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # RIGHT: local SHAP explanation
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### Main Feature Contributions"
        )


        shap_df = pd.DataFrame({

            "feature":
                FEATURES,

            "label":
                [
                    SHAP_LABELS[f]
                    for f in FEATURES
                ],

            "shap_value":
                shap_values,
        })


        shap_df["abs_shap"] = (
            shap_df[
                "shap_value"
            ].abs()
        )


        top6 = (
            shap_df
            .sort_values(
                "abs_shap",
                ascending=False,
            )
            .head(6)
            .copy()
        )


        # Display in contribution order
        label_order = (
            top6
            .sort_values(
                "abs_shap",
                ascending=False,
            )["label"]
            .tolist()
        )


        top6["Direction"] = np.where(
            top6["shap_value"] >= 0,
            "Higher predicted risk",
            "Lower predicted risk",
        )


        base_chart = (
            alt.Chart(top6)
            .mark_bar(
                cornerRadiusEnd=3,
                size=23,
            )
            .encode(

                x=alt.X(
                    "shap_value:Q",
                    title="SHAP value (log-odds)",
                    axis=alt.Axis(
                        grid=True,
                        gridOpacity=0.16,
                        labelFontSize=11,
                        titleFontSize=12,
                    ),
                ),

                y=alt.Y(
                    "label:N",
                    sort=label_order,
                    title=None,
                    axis=alt.Axis(
                        labelLimit=300,
                        labelFontSize=12,
                    ),
                ),

                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(
                        domain=[
                            "Higher predicted risk",
                            "Lower predicted risk",
                        ],
                        range=[
                            "#F59E0B",
                            "#2563EB",
                        ],
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                        labelFontSize=11,
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "label:N",
                        title="Feature",
                    ),

                    alt.Tooltip(
                        "shap_value:Q",
                        title="SHAP value",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=340,
            )
        )


        zero_line = (
            alt.Chart(
                pd.DataFrame(
                    {"x": [0]}
                )
            )
            .mark_rule(
                strokeDash=[5, 4],
                color="#98A2B3",
                strokeWidth=1.2,
            )
            .encode(
                x="x:Q"
            )
        )


        final_chart = (
            base_chart
            + zero_line
        ).configure_view(
            strokeWidth=0
        )


        st.altair_chart(
            final_chart,
            width="stretch",
        )


        st.caption(
            "Positive SHAP values increase the predicted risk, "
            "whereas negative SHAP values decrease it. "
            "Values are shown on the model's raw log-odds scale."
        )


    # ========================================================
    # 16. INPUT DETAILS
    # ========================================================

    st.markdown("")


    with st.expander(
        "View input values",
        expanded=False,
    ):

        input_table = pd.DataFrame({

            "Feature":
                [
                    LABELS[f]
                    for f in FEATURES
                ],

            "Input value":
                [
                    input_values[f]
                    for f in FEATURES
                ],
        })


        st.dataframe(
            input_table,
            width="stretch",
            hide_index=True,
        )


    # ========================================================
    # 17. DOWNLOAD RESULT
    # ========================================================

    output_record = {

        "predicted_probability":
            probability,

        "development_reference_cutpoint":
            THRESHOLD,

        "above_reference_cutpoint":
            int(
                probability
                >= THRESHOLD
            ),
    }


    for feature in FEATURES:

        output_record[feature] = (
            input_values[feature]
        )


    result_df = pd.DataFrame(
        [output_record]
    )


    csv_buffer = io.StringIO()


    result_df.to_csv(
        csv_buffer,
        index=False,
    )


    st.download_button(
        label="Download Prediction Result (CSV)",
        data=csv_buffer.getvalue(),
        file_name=(
            "learning_engagement_"
            "risk_prediction.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.markdown("---")


st.caption(
    "Research-use prediction tool. "
    "The model estimates the probability of a future 7-day decline "
    "in learning engagement using learning-platform behavioral features. "
    "Model predictions represent associations and should not be "
    "interpreted as causal effects."
)