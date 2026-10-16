"""Inference only. Feature construction must use the frozen experiment pipeline."""
from pathlib import Path
import csv
import hashlib
import io
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent

def load_bundle(root=ROOT):
    root = Path(root)
    meta = json.loads((root / 'model/metadata.json').read_text())
    path = root / 'model/HistGB_fullDevelopment.pkl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['model_sha256']:
        raise ValueError('Model checksum verification failed. Restore the verified model.')
    b = joblib.load(path)
    if not {'model','imputer','predictors','scaled'} <= b.keys():
        raise ValueError('The model bundle is missing required objects.')
    if not isinstance(b['model'], HistGradientBoostingClassifier):
        raise ValueError('The loaded model is not a HistGradientBoostingClassifier.')
    f = list(b['predictors'])
    if len(f) != 18 or len(set(f)) != 18 or f != meta['predictors']:
        raise ValueError('The 18 features or their order do not match the verified metadata.')
    if list(b['imputer'].feature_names_in_) != f or b['model'].n_features_in_ != 18:
        raise ValueError('Model and imputer input features do not match.')
    if b['scaled'] and 'scaler' not in b:
        raise ValueError('The saved training scaler is missing.')
    if list(b['model'].classes_) != [0,1]:
        raise ValueError('Outcome classes must be coded as 0 and 1.')
    return b, meta

def read_csv(data):
    text = data.decode('utf-8-sig')
    columns = next(csv.reader(io.StringIO(text)), [])
    if len(columns) != len(set(columns)):
        raise ValueError('The CSV contains duplicate column names.')
    return pd.read_csv(io.StringIO(text), dtype={'student_id':str}, keep_default_na=True)

def validate(frame, predictors):
    if frame.empty:
        raise ValueError('Please provide at least one student.')
    if frame.columns.duplicated().any():
        raise ValueError('The input contains duplicate column names.')
    absent = [f for f in ['student_id', *predictors] if f not in frame.columns]
    if absent:
        raise ValueError('Missing required fields: ' + ', '.join(absent))
    if frame.student_id.isna().any() or frame.student_id.astype(str).str.strip().eq('').any():
        raise ValueError('Student ID must not be blank.')
    raw = frame.loc[:, predictors]
    numeric = raw.apply(pd.to_numeric, errors='coerce')
    invalid = raw.notna() & numeric.isna()
    if invalid.any().any():
        raise ValueError('Features must be numeric. Non-numeric fields: ' + ', '.join(invalid.columns[invalid.any()]))
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError('Features must not contain infinity.')
    # Missing inputs cannot substitute for unavailable peer/history feature construction.
    if numeric.isna().any().any():
        raise ValueError('All 18 features are required. Generate missing history or peer features using the original experiment pipeline; do not substitute zero.')
    limits = {'active_days_7d':7,'active_days_14d':14,'active_day_density_28d':1,
              'active_weeks_in_28d':4,'max_inactive_streak_14d':14,
              'active_weeks_ge3days_28d':4,'activity_transition_count_28d':27}
    signed = {'weekly_active_slope_28d','B_self_relative_7d_vs_29to42d','C_peer_relative_7d_LOO'}
    for f in predictors:
        if f not in signed and (numeric[f]<0).any():
            raise ValueError(f+' must not be negative.')
        if f in limits and (numeric[f]>limits[f]).any():
            raise ValueError(f+' exceeds its allowed range.')
    counts = ['active_days_7d','active_days_14d','days_since_last_activity','active_weeks_in_28d',
              'max_inactive_streak_14d','long_gap_count_28d','active_weeks_ge3days_28d','activity_transition_count_28d']
    if any((numeric[f] % 1 != 0).any() for f in counts):
        raise ValueError('Day, week and event counts must be integers.')
    if (numeric.active_days_7d > numeric.active_days_14d).any():
        raise ValueError('Active days in the past 7 days cannot exceed those in the past 14 days.')
    if (numeric.active_weeks_ge3days_28d > numeric.active_weeks_in_28d).any():
        raise ValueError('Weeks with at least 3 active days cannot exceed active weeks.')
    return numeric

def predict(frame, bundle, metadata):
    x = validate(frame, list(bundle['predictors']))
    transformed = bundle['imputer'].transform(x)
    if bundle['scaled']:
        transformed = bundle['scaler'].transform(transformed)
    p = bundle['model'].predict_proba(transformed)[:,1]
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('The model returned an invalid probability.')
    threshold = metadata['development_oof_youden_threshold']
    if not isinstance(threshold,(int,float)) or not 0 < threshold < 1:
        raise ValueError('The verified development-cohort OOF threshold is missing.')
    out = frame[['student_id']].copy()
    for col in ['domain_id','landmark_day']:
        if col in frame: out[col] = frame[col].values
    out['decline_probability_7d'] = p
    out['risk_class'] = np.where(p >= threshold, 'High risk','Lower risk')
    out['risk_message'] = np.where(p >= threshold,
        'Review recent learning activity alongside the course schedule and offer timely learning support.',
        'Continue monitoring learning activity and maintain routine support.')
    out['classification_threshold'] = threshold
    return out
