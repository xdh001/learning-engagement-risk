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
        raise ValueError('模型文件校验失败，请恢复已验证的模型。')
    b = joblib.load(path)
    if not {'model','imputer','predictors','scaled'} <= b.keys():
        raise ValueError('模型包缺少必要对象。')
    if not isinstance(b['model'], HistGradientBoostingClassifier):
        raise ValueError('实际模型不是HistGradientBoostingClassifier。')
    f = list(b['predictors'])
    if len(f) != 18 or len(set(f)) != 18 or f != meta['predictors']:
        raise ValueError('18项特征或顺序与已核对元数据不一致。')
    if list(b['imputer'].feature_names_in_) != f or b['model'].n_features_in_ != 18:
        raise ValueError('模型与填补器的输入特征不一致。')
    if b['scaled'] and 'scaler' not in b:
        raise ValueError('缺少训练时的标准化器。')
    if list(b['model'].classes_) != [0,1]:
        raise ValueError('结局类别编码不符合0/1定义。')
    return b, meta

def read_csv(data):
    text = data.decode('utf-8-sig')
    columns = next(csv.reader(io.StringIO(text)), [])
    if len(columns) != len(set(columns)):
        raise ValueError('CSV含重复列名。')
    return pd.read_csv(io.StringIO(text), dtype={'student_id':str}, keep_default_na=True)

def validate(frame, predictors):
    if frame.empty:
        raise ValueError('请输入至少一名学生。')
    if frame.columns.duplicated().any():
        raise ValueError('输入含重复列名。')
    absent = [f for f in ['student_id', *predictors] if f not in frame.columns]
    if absent:
        raise ValueError('缺少必要字段：' + '、'.join(absent))
    if frame.student_id.isna().any() or frame.student_id.astype(str).str.strip().eq('').any():
        raise ValueError('学生编号不能为空。')
    raw = frame.loc[:, predictors]
    numeric = raw.apply(pd.to_numeric, errors='coerce')
    invalid = raw.notna() & numeric.isna()
    if invalid.any().any():
        raise ValueError('特征须为数值；非数值字段：' + '、'.join(invalid.columns[invalid.any()]))
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError('特征不能包含无穷大。')
    # Missing inputs cannot substitute for unavailable peer/history feature construction.
    if numeric.isna().any().any():
        raise ValueError('18项必要特征均须提供。请用原实验流程生成缺失的历史或同伴特征；不得填0替代。')
    limits = {'active_days_7d':7,'active_days_14d':14,'active_day_density_28d':1,
              'active_weeks_in_28d':4,'max_inactive_streak_14d':14,
              'active_weeks_ge3days_28d':4,'activity_transition_count_28d':27}
    signed = {'weekly_active_slope_28d','B_self_relative_7d_vs_29to42d','C_peer_relative_7d_LOO'}
    for f in predictors:
        if f not in signed and (numeric[f]<0).any():
            raise ValueError(f+'不能为负数。')
        if f in limits and (numeric[f]>limits[f]).any():
            raise ValueError(f+'超过允许范围。')
    counts = ['active_days_7d','active_days_14d','days_since_last_activity','active_weeks_in_28d',
              'max_inactive_streak_14d','long_gap_count_28d','active_weeks_ge3days_28d','activity_transition_count_28d']
    if any((numeric[f] % 1 != 0).any() for f in counts):
        raise ValueError('日数、周数和次数须为整数。')
    if (numeric.active_days_7d > numeric.active_days_14d).any():
        raise ValueError('近7天活跃日数不能超过近14天。')
    if (numeric.active_weeks_ge3days_28d > numeric.active_weeks_in_28d).any():
        raise ValueError('每周至少活跃3天的周数不能超过活跃周数。')
    return numeric

def predict(frame, bundle, metadata):
    x = validate(frame, list(bundle['predictors']))
    transformed = bundle['imputer'].transform(x)
    if bundle['scaled']:
        transformed = bundle['scaler'].transform(transformed)
    p = bundle['model'].predict_proba(transformed)[:,1]
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('模型返回无效概率。')
    threshold = metadata['development_oof_youden_threshold']
    if not isinstance(threshold,(int,float)) or not 0 < threshold < 1:
        raise ValueError('缺少经核对的开发队列OOF阈值。')
    out = frame[['student_id']].copy()
    for col in ['domain_id','landmark_day']:
        if col in frame: out[col] = frame[col].values
    out['decline_probability_7d'] = p
    out['risk_class'] = np.where(p >= threshold, '高风险','较低风险')
    out['risk_message'] = np.where(p >= threshold,
        '建议教师结合课程安排核查近期学习情况，提供学习支持。',
        '建议继续观察学习变化，保持常规学习支持。')
    out['classification_threshold'] = threshold
    return out
