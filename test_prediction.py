import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[0]))
import numpy as np
import pandas as pd
import pytest
from prediction import load_bundle,predict,read_csv

@pytest.fixture
def resources():
    b,m=load_bundle()
    return b,m

@pytest.fixture
def frame(resources):
    b,_=resources
    # Technical inputs from fitted imputer statistics, never shown as student predictions.
    d=pd.DataFrame([b['imputer'].statistics_]*3,columns=b['predictors'])
    d.insert(0,'student_id',['test-1','test-2','test-3'])
    return d

def test_bundle(resources):
    b,m=resources
    assert len(b['predictors'])==18
    assert b['scaled'] is False
    assert m['development_oof_youden_threshold']==0.3094141475972338

def test_direct_model_batch_and_column_order(frame,resources):
    b,m=resources
    expected=b['model'].predict_proba(b['imputer'].transform(frame[b['predictors']]))[:,1]
    result=predict(frame[frame.columns[::-1]],b,m)
    np.testing.assert_array_equal(result.decline_probability_7d,expected)
    assert len(result)==3
    assert result.student_id.tolist()==frame.student_id.tolist()
    assert result.risk_class.tolist()==np.where(expected>=m['development_oof_youden_threshold'],'高风险','较低风险').tolist()

def test_csv_roundtrip(frame,resources):
    b,m=resources
    frame['student_id']=['001','002','003']
    loaded=read_csv(frame.to_csv(index=False).encode('utf-8-sig'))
    assert loaded.student_id.tolist()==['001','002','003']
    np.testing.assert_allclose(predict(loaded,b,m).decline_probability_7d,predict(frame,b,m).decline_probability_7d,atol=0,rtol=0)

@pytest.mark.parametrize('kind',['missing_column','missing_value','no_peer','infinite','non_numeric','bad_range','bad_count','no_id','empty'])
def test_invalid_inputs(frame,resources,kind):
    b,m=resources
    if kind=='missing_column':frame=frame.drop(columns=b['predictors'][0])
    elif kind=='missing_value':frame.loc[0,b['predictors'][0]]=np.nan
    elif kind=='no_peer':frame.loc[0,'C_peer_relative_7d_LOO']=np.nan
    elif kind=='infinite':frame.loc[0,'mean_activity_gap_28d']=np.inf
    elif kind=='non_numeric':frame[b['predictors'][0]]='bad'
    elif kind=='bad_range':frame.loc[0,'active_days_7d']=8
    elif kind=='bad_count':frame.loc[0,'active_days_7d']=1.5
    elif kind=='no_id':frame.loc[0,'student_id']=''
    elif kind=='empty':frame=frame.iloc[:0]
    with pytest.raises(ValueError):predict(frame,b,m)

def test_duplicate_csv():
    with pytest.raises(ValueError):read_csv(b'student_id,a,a\nx,1,2\n')

def test_app_start_and_single_prediction(frame):
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(Path(__file__).resolve().parents[0]/'app.py'),default_timeout=30).run()
    assert not at.exception
    at.radio[0].set_value('单个学生完整特征').run()
    at.text_area[0].set_value(frame.iloc[:1].to_csv(index=False)).run()
    at.checkbox[0].check().run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric)==1
    assert not at.error
