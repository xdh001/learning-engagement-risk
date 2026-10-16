import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from html import escape
import pandas as pd
import streamlit as st
from prediction import load_bundle, predict, read_csv

st.set_page_config(page_title='Learning Engagement · Risk Prediction', page_icon='🌱', layout='wide', initial_sidebar_state='collapsed')

@st.cache_resource
def resources():
    return load_bundle()

st.markdown('''<style>

:root{--primary-color:#0e8174} .stApp{background:#f5f8fb;color:#20334a;font-family:'Noto Sans SC',sans-serif}
.block-container{max-width:1200px;padding-top:2.3rem;padding-bottom:3rem}
[data-testid="stSidebar"]{display:none} header[data-testid="stHeader"]{background:transparent}
.hero{padding:30px 34px;border-radius:22px;background:linear-gradient(115deg,#123c51,#116f72);color:white;margin-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:2px;color:#a9ded8;margin-bottom:12px}.hero h1{font-size:31px!important;font-weight:650;letter-spacing:1px;color:white;margin:0 0 10px;padding:0}.hero p{color:#d6e9ec;font-size:14px;margin:0;line-height:1.9}
.tag{display:inline-block;border:1px solid #ffffff33;background:#ffffff10;border-radius:30px;padding:5px 12px;margin:16px 8px 0 0;font-size:12px;color:#def2ed}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:26px;border-bottom:1px solid #dce5ed;margin-bottom:20px}
[data-testid="stTabs"] [data-baseweb="tab"]{font-size:16px;padding:12px 4px;height:auto}
[data-testid="stTabs"] [aria-selected="true"]{color:#087b6e!important;font-weight:700}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:#0e8174}
[data-testid="stForm"]{background:white;border:1px solid #e1e8f0;border-radius:18px;padding:24px}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:#fafcfe}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]{background:#0e8174;border-color:#0e8174;border-radius:10px}
[data-testid="stBaseButton-secondary"]{border-color:#dbe4ed;border-radius:10px}
.step{font-size:17px;font-weight:650;margin:20px 0 10px;color:#193e52}.hint{font-size:13px;color:#718294;margin:0 0 15px;line-height:1.8}
.result{background:white;border:1px solid #dae7e9;border-radius:18px;padding:25px 28px;margin:18px 0}.result .prob{font-size:46px;font-weight:700;color:#137e72;line-height:1.4}.badge{display:inline-block;border-radius:30px;padding:5px 13px;font-size:13px;background:#e7f4ef;color:#16715c}.badge.high{background:#fff0e5;color:#b46020}.foot{font-size:12px;color:#8796a5;margin-top:30px;line-height:1.8}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">LEARNING ENGAGEMENT · TEACHING SUPPORT</div><h1>Notice changes. Support learning.</h1><p>Estimate the risk of engagement decline over the next 7 days and identify students who may benefit from timely support.</p><span class="tag">Individual assessment</span><span class="tag">Batch CSV upload</span><span class="tag">Downloadable results</span></div>''', unsafe_allow_html=True)
try:
    bundle, meta = resources()
except Exception as exc:
    st.error(f'Unable to load the model: {exc}')
    st.stop()
features = bundle['predictors']
labels = ['Active days in the past 7 days', 'Active days in the past 14 days', 'Days since last activity', 'Maximum activity gap in the past 28 days', 'Mean activity gap in the past 28 days', 'SD of activity gaps in the past 28 days', 'Active-day density in the past 28 days', 'Active weeks in the past 28 days', 'Weekly active-day ratio: recent 7 vs preceding 21 days', 'Active-day ratio: recent 14 vs preceding 14 days', 'Weekly active-day slope in the past 28 days', 'SD of weekly active days in the past 28 days', 'Longest inactive streak in the past 14 days', 'Long-gap count in the past 28 days', 'Weeks with at least 3 active days in the past 28 days', 'Activity-state transitions in the past 28 days', 'Personal relative activity: recent 7 vs days 29–42', 'Peer-relative activity over the past 7 days (leave-one-out)']
definitions = ['Number of days with valid learning activity in the preceding 7 days.', 'Number of days with valid learning activity in the preceding 14 days.', 'Days elapsed since the most recent valid learning activity.', 'Maximum interval between learning activities in the preceding 28 days.', 'Mean interval between learning activities in the preceding 28 days.', 'Standard deviation of intervals between learning activities in the preceding 28 days.', 'Active days in the preceding 28 days divided by 28.', 'Number of the four weekly windows with at least one active day.', 'Weekly active days in the recent 7 days divided by the weekly average in the preceding 21 days.', 'Active days in the recent 14 days divided by active days in the preceding 14 days.', 'Slope of active-day counts across four consecutive 7-day windows.', 'Standard deviation of active-day counts across four consecutive 7-day windows.', 'Longest run of consecutive inactive days in the preceding 14 days.', 'Number of gaps exceeding the experiment cutoff; the exact cutoff requires verification against the original feature-generation code.', 'Number of four weekly windows containing at least three active days.', 'Number of transitions between daily active and inactive states in the preceding 28 days.', 'Recent 7-day activity relative to personal activity on days 29–42; the exact normalization formula requires verification.', 'Difference from the mean activity of other learners in the same course at the same prediction time; the learner is excluded from the peer mean.']
dictionary = {f: {'name': name, 'definition': definition} for f, name, definition in zip(features, labels, definitions)}
# Explicitly artificial examples, with probabilities always computed by the frozen model.
DEMO = [3,5,1,6,2.3,1.5534552,10/28,4,1,1,0,1.0897247,4,2,2,9,0,-0.26709062]
integers = {'active_days_7d','active_days_14d','days_since_last_activity','active_weeks_in_28d','max_inactive_streak_14d','long_gap_count_28d','active_weeks_ge3days_28d','activity_transition_count_28d'}
upper = {'active_days_7d':7,'active_days_14d':14,'active_day_density_28d':1,'active_weeks_in_28d':4,'max_inactive_streak_14d':14,'active_weeks_ge3days_28d':4,'activity_transition_count_28d':27}
signed = {'weekly_active_slope_28d','B_self_relative_7d_vs_29to42d','C_peer_relative_7d_LOO'}

def populate(demo=False):
    st.session_state['student_id'] = 'DEMO_001' if demo else ''
    for f, v in zip(features, DEMO):
        st.session_state['input_'+f] = (int(v) if f in integers else float(v)) if demo else None
    st.session_state['demo_loaded'] = demo
    st.session_state.pop('single_result', None)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


def show_result(result, key):
    if len(result) == 1:
        row = result.iloc[0]
        high = row.risk_class == 'High risk'
        st.markdown(f'''<div class="result"><div class="hint">Student {escape(str(row.student_id))} · Assessment result</div><span class="badge {'high' if high else ''}">{escape(row.risk_class)}</span><div class="prob">{row.decline_probability_7d:.2%}</div><div>Predicted probability of engagement decline within 7 days</div><p class="hint" style="margin-top:14px;margin-bottom:0">{escape(row.risk_message)}</p></div>''', unsafe_allow_html=True)
    else:
        a,b,c = st.columns(3)
        a.metric('Students assessed', f'{len(result)} students')
        b.metric('High-risk students', f'{(result.risk_class=="High risk").sum()} students')
        c.metric('Mean predicted probability', f'{result.decline_probability_7d.mean():.1%}')
        view = result[['student_id','decline_probability_7d','risk_class','risk_message']].rename(columns={'student_id':'Student ID','decline_probability_7d':'Decline probability','risk_class':'Risk category','risk_message':'Support recommendation'})
        st.dataframe(view, hide_index=True, use_container_width=True, column_config={'Decline probability':st.column_config.NumberColumn(format='percent')})
    st.download_button('Download predictions CSV', csv_bytes(result), 'engagement_risk_predictions.csv', 'text/csv', key=key, use_container_width=True)
    st.caption('Outcome: at least a 50% decline in active days over the next 7 days relative to the average weekly active days in the preceding 14 days. Risk categories use the development-cohort threshold.')

single, batch, guide = st.tabs(['  Individual prediction  ','  Batch prediction  ','  Guide & features  '])
with single:
    left,right=st.columns([1.55,1],gap='large')
    with left:
        st.subheader('Enter learning behavior features')
        st.caption('Enter all 18 precomputed features, or load the demo to try an assessment.')
    with right:
        x,y=st.columns(2)
        x.button('✨ Load demo', on_click=populate, args=(True,), use_container_width=True)
        y.button('Clear form', on_click=populate, use_container_width=True)
    if st.session_state.get('demo_loaded'):
        st.info('Synthetic demo data loaded. Run an assessment to calculate its probability using the actual model. Clear the form before entering real student data.')
    with st.form('student_form',border=True):
        st.text_input('Student ID', key='student_id', placeholder='e.g. S2026001')
        groups=[('01 Recent learning activity',features[:3]+[features[6],features[7],features[14]]),('02 Learning patterns & changes',features[3:6]+features[8:14]+[features[15]]),('03 Personal history & peer comparison',features[16:])]
        for title, group in groups:
            st.markdown(f'<div class="step">{title}</div>',unsafe_allow_html=True)
            if group==features[16:]:
                st.caption('Use values generated by the original feature pipeline. Peer comparison requires other learners from the same course at the same prediction time.')
            cols=st.columns(2)
            for i,f in enumerate(group):
                d=dictionary[f]
                isint=f in integers
                cols[i%2].number_input(d['name'], value=None, min_value=None if f in signed else (0 if isint else 0.0), max_value=(int(upper[f]) if isint else float(upper[f])) if f in upper else None, step=1 if isint else 0.01, format='%d' if isint else '%.6f', key='input_'+f, placeholder='Enter a value', help=d['definition']+'\n\nFeature: '+f)
        st.caption('All 18 features are required. History, relative-activity and gap features must come from the original feature pipeline; do not substitute estimates.')
        submitted=st.form_submit_button('Assess risk →', type='primary',use_container_width=True)
    if submitted:
        st.session_state.pop('single_result',None)
        missing=[dictionary[f]['name'] for f in features if st.session_state.get('input_'+f) is None]
        if missing:
            st.error('Please complete these features: '+', '.join(missing))
        else:
            try:
                frame=pd.DataFrame([{'student_id':st.session_state.student_id,**{f:st.session_state['input_'+f] for f in features}}])
                st.session_state['single_result']=predict(frame,bundle,meta)
            except ValueError as exc:
                st.error(f'Please check your input: {exc}')
    if 'single_result' in st.session_state:
        if st.session_state.get('demo_loaded'): st.caption('Result for synthetic demo data, calculated by the actual model; this is not a real student.')
        show_result(st.session_state.single_result,'single_download')
with batch:
    st.subheader('Assess multiple students')
    st.caption('Use one row per student, with student_id and all 18 features. Download the template and populate it using the original feature pipeline.')
    a,b=st.columns(2)
    a.download_button('↓ Download blank CSV template', csv_bytes(pd.DataFrame(columns=['student_id']+features)), 'feature_template_18.csv','text/csv',use_container_width=True)
    demo_frame=pd.DataFrame([['DEMO_001']+DEMO,['DEMO_002']+DEMO],columns=['student_id']+features)
    b.download_button('↓ Download demo CSV (2 students)',csv_bytes(demo_frame),'demo_features_18.csv','text/csv',use_container_width=True)
    st.caption('The demo CSV contains synthetic data for testing the format and workflow; it does not represent real students.')
    upload=st.file_uploader('Upload student features CSV',type=['csv'],help='UTF-8 encoding. Use the template column names. Leading zeros in student IDs are preserved.')
    batch_frame=None
    if upload is not None:
        try:
            batch_frame=read_csv(upload.getvalue())
            st.caption(f'{len(batch_frame)} rows loaded · Preview of the first 10 rows')
            st.dataframe(batch_frame.head(10),hide_index=True,use_container_width=True)
        except ValueError as exc: st.error(str(exc))
    if st.button('Assess batch →',type='primary',disabled=batch_frame is None,use_container_width=True):
        st.session_state.pop('batch_result',None)
        try: st.session_state['batch_result']=(upload.getvalue(),predict(batch_frame,bundle,meta))
        except ValueError as exc: st.error(f'Please check your CSV: {exc}')
    if upload is not None and 'batch_result' in st.session_state and st.session_state.batch_result[0]==upload.getvalue():
        show_result(st.session_state.batch_result[1],'batch_download')
with guide:
    st.subheader('Three steps to an assessment')
    st.markdown('**1 · Prepare your data**  Obtain all 18 features using the original experiment pipeline.\n\n**2 · Choose an input method**  Use the form for one student or upload a CSV for a class.\n\n**3 · Review the results**  View probabilities, risk categories and support recommendations, then download the results.')
    st.markdown('#### Understanding the prediction')
    st.write('The outcome is a decline of at least 50% in active days during the next 7 days, relative to average weekly active days in the preceding 14 days. Higher probabilities suggest a greater need to review the student’s situation. Consider the course schedule and student feedback when offering support.')
    with st.expander('View all 18 feature names and definitions'):
        st.dataframe(pd.DataFrame([{'CSV field name':f,'Feature':dictionary[f]['name'],'Definition':dictionary[f]['definition']} for f in features]),hide_index=True,use_container_width=True)
        st.info('This application accepts precomputed features. The long-gap cutoff and the normalization formula for personal historical activity still require verification against the original feature-generation code. Use values already generated by that pipeline. These features are not reconstructed from simplified inputs or a single student’s events.')
    with st.expander('Model & classification details'):
        st.write('Predictions use the frozen 18-feature HistGradientBoosting model and its saved imputer, with features ordered exactly as saved in the model bundle.')
        st.write(f'Classification threshold: {meta["development_oof_youden_threshold"]:.16f}, the Youden threshold from out-of-fold predictions in the development cohort.')
st.markdown('<div class="foot">Learning Engagement Risk Prediction · Teaching support & research<br>Use predictions to inform conversations and learning support.</div>',unsafe_allow_html=True)
