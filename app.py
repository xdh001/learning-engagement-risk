import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st
from prediction import load_bundle, predict, read_csv

st.set_page_config(page_title='学习参与度 · 风险预测', page_icon='🌱', layout='wide', initial_sidebar_state='collapsed')

@st.cache_resource
def resources():
    return load_bundle()

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
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
st.markdown('''<div class="hero"><div class="eyebrow">LEARNING ENGAGEMENT · 教学支持工具</div><h1>让每一份学习变化，都被及时看见</h1><p>根据学生的学习行为，评估未来 7 天学习参与度下降的可能性，帮助教师及时了解与支持学生。</p><span class="tag">单个学生评估</span><span class="tag">班级批量导入</span><span class="tag">预测结果下载</span></div>''', unsafe_allow_html=True)
try:
    bundle, meta = resources()
except Exception as exc:
    st.error(f'模型暂时无法加载：{exc}')
    st.stop()
features = bundle['predictors']
dictionary = {d['field']: d for d in meta['feature_dictionary']}
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
        high = row.risk_class == '高风险'
        st.markdown(f'''<div class="result"><div class="hint">学生 {escape(str(row.student_id))} · 本次评估结果</div><span class="badge {'high' if high else ''}">{escape(row.risk_class)}</span><div class="prob">{row.decline_probability_7d:.2%}</div><div>未来 7 天学习参与度下降预测概率</div><p class="hint" style="margin-top:14px;margin-bottom:0">{escape(row.risk_message)}</p></div>''', unsafe_allow_html=True)
    else:
        a,b,c = st.columns(3)
        a.metric('完成评估', f'{len(result)} 人')
        b.metric('高风险学生', f'{(result.risk_class=="高风险").sum()} 人')
        c.metric('平均预测概率', f'{result.decline_probability_7d.mean():.1%}')
        view = result[['student_id','decline_probability_7d','risk_class','risk_message']].rename(columns={'student_id':'学生编号','decline_probability_7d':'下降预测概率','risk_class':'风险分类','risk_message':'支持建议'})
        st.dataframe(view, hide_index=True, use_container_width=True, column_config={'下降预测概率':st.column_config.NumberColumn(format='percent')})
    st.download_button('下载预测结果 CSV', csv_bytes(result), 'engagement_risk_predictions.csv', 'text/csv', key=key, use_container_width=True)
    st.caption('结局：未来7天活跃日数，相比预测前14天的平均每周活跃日数下降至少50%。风险分类采用开发队列确定的阈值。')

single, batch, guide = st.tabs(['  单人预测  ','  批量预测  ','  使用说明与字段  '])
with single:
    left,right=st.columns([1.55,1],gap='large')
    with left:
        st.subheader('填写学生学习行为')
        st.caption('已有18项特征值？填写后即可评估。首次使用可先载入示例。')
    with right:
        x,y=st.columns(2)
        x.button('✨ 载入演示示例', on_click=populate, args=(True,), use_container_width=True)
        y.button('清空表单', on_click=populate, use_container_width=True)
    if st.session_state.get('demo_loaded'):
        st.info('当前已载入人工构造的演示数据，仅用于体验操作；点击下方按钮可使用真实模型计算演示概率。用于实际学生前，请清空并填入真实特征。')
    with st.form('student_form',border=True):
        st.text_input('学生编号', key='student_id', placeholder='例如：S2026001')
        groups=[('01 近期学习情况',features[:3]+[features[6],features[7],features[14]]),('02 学习节奏与变化',features[3:6]+features[8:14]+[features[15]]),('03 个人历史与同伴比较',features[16:])]
        for title, group in groups:
            st.markdown(f'<div class="step">{title}</div>',unsafe_allow_html=True)
            if group==features[16:]:
                st.caption('请使用原特征流程已计算的数值；同伴指标需要同课程、同预测时点的班级数据。')
            cols=st.columns(2)
            for i,f in enumerate(group):
                d=dictionary[f]
                isint=f in integers
                cols[i%2].number_input(d['中文名称'], value=None, min_value=None if f in signed else (0 if isint else 0.0), max_value=(int(upper[f]) if isint else float(upper[f])) if f in upper else None, step=1 if isint else 0.01, format='%d' if isint else '%.6f', key='input_'+f, placeholder='请输入', help=d['定义']+'\n\n字段：'+f)
        st.caption('全部18项均需提供。历史、相对与间隔特征应由原实验特征流程生成，请勿使用估计值填补。')
        submitted=st.form_submit_button('开始风险评估 →', type='primary',use_container_width=True)
    if submitted:
        st.session_state.pop('single_result',None)
        missing=[dictionary[f]['中文名称'] for f in features if st.session_state.get('input_'+f) is None]
        if missing:
            st.error('请补齐以下学习行为数据：'+'、'.join(missing))
        else:
            try:
                frame=pd.DataFrame([{'student_id':st.session_state.student_id,**{f:st.session_state['input_'+f] for f in features}}])
                st.session_state['single_result']=predict(frame,bundle,meta)
            except ValueError as exc:
                st.error(f'请检查输入：{exc}')
    if 'single_result' in st.session_state:
        if st.session_state.get('demo_loaded'): st.caption('以下为演示数据的真实模型计算结果，不代表真实学生。')
        show_result(st.session_state.single_result,'single_download')
with batch:
    st.subheader('一次评估多名学生')
    st.caption('每行一名学生，包含学生编号与全部18项特征。先下载模板，再按原特征流程填入数据。')
    a,b=st.columns(2)
    a.download_button('↓ 下载空白 CSV 模板', csv_bytes(pd.DataFrame(columns=['student_id']+features)), 'feature_template_18.csv','text/csv',use_container_width=True)
    demo_frame=pd.DataFrame([['DEMO_001']+DEMO,['DEMO_002']+DEMO],columns=['student_id']+features)
    b.download_button('↓ 下载演示 CSV（2人）',csv_bytes(demo_frame),'demo_features_18.csv','text/csv',use_container_width=True)
    st.caption('演示CSV为人工构造数据，用于测试格式和操作，不代表真实学生。')
    upload=st.file_uploader('上传学生特征 CSV',type=['csv'],help='UTF-8编码；列名必须与模板一致，学生编号可保留前导零。')
    batch_frame=None
    if upload is not None:
        try:
            batch_frame=read_csv(upload.getvalue())
            st.caption(f'已读取 {len(batch_frame)} 行 · 下方预览前10行')
            st.dataframe(batch_frame.head(10),hide_index=True,use_container_width=True)
        except ValueError as exc: st.error(str(exc))
    if st.button('开始批量评估 →',type='primary',disabled=batch_frame is None,use_container_width=True):
        st.session_state.pop('batch_result',None)
        try: st.session_state['batch_result']=(upload.getvalue(),predict(batch_frame,bundle,meta))
        except ValueError as exc: st.error(f'请检查CSV：{exc}')
    if upload is not None and 'batch_result' in st.session_state and st.session_state.batch_result[0]==upload.getvalue():
        show_result(st.session_state.batch_result[1],'batch_download')
with guide:
    st.subheader('三步完成评估')
    st.markdown('**1 · 准备数据**　从与原实验一致的特征流程取得18项学习行为指标。\n\n**2 · 选择方式**　单名学生填写中文表单，多名学生上传CSV。\n\n**3 · 查看结果**　查看下降概率、风险分类与支持建议，按需下载结果。')
    st.markdown('#### 如何理解预测结果')
    st.write('主要结局：以预测前14天平均每周活跃日数为基线，未来7天活跃日数下降至少50%。较高的预测概率表示更需要及时了解学生情况；建议结合课程安排和学生反馈提供支持。')
    with st.expander('查看18项字段的名称与定义'):
        st.dataframe(pd.DataFrame(meta['feature_dictionary']).rename(columns={'field':'CSV字段名'}),hide_index=True,use_container_width=True)
        st.info('目前支持已计算特征输入。较长间隔的时长阈值、个人历史相对指标的归一化公式仍需与原始生成代码核对；请直接使用同一流程已生成的数值。本页面不从简化输入或单个学生事件推算这些字段。')
    with st.expander('模型与分类说明'):
        st.write('使用已完成训练的18变量HistGradientBoosting模型及其保存的缺失值填补器，按模型内变量顺序推理，不重新训练、不更换模型。')
        st.write(f'分类阈值：{meta["development_oof_youden_threshold"]:.16f}，来自开发队列OOF的Youden阈值。')
st.markdown('<div class="foot">学习参与度风险预测 · 教学支持与研究用途<br>预测结果供教师了解学习变化、安排沟通与支持时参考。</div>',unsafe_allow_html=True)
