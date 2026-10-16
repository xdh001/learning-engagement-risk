import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from prediction import load_bundle, predict, read_csv

st.set_page_config(page_title='学习参与度风险预测 · HistGB',page_icon='📊',layout='wide')
@st.cache_resource
def resources():
    return load_bundle()

st.title('未来7天学习参与度下降风险预测')
st.caption('HistGradientBoosting · 18项学习行为特征 · 冻结模型推理')
st.write('结局定义：以预测前14天平均每周活跃日数为基线，未来7天活跃日数下降至少50%。')
try:
    bundle, meta = resources()
except Exception as e:
    st.error('模型加载失败：'+str(e)); st.stop()
features = list(bundle['predictors'])
with st.sidebar:
    st.header('模型信息')
    st.write('实际加载：'+type(bundle['model']).__name__)
    st.write(f'输入：{len(features)}项；标准化：{bundle["scaled"]}')
    st.write('开发队列OOF Youden阈值：'+str(meta['development_oof_youden_threshold']))
    st.caption('模型SHA256：'+meta['model_sha256'])
    st.write('研究报告ROC-AUC：开发0.7220；时间验证0.7039；KU外部0.7100。')
st.info('请使用原实验特征工程生成的18项特征。单个学生也须提供历史及同课程同期同伴相对特征。本页面不从简单活跃日数推算复杂特征。')
st.warning('部分特征的精确归一化公式与长间隔阈值尚需原始生成代码核对；请勿仅根据字段中文名手工构造这些值。缺失必要特征时不生成预测。')
with st.expander('18项字段说明与输入要求'):
    st.dataframe(pd.DataFrame(meta['feature_dictionary']),hide_index=True,use_container_width=True)
    st.caption('字段顺序直接来自模型包。long_gap_count_28d、B_self_relative_7d_vs_29to42d的精确定义待原始特征生成代码确认；当前仅接收冻结流程已计算的值。')
    st.download_button('下载18变量CSV空模板',pd.DataFrame(columns=['student_id',*features]).to_csv(index=False).encode('utf-8-sig'),'HistGB_18_template.csv','text/csv')
mode = st.radio('输入方式',['CSV批量预测','单个学生完整特征'],horizontal=True)
frame = None
if mode == 'CSV批量预测':
    uploaded = st.file_uploader('上传UTF-8 CSV（学生编号student_id + 全部18项特征）',type=['csv'])
    if uploaded:
        try:
            frame = read_csv(uploaded.getvalue())
            st.dataframe(frame.head(20),hide_index=True)
        except Exception as e: st.error('CSV读取失败：'+str(e))
else:
    st.caption('粘贴单行CSV，须含表头、student_id和全部18项特征；不提供任何默认特征值。')
    text = st.text_area('单个学生CSV',height=180,placeholder=','.join(['student_id',*features])+'\n')
    if text.strip():
        try:
            frame = read_csv(text.encode())
            if len(frame)!=1:
                st.error('单个学生模式仅允许一行数据。'); frame=None
        except Exception as e: st.error('CSV读取失败：'+str(e))
confirmed = st.checkbox('确认所有特征由与训练一致的流程生成，历史和同伴特征均有真实数据依据。')
if st.button('计算预测风险',type='primary',disabled=frame is None or not confirmed):
    try:
        result = predict(frame,bundle,meta)
        st.success(f'已完成{len(result)}个预测时点。')
        if len(result)==1:
            st.metric('未来7天学习参与度下降概率',f'{result.iloc[0].decline_probability_7d:.2%}')
            st.write('风险分类：'+result.iloc[0].risk_class)
            st.write(result.iloc[0].risk_message)
        st.dataframe(result,hide_index=True,use_container_width=True)
        st.download_button('下载预测结果CSV',result.to_csv(index=False).encode('utf-8-sig'),'HistGB_predictions.csv','text/csv')
    except Exception as e: st.error('未生成预测：'+str(e))
st.caption('风险分类用于辅助教师核查并提供学习支持；预测结果不等同于实际结局。当前版本未启用SHAP解释。')
