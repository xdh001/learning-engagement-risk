# 学习参与度下降风险预测（HistGB / 18变量）

本项目使用用户提供的冻结 `HistGB_fullDevelopment.pkl`，未重新训练、调参或筛选变量。模型为 `HistGradientBoostingClassifier`，输入顺序直接读取模型包中的 `predictors`。通过保存的 `imputer` 变换，再由 `predict_proba` 计算正类概率；`scaled=False`，不使用包内的scaler。

结局：以预测前14天平均每周活跃日数为基线，未来7天活跃日数下降至少50%。

## 已实现

- 中文界面；单个学生完整特征CSV和批量CSV。
- 学生编号、真实模型概率、风险分类、学习支持提示、结果CSV下载。
- 精确18变量顺序、模型SHA256、填补器特征及模型类别核对。
- 缺失字段、空值、无同伴特征、非数值、无穷值和明显范围错误时阻止预测。
- 无默认学生特征、模拟预测、旧算法回退或旧解释图。

阈值为 **0.3094141475972338**，从归档实验包 `SENSITIVITY_HistGB_FULL18_REDUCED13_FINAL.zip` 中 `Supplementary_Table_S4_HistGB_Full18_vs_Reduced13.csv` 的Development/Full18行读取。原表副本位于 `model/threshold_source.csv`。对应AUC为0.7220159120 / 0.7039030448 / 0.7099506806。

## 输入

每行是一个学生在一个预测时点的数据。必填 `student_id` 和模型包中的全部18项特征；可附带 `domain_id`、`landmark_day`。学生编号以字符串读取。下载空模板或查看 `model/feature_template.csv`；列名必须精确匹配，列顺序可任意，程序按模型顺序重排。

完整字段字典位于 `model/metadata.json`，并在页面展示。字典依据：`Y50_21候选特征描述_18变量建模_已补齐18项统计_Table1_S1_S2.xlsx`。

**当前是严格预计算特征输入模式。** 不能仅用一个学生的少量活跃日数生成全部特征。`C_peer_relative_7d_LOO`须来自同课程同期的真实留一同伴数据；`B_self_relative_7d_vs_29to42d`需要更早历史数据。`long_gap_count_28d`的具体长间隔阈值及B变量的归一化公式，在现有字典中仍标注待核对，不能自行推定。请仅导入原实验流程生成的数据。未实现原始事件级自动特征构建，也未启用SHAP。

训练时保存的中位数填补器被原样复用。交互界面对必要特征缺失采取更严格策略：不允许用缺失值代替无法构造的历史/同伴信息，因此拒绝含空值的数据。训练/离线验证数据中的原生缺失值仍可在独立原模型审计中按原流程填补。

## 本地运行

验证环境为Python **3.12**；scikit-learn **1.4.2**（由pickle内保存版本确认）。NumPy/joblib原训练确切版本未包含在模型包中，requirements锁定的是实际成功测试的兼容版本。不要使用更高版本sklearn直接加载pickle。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Windows可使用 `.venv\Scripts\python -m streamlit run app.py`。

## 验证

```bash
.venv/bin/pip install pytest==8.3.5
.venv/bin/python -m pytest tests -q
```

完整结果见 `VALIDATION.md`。真实学生数据不纳入公开仓库或交付包。

## 部署目标

现有应用：https://learning-engagement-risk-xdh001.streamlit.app/

部署仓库：xdh001/learning-engagement-risk；分支main；入口app.py。

2026-10-16通过xdh001网页会话上传独立HistGB程序，保留原网址。实际云端运行结果以最新部署验证记录为准。
