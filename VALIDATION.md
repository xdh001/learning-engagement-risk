# 部署验证记录（2026-10-15）

## 模型核对

- 用户上传模型字节原样保留，SHA256：`580362b81380f9e899ddc3f71ea0bc117fbd21d15eb38ddb3fcb0f8952e0f096`。
- 模型：HistGradientBoostingClassifier；learning_rate=0.06；max_iter=240；min_samples_leaf=40；l2_regularization=1.0；random_state=20261008。
- 真实包键：model、imputer、scaler、predictors、scaled。
- 模型n_features_in_=18，imputer.feature_names_in_与predictors顺序逐项一致；scaled=False；正类为1。
- sklearn训练版本：1.4.2；测试运行Python 3.12、NumPy 1.26.4、joblib 1.4.2、pandas 2.2.3。
- 锁定pyarrow 19.0.1以兼容NumPy 1.26.4；未固定的新pyarrow版本曾导致界面测试失败，修正后全部通过。

## 真实数据审计

从已有 `Y50_18_ALL10_TESTED_V2.zip` 读取KU原始冻结特征数据，全程未重训。

- 全部KU：38,991个时点，真实模型计算ROC-AUC **0.7099506806148939**，与归档Full18实验结果一致。
- 从前80行中取全部18特征完整的58行，应用预测路径与原模型 `model.predict_proba(imputer.transform(X[predictors]))[:,1]` 逐条比较，最大绝对误差 **0.0**。
- 这58行预测概率范围：0.13414849599993436–0.6286203197810651。
- 此一致性核对是相同冻结模型的两条代码路径对照；未获得该18变量模型独立保存的逐行历史概率，因此不宣称完成与独立历史概率文件的逐条核对。
- 全量KU离线AUC核对包含原生缺失值，经原imputer处理；网页严格输入模式拒绝必要输入缺失。

## 自动化与服务检查

`OMP_NUM_THREADS=2 python -m pytest tests -q`：**14 passed**。

覆盖真实模型加载、18特征、列顺序重排、单个/多行推理、直接预测概率一致性、冻结阈值分类、CSV读写及编号前导零、缺字段/空值/缺同伴值/非数值/无穷值/非法范围/计数/学生编号、重复CSV表头，以及Streamlit AppTest启动和单个学生实际提交预测。

技术测试输入来自填补器已保存统计量，仅用作自动化测试；不在网页伪装为真实学生样例。

独立Streamlit服务实际启动，`/_stcore/health` 返回HTTP 200、正文 `ok`。

## 云端部署与浏览器验证（2026-10-16）

已更新xdh001/learning-engagement-risk的main分支，入口app.py；保留原网址：https://learning-engagement-risk-xdh001.streamlit.app/。

提交记录：模型与真实阈值06abd9c71e28fd02b610687a88244f98153d72ee；独立应用e1e7b71480988dfb0666fdd785d872658be4f4b1。旧XGBoost模型文件已从当前版本删除，可从Git历史恢复。

云端原Python3.14导致scikit-learn1.4.2安装失败，已在现有应用设置中切换Python3.12，随后正常启动。

浏览器显示实际模型HistGradientBoostingClassifier、18特征、scaled=False，SHA256与上传原件一致。

使用两条标注为TECHNICAL_TEST的技术测试输入（来自填补器统计量，不代表真实学生），网页完成批量预测并成功下载CSV。线上概率与本地原模型最大差值5.551115123125783e-17，为CSV浮点读写精度。

单条模式预测显示27.27%，与本地同一输入一致。只提供student_id和active_days_7d时，网页明确列出缺失的17项特征，未产生预测。

当前仅支持原实验流程预计算特征输入，尚未实现原始事件自动构建及HistGB SHAP解释。
