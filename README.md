# 🛒 电商客户价值分层与行为预测系统 (AI-Powered Customer Segmentation)

本项目是一套完整的端到端机器学习流程，旨在通过 **RFM 模型** 对电商客户进行精准画像，并结合 **KMeans 聚类** 与 **随机森林 (Random Forest)** 实现自动化的客户分类与实时价值预测。

## 🌟 项目核心亮点
- **闭环建模**：实现了从“无监督学习 (无标签标注)”到“有监督学习 (分类预测)”的工业级算法闭环。
- **极致精度**：随机森林分类器在验证集上达到了 **99% 的 F1-Score**，证明了分群逻辑的极高一致性。
- **深度业务洞察**：通过 **Feature Importance (特征重要性)** 分析发现 **Frequency (消费频率)** 是衡量用户忠诚度的核心指标。

## 📊 算法架构与流程
1. **数据清洗 (Data Cleaning)**：处理编码异常、缺失值，并剔除退货及非活跃用户记录。
2. **特征工程 (Feature Engineering)**：
   - 构建 **RFM** 矩阵 (Recency, Frequency, Monetary)。
   - **Log Transformation**：消除长尾分布，处理偏态数据。
   - **StandardScaler**：量纲统一化处理。
3. **聚类分析 (Clustering)**：
   - 使用 **Elbow Method (肘部法)** 确定 $K=2$ 或 $K=3$ 为最佳聚类数。
   - 利用 **PCA (主成分分析)** 降维实现 2D 可视化。
4. **分类预测 (Classification)**：
   - 以 KMeans 标签为目标，训练 **Random Forest Classifier**。
   - 模型持久化：通过 `joblib` 固化模型，支持实时推理接口。

## 📈 可视化成果
- **3D 可视化**：直观展示用户在 R、F、M 空间的聚集分布。
- **蛇形图 (Snake Plot)**：清晰对比不同客户群体在标准化指标上的行为差异。

## 🚀 快速启动 (How to Run)
1. 安装依赖：`pip install -r requirements.txt`
2. 运行分析：`python main_analysis.py` (或运行 Notebook 记录)
3. 实时预测示例：
   ```python
   import joblib
   # 输入 R, F, M 即可获得画像分类
   model = joblib.load('models/rf_model.pkl')


# 🛒 E-commerce Customer Segmentation AI System

This repository provides a comprehensive end-to-end Machine Learning pipeline designed to segment e-commerce customers using the **RFM Model**. It integrates **Unsupervised Learning (K-Means)** for automated labeling and **Supervised Learning (Random Forest)** for real-time customer value prediction.

## 🌟 Key Project Highlights
- **ML Closed-loop Pipeline**: Successfully transitioned from discovery (Clustering) to production (Classification).
- **High Precision**: The Random Forest classifier achieved a **99% F1-Score**, demonstrating robust consistency in segment logic.
- **Data-Driven Insights**: Feature Importance analysis identified **Frequency** as the most critical metric for defining customer loyalty in this dataset.

## 📊 Technical Workflow
1. **Data Cleaning**: Handled Unicode decoding, missing values, and filtered out non-active or returned transaction records.
2. **Feature Engineering**:
   - Constructed **RFM** (Recency, Frequency, Monetary) matrix.
   - **Log Transformation**: Corrected skewed data distributions.
   - **StandardScaler**: Normalized features for distance-based algorithms.
3. **Clustering (Unsupervised)**:
   - Determined optimal $K$ using the **Elbow Method**.
   - Applied **PCA (Principal Component Analysis)** for 2D visualization and cluster separation analysis.
4. **Classification (Supervised)**:
   - Trained a **Random Forest Classifier** using K-Means clusters as ground truth.
   - **Model Persistence**: Exported trained models via `joblib` for deployment-ready inference.

## 📈 Visual Analytics
- **3D Interactive Scatter Plot**: Visualizes customer distribution in R-F-M space.
- **Snake Plot**: Compares behavioral patterns across different segments using standardized scales.
- **Feature Importance**: Ranked the impact of R, F, and M on final classification.

## 🚀 Quick Start
### Prerequisites
```bash
pip install -r requirements.txt   
