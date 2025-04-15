import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime


# 1. 数据加载与预处理
def load_data(path):
    """加载并预处理数据"""
    df = pd.read_csv(path)

    # 数据质量检查
    assert 'status' in df.columns, "数据必须包含status列"
    print(f"数据分布:\n{df['status'].value_counts()}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")

    # 识别数值列和非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    print(f"\n数值列: {list(numeric_cols)}")
    print(f"非数值列: {list(non_numeric_cols)}")

    # 仅对数值列填充中位数
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


# 2. 特征工程
def feature_engineering(df):
    """特征处理"""
    # 分离特征和标签
    features = df.drop(['status', 'name'], axis=1)  # 移除无关列
    labels = df['status'].values

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels, scaler


# 3. 模型优化
def optimize_model(X_train, y_train):
    """使用网格搜索优化XGBoost"""

    # 改进的参数网格（基于文献和实验）
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],  # 更精细的学习率
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],  # 添加gamma参数控制分裂
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0.1, 1],
        'n_estimators': [100, 200]
    }

    # 改进的交叉验证策略（分层K折）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 使用早停的基准模型
    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=10,
        random_state=39
    )

    # 网格搜索配置
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',  # 使用AUC作为评估指标
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# 4. 评估与可视化
def evaluate_model(model, X_test, y_test):
    """模型评估与结果可视化"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    print(f"\nAUC分数: {roc_auc_score(y_test, y_proba):.4f}")

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.show()

    # 特征重要性
    plt.figure(figsize=(10, 6))
    feat_imp = pd.Series(model.feature_importances_,
                         index=df.drop(['status', 'name'], axis=1).columns)
    feat_imp.nlargest(15).plot(kind='barh')
    plt.title('Top 15特征重要性')
    plt.show()


# 主流程
if __name__ == "__main__":
    # 数据加载
    df = load_data('data/parkinsons.csv')

    # 特征工程
    X, y, scaler = feature_engineering(df)

    # 数据分割（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=39
    )

    # 模型优化
    print("\n开始参数优化...")
    best_model, best_params = optimize_model(X_train, y_train)
    print(f"\n最佳参数: {best_params}")

    # 模型评估
    evaluate_model(best_model, X_test, y_test)

    # 模型保存（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"parkinson_model_v{timestamp}.pkl"
    joblib.dump({'model': best_model, 'scaler': scaler}, model_path)
    print(f"\n模型已保存到: {model_path}")