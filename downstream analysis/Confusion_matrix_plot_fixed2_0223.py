import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# 导入训练所需的库（与原MLP结构保持一致）
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------- 1. 核心配置（集中在开头，方便修改） --------------------------
# 原始数据集路径
DATA_PATH = "UNAGI/data/haniffa_covid/training_set_status/all_emb_covid_status.csv"
# 图片保存文件夹路径
SAVE_DIR = "UNAGI/data/haniffa_covid/plot_for_figure/Confusion_Matrix"
# 图片保存文件名
SAVE_FILENAME = "cell_type_confusion_matrix_fixed2.png"
# 关键列名配置
FEATURE_COLS_PREFIX = "latent_dim_"  # 特征列前缀
LABEL_COL = "cell_type"  # 真实标签列名
# 训练配置（与原MLP保持一致）
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 150
BATCH_SIZE = 32
# 新增：早停机制配置
PATIENCE = 25  # 验证集性能连续多少轮不提升就停止训练
MONITOR = "val_loss"  # 监控指标（val_loss/val_accuracy）
MIN_DELTA = 0.0001  # 最小改进值，小于此值视为无改进
RESTORE_BEST_WEIGHTS = True  # 训练停止后，恢复到性能最好的权重
# 可视化配置
FIGSIZE = (16, 14)  # 画布尺寸（适配多类别）
DPI = 800  # 保存分辨率
CMAP = "Oranges"  # 配色方案（可改为Reds/Greens等）

# -------------------------- 新增：比例验证函数 --------------------------
def verify_class_distribution(df, y_train, y_test, class_names, class_mapping):
    """验证训练集/测试集的类别分布是否与原数据集一致"""
    print("\n===== 类别分布验证（按比例抽样检查） =====")
    
    # 原数据集分布
    original_counts = df[LABEL_COL].value_counts().sort_index()
    original_total = len(df)
    
    # 训练集分布
    train_counts = pd.Series(y_train).value_counts().sort_index()
    train_total = len(y_train)
    
    # 测试集分布
    test_counts = pd.Series(y_test).value_counts().sort_index()
    test_total = len(y_test)
    
    # 打印每个类别的详细比例
    print(f"{'类别名称':<20} {'原数据集数量':<10} {'原占比':<8} {'训练集数量':<10} {'训练占比':<8} {'测试集数量':<10} {'测试占比':<8}")
    print("-" * 90)
    
    for class_name in class_names:
        class_idx = class_mapping[class_name]
        
        # 原数据集
        orig_cnt = original_counts.get(class_name, 0)
        orig_pct = orig_cnt / original_total * 100
        
        # 训练集
        train_cnt = train_counts.get(class_idx, 0)
        train_pct = train_cnt / train_total * 100 if train_total > 0 else 0
        
        # 测试集
        test_cnt = test_counts.get(class_idx, 0)
        test_pct = test_cnt / test_total * 100 if test_total > 0 else 0
        
        print(f"{class_name:<20} {orig_cnt:<10} {orig_pct:.2f}% {'':<0} {train_cnt:<10} {train_pct:.2f}% {'':<0} {test_cnt:<10} {test_pct:.2f}%")
    
    # 汇总验证
    print("-" * 90)
    print(f"{'总计':<20} {original_total:<10} 100.00% {'':<0} {train_total:<10} 100.00% {'':<0} {test_total:<10} 100.00%")
    print("===== 分布验证完成 =====")

# -------------------------- 2. 加载数据（强化分层抽样+比例验证） --------------------------
def load_data_auto_mapping(data_path, feature_prefix, label_col):
    """加载数据并自动生成 字符串标签→数字 的映射，强化分层抽样"""
    # 读取数据集
    df = pd.read_csv(data_path)
    
    # 移除cell_type为NaN的行（与原训练逻辑一致）
    initial_count = df.shape[0]
    df = df.dropna(subset=[label_col])
    if df.shape[0] < initial_count:
        print(f"警告：已移除 {initial_count - df.shape[0]} 个{label_col}为NaN的样本")
    
    # 提取特征列
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    X = df[feature_cols].values
    
    # 自动获取所有唯一的字符串标签，并按字母顺序生成映射
    unique_labels = sorted(df[label_col].unique())
    class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    class_names = unique_labels  # 类别名称列表
    
    # 自动将字符串标签转为数字
    y = df[label_col].map(class_mapping).values
    
    # 强化分层抽样逻辑：兼容单样本类别
    # 方法：将单样本类别全部放入训练集，其余类别正常分层
    stratify_flag = None
    class_counts = pd.Series(y).value_counts()
    single_sample_classes = class_counts[class_counts == 1].index.tolist()
    multi_sample_classes = class_counts[class_counts > 1].index.tolist()
    
    if len(single_sample_classes) > 0:
        print(f"\n注意：发现 {len(single_sample_classes)} 个类别仅含1个样本，将全部放入训练集：")
        for cls_idx in single_sample_classes:
            cls_name = [k for k, v in class_mapping.items() if v == cls_idx][0]
            print(f"  - {cls_name} (类别索引{cls_idx})")
        
        # 分离单样本和多样本数据
        single_mask = np.isin(y, single_sample_classes)
        multi_mask = ~single_mask
        
        X_single = X[single_mask]
        y_single = y[single_mask]
        X_multi = X[multi_mask]
        y_multi = y[multi_mask]
        
        # 对多样本数据进行分层抽样
        X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
            X_multi, y_multi, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_multi
        )
        
        # 合并：单样本全部加入训练集
        X_train = np.vstack([X_multi_train, X_single]) if len(X_single) > 0 else X_multi_train
        y_train = np.hstack([y_multi_train, y_single]) if len(y_single) > 0 else y_multi_train
        X_test = X_multi_test
        y_test = y_multi_test
        
    else:
        # 所有类别都有多个样本，直接分层抽样
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    
    # 打印基础信息
    print(f"\n数据加载完成：")
    print(f"  - 总样本数：{X.shape[0]}，训练集：{X_train.shape[0]}，测试集：{X_test.shape[0]}")
    print(f"  - 特征维度：{X.shape[1]}，类别数：{len(class_names)}")
    print("自动生成的标签映射关系：")
    for label, idx in class_mapping.items():
        print(f"  {label} → {idx}")
    
    # 新增：验证类别分布比例
    verify_class_distribution(df, y_train, y_test, class_names, class_mapping)
    
    return X_train, X_test, y_train, y_test, class_names

# -------------------------- 3. 创建MLP模型（与原结构完全一致） --------------------------
def create_mlp_model(input_dim, num_classes):
    """创建与原代码完全一致的MLP模型"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# -------------------------- 4. 训练模型并预测（新增早停机制） --------------------------
def train_model_and_predict(X_train, X_test, y_train, y_test, class_names):
    """训练MLP模型并返回测试集预测结果（集成早停）"""
    num_classes = len(class_names)
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = create_mlp_model(input_dim, num_classes)
    
    # 新增：创建早停回调实例
    early_stopping = EarlyStopping(
        monitor=MONITOR,          # 监控验证集损失
        patience=PATIENCE,        # 容忍多少轮无改进
        min_delta=MIN_DELTA,      # 最小改进阈值
        restore_best_weights=RESTORE_BEST_WEIGHTS,  # 恢复最优权重
        verbose=1                 # 打印停止信息
    )
    
    # 训练模型（添加早停回调）
    print(f"\n开始训练MLP模型（启用早停机制）：")
    print(f"  - 训练轮数：{EPOCHS}（最多），早停耐心值：{PATIENCE}")
    print(f"  - 监控指标：{MONITOR}，最小改进值：{MIN_DELTA}")
    print(f"  - 批次大小：{BATCH_SIZE}")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]  # 新增：传入早停回调
    )
    
    # 在测试集上预测
    print("\n开始在测试集上预测...")
    y_score = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_score, axis=1)
    
    # 计算并打印准确率（验证训练效果）
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    # 新增：打印实际训练轮数
    actual_epochs = len(history.history['loss'])
    print(f"实际训练轮数：{actual_epochs}（早停触发前）")
    
    return y_pred

# -------------------------- 5. 绘制并保存混淆矩阵 --------------------------
def plot_beautiful_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制美观的混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 归一化（显示百分比，更易对比）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建画布
    plt.figure(figsize=FIGSIZE)
    
    # 绘制混淆矩阵（热力图）
    ax = sns.heatmap(
        cm_normalized,  # 归一化后的矩阵
        annot=True,  # 显示数值
        fmt='.2f',  # 数值格式（保留2位小数）
        cmap=CMAP,  # 配色
        cbar=True,  # 显示颜色条
        cbar_kws={'shrink': 0.8, 'label': 'Normalized Value', 'alpha': 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,  # 格子边框宽度
        annot_kws={
            'size': 10,  # 标注文字大小
            'weight': 'bold'  # 标注文字加粗
        }
    )
    
    # 美化配置
    # 标题
    plt.title('Confusion Matrix of Cell Type Classifier', 
              fontsize=26, fontweight='bold', pad=30)
    # 坐标轴标签
    plt.xlabel('Predicted Label', fontsize=22, fontweight='bold', labelpad=20)
    plt.ylabel('True Label', fontsize=22, fontweight='bold', labelpad=20)
    
    # X轴刻度：设置字号+加粗+旋转
    xtick_labels = ax.get_xticklabels()
    for label in xtick_labels:
        label.set_fontsize(14)
        label.set_fontweight('bold')
        label.set_rotation(45)
        label.set_ha('right')
    ax.set_xticklabels(xtick_labels)
    
    # Y轴刻度：设置字号+加粗
    ytick_labels = ax.get_yticklabels()
    for label in ytick_labels:
        label.set_fontsize(14)
        label.set_fontweight('bold')
        label.set_rotation(0)
    ax.set_yticklabels(ytick_labels)
    
    # 颜色条美化
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), 
                       fontsize=16, fontweight='bold', rotation=-90, va="bottom")
    cbar.ax.tick_params(labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor='white')
    print(f"\n混淆矩阵已保存至：{save_path}")
    plt.show()

# -------------------------- 主流程执行 --------------------------
if __name__ == "__main__":
    # 1. 加载数据
    X_train, X_test, y_train, y_test, class_names = load_data_auto_mapping(
        DATA_PATH, FEATURE_COLS_PREFIX, LABEL_COL
    )
    
    # 2. 训练模型并获取测试集预测结果
    y_pred = train_model_and_predict(X_train, X_test, y_train, y_test, class_names)
    
    # 3. 绘制并保存混淆矩阵
    save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)
    plot_beautiful_confusion_matrix(y_test, y_pred, class_names, save_path)
    
    print("\n所有流程执行完成：已重新训练模型并生成混淆矩阵，未保存任何模型文件")