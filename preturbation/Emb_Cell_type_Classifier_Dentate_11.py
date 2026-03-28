import scanpy as sc
import seaborn as sns
import joblib  # 统一导入joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
# 导入UMAP库
from umap import UMAP
# 导入Keras相关库用于模型保存
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# 使用scikeras替代tensorflow.keras.wrappers.scikit_learn
from scikeras.wrappers import KerasClassifier
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional  # 添加这行导入
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from tensorflow.keras.models import load_model, Sequential
import glob
import warnings
warnings.filterwarnings("ignore")

def load_embedding_data(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder, list]:
    """读取带细胞类型的嵌入文件，准备特征和标签数据"""
    try:
        print(f"正在读取嵌入数据文件：{csv_path}")
        df = pd.read_csv(csv_path, index_col=0)

        if "cell_type" not in df.columns:
            raise ValueError("CSV文件中未找到'cell_type'列，请检查输入文件")

        # 移除cell_type为NaN的行
        initial_count = df.shape[0]
        df = df.dropna(subset=["cell_type"])
        if df.shape[0] < initial_count:
            print(f"警告：已移除 {initial_count - df.shape[0]} 个cell_type为NaN的样本")

        # 提取特征和标签
        X = df.drop("cell_type", axis=1).values
        y = df["cell_type"].values

        # 标签编码
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_.tolist()

        print(f"数据加载完成，共 {X.shape[0]} 个样本，{X.shape[1]} 个特征，{len(class_names)} 种细胞类型")
        print("细胞类型分布：")
        for cls, count in zip(class_names, np.bincount(y_encoded)):
            print(f"  {cls}: {count} 个样本")

        return X, y_encoded, df, le, class_names

    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        raise
    except Exception as e:
        print(f"加载数据时发生错误 - {e}")
        raise

def create_mlp_model(input_dim, num_classes):
    """创建多层感知器模型，用于保存为H5格式"""
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

def train_cell_type_classifier(X: np.ndarray, y: np.ndarray, class_names: list,
                               test_size: float = 0.2, random_state: int = 42) -> tuple[KerasClassifier, dict]:
    """训练细胞类型分类器，划分训练集和测试集"""
    print("\n开始划分训练集和测试集...")
    # 处理极端情况：若某类细胞样本数=1，分层采样会报错
    unique_classes = np.unique(y)
    stratify_flag = y if all(np.bincount(y) >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_flag
    )

    print(f"训练集: {X_train.shape[0]} 个样本，测试集: {X_test.shape[0]} 个样本")

    # 创建并训练MLP模型（用于保存为H5格式）
    print("开始训练神经网络分类器...")
    num_classes = len(np.unique(y))

    # 使用scikeras的KerasClassifier
    model = KerasClassifier(
        model=lambda: create_mlp_model(X_train.shape[1], num_classes),
        epochs=150,
        batch_size=32,
        verbose=1
    )

    model.fit(X_train, y_train)

    # 预测并评估
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算评估指标
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 生成分类报告
    class_report = classification_report(
        y_test, y_pred_test, target_names=class_names, zero_division=0
    )

    # 整理结果
    results = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "classification_report": class_report
    }

    return model, results

def visualize_classification_results(model, results, le: LabelEncoder, save_dir: str = "visualizations"):
    """可视化分类器的分类效果，使用UMAP"""
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建可视化结果目录：{save_dir}")

    # 1. 混淆矩阵可视化
    print("\n绘制混淆矩阵...")
    cm = confusion_matrix(results["y_test"], results["y_pred_test"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    )

    plt.figure(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("confusion_matrix of test")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"混淆矩阵已保存至：{cm_path}")

    # 2. 分类报告打印
    print("\n测试集分类报告：")
    print(results["classification_report"])

    # 3. UMAP降维可视化
    print("进行UMAP降维可视化...")
    umap = UMAP(n_components=2, random_state=42)
    X_test_umap = umap.fit_transform(results["X_test"])

    plt.figure(figsize=(12, 10))

    # 绘制真实标签
    sns.scatterplot(
        x=X_test_umap[:, 0],
        y=X_test_umap[:, 1],
        hue=le.inverse_transform(results["y_test"]),
        palette="tab10",
        alpha=0.7,
        s=50,
        marker="o"  # 真实标签用圆形
    )

    # 绘制预测标签
    sns.scatterplot(
        x=X_test_umap[:, 0],
        y=X_test_umap[:, 1],
        hue=le.inverse_transform(results["y_pred_test"]),
        style=le.inverse_transform(results["y_pred_test"]),
        palette="tab10",
        alpha=0.5,
        s=100,
        marker="X"  # 预测标签用叉形
    )

    # 手动添加图例说明
    plt.legend(
        title='Celltype and label',
        labels=[f'{cls} (true)' for cls in le.classes_] + [f'{cls} (perdict)' for cls in le.classes_],
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8
    )

    plt.title(f"UMAP(accuracy: {results['test_acc']:.4f})")
    plt.xlabel("UMAP dim1")
    plt.ylabel("UMAP dim2")
    plt.tight_layout()
    umap_path = os.path.join(save_dir, "umap_visualization.png")
    plt.savefig(umap_path, dpi=300)
    print(f"UMAP可视化结果已保存至：{umap_path}")

def save_classifier_model(model, le: LabelEncoder, model_dir: str = "Classfi_models",
                          model_name: str = "cell_type_classifier.h5"):
    """保存训练好的分类器模型为H5文件和标签编码器"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"已创建模型保存目录：{model_dir}")

    # 保存Keras模型为H5格式
    model_path = os.path.join(model_dir, model_name)
    model.model_.save(model_path)  # scikeras使用model_属性访问内部模型
    print(f"分类器模型已保存至：{model_path}")

    # 保存标签编码器
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    joblib.dump(le, le_path)
    print(f"标签编码器已保存至：{le_path}")

def add_celltype_to_embedding(csv_path: str,
                              h5ad_path: str,
                              output_csv_name: str = "embeddings_with_celltype.csv") -> None:
    """为嵌入文件添加细胞类型信息"""
    try:
        print(f"正在读取CSV文件：{csv_path}")
        emb_df = pd.read_csv(csv_path, index_col=0)
        print("CSV文件读取完成")

        print(f"正在读取h5ad文件：{h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        adata.obs.index = adata.obs.index.astype(str)
        print("h5ad文件读取完成")

    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误 - {e}")
        return

    # 匹配细胞类型
    print("正在匹配细胞类型...")
    cluster_col = None
    for candidate in ["clusters", "leiden", "louvain","celltype","clusters_fig6_broad_final"]:
        if candidate in adata.obs.columns:
            cluster_col = candidate
            break
    if cluster_col is None:
        raise ValueError("h5ad文件未找到cluster列（如'clusters'、'leiden'）")

    emb_df["cell_type"] = emb_df.index.map(adata.obs[cluster_col])
    unmatched_count = emb_df["cell_type"].isna().sum()
    if unmatched_count > 0:
        print(f"警告：{unmatched_count} 个细胞未找到对应的cluster")

    # 保存结果
    output_dir = "training_set"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_csv_name)
    emb_df.to_csv(output_path, index=True, encoding="utf-8")
    print(f"结果已保存至：{output_path}")

def training_Classifier(embedding_csv_path: str):
    """主函数：串联所有训练分类器并可视化流程"""
    # 1. 加载数据
    X, y, df, le, class_names = load_embedding_data(embedding_csv_path)

    # 2. 训练分类器
    model, results = train_cell_type_classifier(
        X=X, y=y, class_names=class_names, test_size=0.2, random_state=42
    )

    # 3. 可视化分类结果
    visualize_classification_results(model, results, le)

    # 4. 保存模型
    save_classifier_model(model, le)

    print("\n所有流程执行完成！")

def load_cell_latent_csv(csv_path: str, index_col: str = "index",
                         feature_prefix: str = "latent_dim_") -> tuple[np.ndarray, List[str], pd.Series]:
    """
    从CSV文件中读取细胞潜在向量数据，提取核心信息

    参数：
        csv_path: CSV文件路径（需包含索引列、64个latent_dim特征列、cell_type列）
        index_col: 索引列名称（如cell_id、cell_index，默认index）
        feature_prefix: 特征列前缀（默认latent_dim_，匹配latent_dim_0~latent_dim_63）

    返回：
        cell_latents: 64维细胞潜在向量（shape: [细胞数量, 64]）
        feature_names: 64个特征列的名称列表（latent_dim_0~latent_dim_63）
        cell_types: 细胞类型列（Series，索引与细胞潜在向量对应）
    """
    # 1. 读取CSV文件
    try:
        df = pd.read_csv(csv_path, index_col=index_col)
        print(f"成功读取CSV文件：{os.path.abspath(csv_path)}")
        print(f"CSV文件结构：{df.shape[0]}个细胞，{df.shape[1]}列（含特征列+cell_type列）")
    except Exception as e:
        raise FileNotFoundError(f"读取CSV失败：{str(e)}")

    # 2. 验证cell_type列是否存在
    if "cell_type" not in df.columns:
        raise ValueError(f"CSV文件缺少必需的'cell_type'列，请检查文件结构")

    # 3. 提取64个特征列（匹配latent_dim_0~latent_dim_63）
    feature_names = [col for col in df.columns if col.startswith(feature_prefix)]
    # 按特征编号排序（确保latent_dim_0在前，latent_dim_63在后）
    feature_names.sort(key=lambda x: int(x.split(feature_prefix)[-1]))

    # 验证特征列数量是否为64
    if len(feature_names) != 64:
        raise ValueError(f"特征列数量为{len(feature_names)}，需严格为64列（latent_dim_0~latent_dim_63）")

    # 4. 提取核心数据
    cell_latents = df[feature_names].values  # 64维潜在向量（纯数值）
    cell_types = df["cell_type"]  # 细胞类型列

    print(f"提取完成：{cell_latents.shape[0]}个细胞，64维特征（{feature_names[0]}~{feature_names[-1]}）")
    return cell_latents, feature_names, cell_types

def perturb_all_point(
        csv_path: str,  # 输入：细胞潜在向量CSV文件路径
        model: Sequential,  # 预测模型（输入64维，输出各细胞类型概率）
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射（如{0:"T细胞", 1:"B细胞"}）
        scaler=None,  # 数据标准化器（若模型训练时使用了标准化，需传入）
        index_col: str = "index",  # CSV中的索引列名称
        save_dir: str = "./perturbation_results"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取CSV文件，对64维细胞潜在向量的每个特征施加20种强度扰动（0.5×~10×），记录各细胞类型预测占比

    参数：
        csv_path: 细胞潜在向量CSV文件路径（含索引列、64特征列、cell_type列）
        model: 用于预测细胞类型的Keras模型
        cell_type_mapping: 细胞类型ID与名称的映射字典（如{0:"T细胞", 1:"B细胞"}）
        scaler: 数据标准化器（与模型训练时使用的一致，无标准化则传None）
        index_col: CSV中的索引列名称（默认index）

    返回：
        perturb_result_df: 包含所有扰动信息（特征、强度）和各细胞类型占比的DataFrame
        cell_types: 从CSV中提取的细胞类型列（用于后续分析）
    """
    # 1. 从CSV读取数据（调用上面的加载函数）
    cell_latents, feature_names, cell_types = load_cell_latent_csv(
        csv_path=csv_path,
        index_col=index_col
    )
    n_cells = cell_latents.shape[0]  # 细胞数量
    all_cell_types = sorted(cell_type_mapping.items(), key=lambda x: x[0])  # 排序后的细胞类型
    cell_type_ids = [cid for cid, _ in all_cell_types]
    cell_type_names = [cname for _, cname in all_cell_types]

    # 2. 初始化存储结果的数据结构
    perturbed_records = []

    print(f"\n扰动初始化完成：")
    print(f"  - 细胞数量：{n_cells}，特征维度：64（{feature_names[0]}~{feature_names[-1]}）")
    print(f"  - 细胞类型列表：{cell_type_names}（共{len(cell_type_names)}种）")
    print(f"  - 扰动计划：每个特征×20种强度（0.5× → 1.0× → ... → 10.0×）")

    # 3. 遍历每个特征+20种强度，生成扰动向量并预测
    print("\n开始生成扰动向量并预测...")
    for feat_idx in range(len(feature_names)):  # 遍历64个特征
        feat_name = feature_names[feat_idx]

        # 遍历20种扰动强度（1→20对应0.5×→10×，步长0.5）
        for strength_step in range(1, 21):
            strength_multiplier = strength_step * 0.5  # 当前扰动强度

            # --------------------------
            # 核心：生成扰动后的潜在向量（仅修改当前特征）
            # --------------------------
            perturbed_vecs = np.copy(cell_latents)
            perturbed_vecs[:, feat_idx] *= strength_multiplier  # 扰动当前特征，其他特征不变

            # --------------------------
            # 模型预测（匹配训练时的数据格式）
            # --------------------------
            if scaler is not None:
                # 标准化：转为DataFrame（匹配scaler的特征名）
                perturbed_df = pd.DataFrame(perturbed_vecs, columns=feature_names)
                perturbed_scaled = scaler.transform(perturbed_df)
            else:
                perturbed_scaled = perturbed_vecs

            # 预测概率→取最大概率对应的细胞类型ID
            pred_proba = model.predict(perturbed_scaled, verbose=0)  # [n_cells, n_cell_types]
            pred_cell_type_ids = np.argmax(pred_proba, axis=1)  # 每个细胞的预测类型

            # --------------------------
            # 统计各细胞类型占比
            # --------------------------
            cell_type_count = {cid: 0 for cid in cell_type_ids}
            for cid in pred_cell_type_ids:
                if cid in cell_type_count:
                    cell_type_count[cid] += 1

            # 构建单条扰动记录
            record = {
                "feature_index": feat_idx,  # 特征索引（0~63）
                "feature_name": feat_name,  # 特征名称（latent_dim_xx）
                "strength_multiplier": strength_multiplier,  # 扰动强度（0.5~10.0）
                "total_cells": n_cells,  # 总细胞数
                "original_cell_type_dist": cell_types.value_counts().to_dict()  # 原始细胞类型分布
            }

            # 添加各细胞类型的计数与占比
            for cid, cname in all_cell_types:
                count = cell_type_count[cid]
                ratio = count / n_cells if n_cells > 0 else 0.0
                record[f"{cname}_count"] = count  # 细胞类型计数
                record[f"{cname}_ratio"] = round(ratio, 4)  # 细胞类型占比（保留4位小数）

            perturbed_records.append(record)

        # 打印进度（每处理10个特征提示一次）
        if (feat_idx + 1) % 10 == 0 or (feat_idx + 1) == len(feature_names):
            print(f"已处理 {feat_idx + 1}/64 个特征（当前特征：{feat_name}）")

    # 4. 整理结果为DataFrame并保存
    perturb_result_df = pd.DataFrame(perturbed_records)
    save_filename = "cell_latents_perturbation_results.csv"
    
    # 确保保存目录存在（不存在则创建）
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    
    # 保存CSV
    perturb_result_df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\n所有扰动结果已保存至：{os.path.abspath(save_path)}")

    return perturb_result_df, cell_types

def run_perturbation(
        csv_path: str,  # 输入：细胞潜在向量CSV文件路径
        model_path: str,  # 细胞类型分类器模型路径（.h5格式）
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        scaler=None,  # 数据标准化器
        index_col: str = "index",  # CSV中的索引列名称
        target_feature: Optional[int] = None, # 要可视化的特征索引（默认0号特征）
        save_dir_path: str = "",  # 新增：保存扰动趋势图的目录路径
        cell_type_colors: Optional[Dict[str, str]] = None
) -> None:
    """
    端到端运行CSV文件的细胞潜在向量扰动流程：读取CSV→生成扰动结果→绘制细胞类型趋势图

    参数：
        target_feature: 要可视化的特征索引（0~63），默认0号特征
    """
    # 1. 加载分类器模型
    print("=" * 60)
    print("1. 加载细胞类型分类器模型")
    print("=" * 60)
    try:
        model = load_model(model_path)
        print(f"成功加载模型：{os.path.abspath(model_path)}")
    except Exception as e:
        raise FileNotFoundError(f"模型加载失败：{str(e)}")

    # 2. 执行扰动分析（读取CSV+扰动+预测）
    print("\n" + "=" * 60)
    print("2. 读取CSV并执行细胞潜在向量扰动")
    print("=" * 60)
    perturb_result_df, cell_types = perturb_all_point(
        csv_path=csv_path,
        model=model,
        cell_type_mapping=cell_type_mapping,
        scaler=scaler,
        index_col=index_col,
        save_dir=save_dir_path
    )

    # 3. 确定可视化的目标特征
    print("\n" + "=" * 60)
    print("3. 准备可视化扰动趋势图")
    print("=" * 60)
    for target_feature in range(0,64):
        # 验证目标特征索引有效性
        if target_feature is None:
            target_feature = 0
            print(f"未指定目标特征，默认可视化第 {target_feature} 号特征")
        else:
            if not (0 <= target_feature < 64):
                raise ValueError(f"目标特征索引需在0~63之间（当前输入：{target_feature}）")

        # 获取目标特征名称
        target_feat_name = perturb_result_df[
            perturb_result_df["feature_index"] == target_feature
            ]["feature_name"].iloc[0]
        print(f"可视化目标：特征{target_feature}（{target_feat_name}）的扰动趋势")
        print(f"细胞类型分布参考：{cell_types.value_counts().to_dict()}")

        # 4. 绘制扰动趋势图
        # plot_perturbation_trends(
        #     perturb_result_df=perturb_result_df,
        #     target_feature_idx=target_feature,
        #     target_feature_name=target_feat_name,
        #     cell_type_mapping=cell_type_mapping,
        #     total_cells=cell_types.shape[0],
        #     plot_save_dir=save_dir_path
        # )

        save_perturbation_trends_changeYlim(
            perturb_result_df=perturb_result_df,
            target_feature_idx=target_feature,
            target_feature_name=target_feat_name,
            cell_type_mapping=cell_type_mapping,
            total_cells=cell_types.shape[0],
            save_dir=save_dir_path,
            cell_type_colors=cell_type_colors
        )

def plot_perturbation_trends(
        perturb_result_df: pd.DataFrame,
        target_feature_idx: int,  # 目标特征索引
        target_feature_name: str,  # 目标特征名称
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        total_cells: int,  # 总细胞数（用于图表标题）
        plot_save_dir: Optional[str] = None  # 图像保存目录（默认当前目录）
) -> None:
    """
    绘制指定特征的“扰动强度-各细胞类型占比”关系图
    """
    # 1. 筛选目标特征的扰动数据
    target_data = perturb_result_df[
        perturb_result_df["feature_index"] == target_feature_idx
        ].copy()
    if target_data.empty:
        raise ValueError(f"未找到特征{target_feature_idx}的扰动数据")

    # 按扰动强度排序（保证曲线平滑）
    target_data = target_data.sort_values("strength_multiplier").reset_index(drop=True)
    cell_type_names = [cell_type_mapping[ctid] for ctid in sorted(cell_type_mapping.keys())]

    # 2. 创建图像
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 支持中英文
    plt.figure(figsize=(14, 8))

    # 3. 为每种细胞类型绘制趋势曲线（高区分度配色）
    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_type_names)))  # 12种颜色，覆盖多种细胞类型
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'x', '+']  # 循环标记

    for i, cname in enumerate(cell_type_names):
        ratio_col = f"{cname}_ratio"
        if ratio_col not in target_data.columns:
            print(f"警告：细胞类型{cname}无占比数据，跳过绘制")
            continue

        # 绘制曲线
        plt.plot(
            target_data["strength_multiplier"],  # x轴：扰动强度（0.5~10.0）
            target_data[ratio_col],  # y轴：细胞类型占比
            label=cname,
            color=colors[i],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=7,
            alpha=0.8
        )

    # 4. 添加原始强度参考线（×1.0，红色虚线突出）
    plt.axvline(
        x=1.0,
        color='#FF4444',
        linestyle='--',
        linewidth=2,
        alpha=0.9,
        label='ori_strength(×1.0)'
    )

    # 5. 设置图表属性
    plt.title(
        f'The influence of {target_feature_idx}({target_feature_name})perturbation\n'
        f'({total_cells}cells)',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('perturbation intensity', fontsize=14, fontweight='bold')
    plt.ylabel('Cell_type distribution', fontsize=14, fontweight='bold')

    plt.xlim(0, 10.5)  # 覆盖20种强度
    plt.ylim(-0.02, 1.02)  # 占比范围0~1，留空白避免贴边

    plt.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)  # 辅助网格线
    plt.legend(
        title='Cell type',
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        shadow=True
    )

    # 6. 保存图像
    if plot_save_dir is None:
        plot_save_dir = os.getcwd()
    os.makedirs(plot_save_dir, exist_ok=True)

    # 文件名包含特征信息
    plot_filename = f"cell_type_perturb_trend_feat{target_feature_idx}_{target_feature_name}.png"
    plot_path = os.path.join(plot_save_dir, plot_filename)

    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"\n扰动趋势图已保存至：{os.path.abspath(plot_path)}")

    # 显示图像
    plt.show()

def save_perturbation_trends(
        perturb_result_df: pd.DataFrame,
        target_feature_idx: int,  # 目标特征索引
        target_feature_name: str,  # 目标特征名称
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        total_cells: int,  # 总细胞数（用于图表标题）
        save_dir: str,  # 图像保存目录（改为必填参数）
) -> None:
    """
    绘制指定特征的“扰动强度-各细胞类型占比”关系图并直接保存，不显示图窗
    """
    # 1. 筛选目标特征的扰动数据
    target_data = perturb_result_df[
        perturb_result_df["feature_index"] == target_feature_idx
        ].copy()
    if target_data.empty:
        raise ValueError(f"未找到特征{target_feature_idx}的扰动数据")

    # 按扰动强度排序（保证曲线平滑）
    target_data = target_data.sort_values("strength_multiplier").reset_index(drop=True)
    cell_type_names = [cell_type_mapping[ctid] for ctid in sorted(cell_type_mapping.keys())]

    # 2. 创建图像
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 支持中英文
    plt.figure(figsize=(14, 8))

    # 3. 为每种细胞类型绘制趋势曲线（高区分度配色）
    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_type_names)))  # 12种颜色，覆盖多种细胞类型
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'x', '+']  # 循环标记

    for i, cname in enumerate(cell_type_names):
        ratio_col = f"{cname}_ratio"
        if ratio_col not in target_data.columns:
            print(f"警告：细胞类型{cname}无占比数据，跳过绘制")
            continue

        # 绘制曲线
        plt.plot(
            target_data["strength_multiplier"],  # x轴：扰动强度（0.5~10.0）
            target_data[ratio_col],  # y轴：细胞类型占比
            label=cname,
            color=colors[i],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=7,
            alpha=0.8
        )

    # 4. 添加原始强度参考线（×1.0，红色虚线突出）
    plt.axvline(
        x=1.0,
        color='#FF4444',
        linestyle='--',
        linewidth=2,
        alpha=0.9,
        label='ori_strength(×1.0)'
    )

    # 5. 设置图表属性
    plt.title(
        f'The influence of {target_feature_idx}({target_feature_name})perturbation\n'
        f'({total_cells}cells)',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('perturbation intensity', fontsize=14, fontweight='bold')
    plt.ylabel('Cell_type distribution', fontsize=14, fontweight='bold')

    plt.xlim(0, 10.5)  # 覆盖20种强度
    plt.ylim(-0.02, 1.02)  # 占比范围0~1，留空白避免贴边

    plt.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)  # 辅助网格线
    plt.legend(
        title='Cell type',
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        shadow=True
    )

    # 6. 保存图像
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 文件名包含特征信息
    plot_filename = f"cell_type_perturb_trend_feat{target_feature_idx}_{target_feature_name}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"\n扰动趋势图已保存至：{os.path.abspath(plot_path)}")

    # 清除当前图像，释放内存
    plt.close()
     
def save_perturbation_trends_changeYlim(
        perturb_result_df: pd.DataFrame,
        target_feature_idx: int,  # 目标特征索引
        target_feature_name: str,  # 目标特征名称
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        total_cells: int,  # 总细胞数（用于图表标题）
        save_dir: str,  # 图像保存目录（改为必填参数）
        cell_type_colors: Optional[Dict[str, str]] = None  # 新增：细胞类型→颜色映射，可选
) -> None:
    """
    绘制指定特征的“扰动强度-各细胞类型占比”关系图并直接保存，不显示图窗
    y轴范围将根据数据最大值自动确定，支持指定细胞类型颜色
    """
    # 1. 筛选目标特征的扰动数据
    target_data = perturb_result_df[
        perturb_result_df["feature_index"] == target_feature_idx
        ].copy()
    if target_data.empty:
        raise ValueError(f"未找到特征{target_feature_idx}的扰动数据")

    # 按扰动强度排序（保证曲线平滑）
    target_data = target_data.sort_values("strength_multiplier").reset_index(drop=True)
    cell_type_names = [cell_type_mapping[ctid] for ctid in sorted(cell_type_mapping.keys())]

    # 2. 处理颜色参数：若未指定，使用原配色方案
    if cell_type_colors is None:
        cell_type_colors = {
            cname: plt.cm.Set3(i / len(cell_type_names)) 
            for i, cname in enumerate(cell_type_names)
        }
    else:
        # 确保所有细胞类型都有颜色（未指定的用原方案补充）
        for cname in cell_type_names:
            if cname not in cell_type_colors:
                cell_type_colors[cname] = plt.cm.Set3(list(cell_type_names).index(cname) / len(cell_type_names))

    # 3. 创建图像
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 支持中英文
    plt.figure(figsize=(14, 8))

    # 收集所有y值用于确定y轴范围
    all_y_values = []

    # 4. 为每种细胞类型绘制趋势曲线
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'x', '+']  # 循环标记
    for i, cname in enumerate(cell_type_names):
        ratio_col = f"{cname}_ratio"
        if ratio_col not in target_data.columns:
            print(f"警告：细胞类型{cname}无占比数据，跳过绘制")
            continue

        # 收集y值
        y_values = target_data[ratio_col].values
        all_y_values.extend(y_values)

        # 绘制曲线（使用指定颜色）
        plt.plot(
            target_data["strength_multiplier"],  # x轴：扰动强度（0.5~10.0）
            y_values,  # y轴：细胞类型占比
            label=cname,
            color=cell_type_colors[cname],  # 从映射中获取颜色
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=7,
            alpha=0.8
        )

    # 5. 添加原始强度参考线（×1.0，红色虚线突出）
    plt.axvline(
        x=1.0,
        color='#FF4444',
        linestyle='--',
        linewidth=2,
        alpha=0.9,
        label='ori_strength(×1.0)'
    )

    # 6. 设置图表属性
    plt.title(
        f'The influence of {target_feature_idx}({target_feature_name})perturbation\n'
        f'({total_cells}cells)',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('perturbation intensity', fontsize=14, fontweight='bold')
    plt.ylabel('Cell_type distribution', fontsize=14, fontweight='bold')

    plt.xlim(0, 10.5)  # 覆盖20种强度
    
    # 根据数据最大值动态设置y轴范围
    if all_y_values:  # 确保有数据
        max_y = max(all_y_values)
        upper_y = min(max_y * 1.1, 1.02)
        plt.ylim(-0.02, upper_y)
    else:
        plt.ylim(-0.02, 1.02)

    plt.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)  # 辅助网格线
    plt.legend(
        title='Cell type',
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        shadow=True
    )

    # 7. 保存图像
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = f"cell_type_perturb_trend_feat{target_feature_idx}_{target_feature_name}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"\n扰动趋势图已保存至：{os.path.abspath(plot_path)}")
    plt.close()

def generate_cell_type_mapping_from_h5ad(h5ad_file_path: str, cluster_col: str = "cluster") -> Dict[int, str]:
    """
    读取h5ad文件，从obs[cluster_col]提取细胞类型，生成模型ID→细胞类型名称的映射
    （默认按cluster名称排序分配ID，确保映射稳定）
    """
    # 读取h5ad文件
    try:
        adata = sc.read(h5ad_file_path)
        print(f"成功读取h5ad文件：{os.path.abspath(h5ad_file_path)}")
    except Exception as e:
        raise FileNotFoundError(f"h5ad文件读取失败：{str(e)}")

    # 验证obs中是否存在指定的cluster列
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"h5ad文件的obs中缺少'{cluster_col}'列，请检查列名")

    # 提取去重的cluster名称并排序（确保每次运行ID分配一致）
    unique_clusters = sorted(adata.obs[cluster_col].unique().tolist())

    # 生成映射：按排序后的cluster顺序分配ID（0,1,2...）
    cell_type_mapping = {idx: cluster_name for idx, cluster_name in enumerate(unique_clusters)}

    # 打印映射结果（便于验证）
    print(f"\n从h5ad自动生成的CELL_TYPE_MAPPING：")
    for id, cluster in cell_type_mapping.items():
        print(f"  ID {id}: {cluster}")

    return cell_type_mapping

def generate_cell_type_mapping_from_csv(csv_file_path: str, cell_type_col: str = "cell_type") -> Dict[int, str]:
    """
    读取CSV文件，从指定的细胞类型列提取唯一值，生成模型ID→细胞类型名称的映射
    （按细胞类型名称排序分配ID，确保映射稳定）
    
    参数:
        csv_file_path: CSV文件路径
        cell_type_col: 细胞类型所在列名（默认"cell_type"）
    
    返回:
        字典，键为整数ID，值为细胞类型名称
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取CSV文件：{os.path.abspath(csv_file_path)}")
    except Exception as e:
        raise FileNotFoundError(f"CSV文件读取失败：{str(e)}")
    
    # 验证列是否存在
    if cell_type_col not in df.columns:
        raise ValueError(f"CSV文件中缺少'{cell_type_col}'列，请检查列名")
    
    # 提取去重的细胞类型并排序（确保ID分配稳定）
    unique_cell_types = sorted(df[cell_type_col].dropna().unique().tolist())
    
    if not unique_cell_types:
        raise ValueError(f"列'{cell_type_col}'中未找到有效细胞类型数据")
    
    # 生成映射（ID从0开始）
    cell_type_mapping = {idx: cell_type for idx, cell_type in enumerate(unique_cell_types)}
    
    # 打印映射结果（便于验证）
    print(f"\n从CSV生成的CELL_TYPE_MAPPING：")
    for id_, cell_type in cell_type_mapping.items():
        print(f"  ID {id_}: {cell_type}")
    
    return cell_type_mapping

def minus_emb_each_cell_type(feature_file, latent_file,
                           output_dir="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/7_types/minus_emb"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取特征数据CSV文件
    print("读取特征数据...")
    feature_df = pd.read_csv(feature_file)

    # 确定所有细胞类型列（包含_count的列）
    cell_type_columns = [col for col in feature_df.columns if '_count' in col]
    cell_types = [col.replace('_count', '') for col in cell_type_columns]

    print(f"发现的细胞类型: {cell_types}")

    # 2. 读取潜在空间矩阵
    print("\n读取潜在空间矩阵...")
    latent_df = pd.read_csv(latent_file, index_col=0)  # 假设第一列是行名
    print(f"潜在空间矩阵原始形状: {latent_df.shape}")

    # 分离出Cell_type列（非数值）和特征列（数值）
    if 'Cell_type' in latent_df.columns:
        cell_type_column = latent_df['Cell_type'].copy()  # 保存Cell_type列
        numeric_columns = latent_df.columns[latent_df.columns != 'Cell_type']  # 数值列
        numeric_df = latent_df[numeric_columns].copy()  # 只处理数值列
        print(f"已分离出Cell_type列，保留 {len(numeric_columns)} 个数值列进行处理")
    else:
        print("警告: 未找到Cell_type列，将处理所有列")
        cell_type_column = None
        numeric_df = latent_df.copy()

    # 转换所有数值列到数值类型，无法转换的会设为NaN
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # 检查并处理NaN值
    if numeric_df.isnull().any().any():
        print("警告：潜在空间矩阵中存在无法转换为数值的值，已设为NaN并填充为0")
        numeric_df = numeric_df.fillna(0)

    # 3. 为每个细胞类型单独处理
    for cell_type in cell_types:
        print(f"\n处理细胞类型: {cell_type}")

        # 找到该细胞类型数量最多的那一行
        count_col = f"{cell_type}_count"
        max_row_idx = feature_df[count_col].idxmax()
        max_row = feature_df.loc[max_row_idx]

        print(
            f"{cell_type} 最多数量的行: 特征={max_row['feature_name']}, 系数={max_row['strength_multiplier']}, 数量={max_row[count_col]}")

        # 获取特征名称和扰动系数
        feature_name = max_row['feature_name']
        multiplier = max_row['strength_multiplier']

        # 检查特征是否存在于数值列中
        if feature_name not in numeric_df.columns:
            print(f"警告: 特征 {feature_name} 在潜在空间矩阵的数值列中未找到，跳过此细胞类型")
            continue

        # 复制原始数值矩阵用于此细胞类型的扰动
        perturbed_numeric = numeric_df.copy()

        # 应用扰动
        try:
            if multiplier>=2:
                # 对该特征列应用扰动
                perturbed_numeric[feature_name] = perturbed_numeric[feature_name] * 2
                print(f"已扰动特征 {feature_name}，系数: 2")
            else:
                # 对该特征列应用扰动
                perturbed_numeric[feature_name] = perturbed_numeric[feature_name] * (multiplier)
                print(f"已扰动特征 {feature_name}，系数: {multiplier}")
        except Exception as e:
            print(f"扰动特征 {feature_name} 时出错: {str(e)}")
            continue

        # 计算数值列的差值矩阵
        try:
            diff_numeric = perturbed_numeric - numeric_df
        except Exception as e:
            print(f"计算差值矩阵时出错: {str(e)}")
            continue

        # # 如果存在Cell_type列，将其添加回结果中
        # if cell_type_column is not None:
        #     perturbed_latent = pd.concat([perturbed_numeric, cell_type_column], axis=1)
        #     diff_df = pd.concat([diff_numeric, cell_type_column], axis=1)
        # else:
        #     perturbed_latent = perturbed_numeric
        #     diff_df = diff_numeric
        perturbed_latent = perturbed_numeric
        diff_df = diff_numeric

        # 保存扰动后的矩阵和差值矩阵
        perturbed_filename = os.path.join(output_dir, f"{cell_type}_perturbed.csv")
        diff_filename = os.path.join(output_dir, f"{cell_type}_diff.csv")

        perturbed_latent.to_csv(perturbed_filename)
        diff_df.to_csv(diff_filename)

        print(f"{cell_type} 扰动后矩阵已保存到 {perturbed_filename}")
        print(f"{cell_type} 差值矩阵已保存到 {diff_filename}")

    print("\n所有细胞类型处理完成")

def process_feature_perturbations(feature_file, latent_file,
                                  output_dir=r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE171993/type_experiment/feature_perturbations"):
    """
    对每个特征列应用不同扰动系数，并将结果按系数分类保存到对应子文件夹
    """
    # 定义扰动系数
    multipliers = [1.5, 2, 2.5]
    # 为系数创建安全的文件夹名称（将.替换为p）
    multiplier_folder_names = [f"multiplier_{str(m).replace('.', 'p')}" for m in multipliers]

    # 1. 创建完整的输出目录结构
    # 主文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 为每个扰动类型（perturbed和diff）创建带系数的子文件夹
    dir_structure = {}
    for perturb_type in ['perturbed', 'diff']:
        dir_structure[perturb_type] = {}
        # 创建perturbed或diff主文件夹
        main_type_dir = os.path.abspath(os.path.join(output_dir, perturb_type))
        os.makedirs(main_type_dir, exist_ok=True)

        # 为每个系数创建子文件夹
        for m, folder_name in zip(multipliers, multiplier_folder_names):
            coeff_dir = os.path.abspath(os.path.join(main_type_dir, folder_name))
            os.makedirs(coeff_dir, exist_ok=True)
            dir_structure[perturb_type][m] = coeff_dir

            # 验证目录是否创建成功
            if not os.path.isdir(coeff_dir):
                raise OSError(f"无法创建目录: {coeff_dir}")

    # 打印目录结构
    print(f"输出根目录: {os.path.abspath(output_dir)}")
    for perturb_type in dir_structure:
        print(f"\n{perturb_type} 目录结构:")
        for m in dir_structure[perturb_type]:
            print(f"  系数 {m}: {dir_structure[perturb_type][m]}")

    # 2. 读取潜在空间矩阵
    print("\n读取潜在空间矩阵...")
    try:
        latent_df = pd.read_csv(latent_file, index_col=0)  # 假设第一列是行名
        print(f"潜在空间矩阵原始形状: {latent_df.shape}")
    except Exception as e:
        raise ValueError(f"读取潜在空间矩阵失败: {str(e)}")

    # 分离出Cell_type列（非数值）和特征列（数值）
    if 'cell_type' in latent_df.columns:
        cell_type_column = latent_df['cell_type'].copy()  # 保存Cell_type列
        feature_columns = [col for col in latent_df.columns if col != 'cell_type']  # 特征列
        numeric_df = latent_df[feature_columns].copy()  # 只处理数值列
        print(f"已分离出cell_type列，找到 {len(feature_columns)} 个特征列")
    else:
        print("未找到cell_type列，所有列都将作为特征处理")
        cell_type_column = None
        feature_columns = latent_df.columns.tolist()
        numeric_df = latent_df.copy()

    # 转换所有特征列到数值类型，无法转换的会设为NaN
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # 检查并处理NaN值
    if numeric_df.isnull().any().any():
        print("警告：潜在空间矩阵中存在无法转换为数值的值，已设为NaN并填充为0")
        numeric_df = numeric_df.fillna(0)

    print(f"将对每个特征应用以下扰动系数: {multipliers}")

    # 3. 对每个特征列单独处理
    for feature_idx, feature_name in enumerate(feature_columns, 1):
        # 清理特征名中的特殊字符，避免影响文件命名
        clean_feature_name = feature_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*',
                                                                                                         '_').replace(
            '?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

        print(f"\n处理特征 ({feature_idx}/{len(feature_columns)}): {feature_name} (清理后: {clean_feature_name})")

        # 对每个扰动系数分别处理
        for multiplier in multipliers:
            # 复制原始数值矩阵用于此特征的扰动
            perturbed_numeric = numeric_df.copy()

            # 应用扰动：只对当前特征列乘以扰动系数
            try:
                perturbed_numeric[feature_name] = perturbed_numeric[feature_name] * multiplier
            except Exception as e:
                print(f"扰动特征 {feature_name} 时出错: {str(e)}，将跳过此特征")
                continue

            # 计算差值矩阵（扰动后 - 原始）
            try:
                diff_numeric = perturbed_numeric - numeric_df
            except Exception as e:
                print(f"计算差值矩阵时出错: {str(e)}")
                continue

            # # 如果存在Cell_type列，将其添加回结果中
            # if cell_type_column is not None:
            #     perturbed_result = pd.concat([perturbed_numeric, cell_type_column], axis=1)
            #     diff_result = pd.concat([diff_numeric, cell_type_column], axis=1)
            # else:
            #     perturbed_result = perturbed_numeric
            #     diff_result = diff_numeric

            perturbed_result = perturbed_numeric
            diff_result = diff_numeric

            # 构造文件名
            perturbed_filename = f"feature_{clean_feature_name}_perturbed.csv"
            diff_filename = f"feature_{clean_feature_name}_diff.csv"

            # 获取当前系数对应的文件夹路径
            perturbed_path = os.path.join(dir_structure['perturbed'][multiplier], perturbed_filename)
            diff_path = os.path.join(dir_structure['diff'][multiplier], diff_filename)

            # 保存文件
            try:
                perturbed_result.to_csv(perturbed_path)
                diff_result.to_csv(diff_path)

                # 验证文件是否保存成功
                if os.path.exists(perturbed_path) and os.path.getsize(perturbed_path) > 0:
                    print(f"已保存扰动文件: {perturbed_path}")
                else:
                    print(f"警告: 扰动文件保存失败或为空: {perturbed_path}")

                if os.path.exists(diff_path) and os.path.getsize(diff_path) > 0:
                    print(f"已保存差值文件: {diff_path}")
                else:
                    print(f"警告: 差值文件保存失败或为空: {diff_path}")

            except Exception as e:
                print(f"保存文件时出错: {str(e)}")
                continue

    print("\n所有特征处理完成")

def process_feature_perturbations_add_predType(feature_file, latent_file, classifier_model_path, label_encoder_path,
                                  output_dir=r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/7_types/feature_perturbations"):
    """
    对每个特征列应用不同扰动系数，并将结果按系数分类保存到对应子文件夹，
    同时添加扰动后的细胞类型预测结果
    """
    # 新增：加载分类器模型和标签编码器
    print("加载分类器模型和标签编码器...")
    try:
        from tensorflow.keras.models import load_model
        import joblib
        # 加载模型
        model = load_model(classifier_model_path)
        print(f"成功加载分类器模型: {classifier_model_path}")
        # 加载标签编码器
        le = joblib.load(label_encoder_path)
        print(f"成功加载标签编码器: {label_encoder_path}")
    except Exception as e:
        raise ValueError(f"加载模型或标签编码器失败: {str(e)}")

    # 定义扰动系数
    multipliers = [1.5, 2, 2.5]
    # 为系数创建安全的文件夹名称（将.替换为p）
    multiplier_folder_names = [f"multiplier_{str(m).replace('.', 'p')}" for m in multipliers]

    # 1. 创建完整的输出目录结构
    # 主文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 为每个扰动类型（perturbed和diff）创建带系数的子文件夹
    dir_structure = {}
    for perturb_type in ['perturbed', 'diff']:
        dir_structure[perturb_type] = {}
        # 创建perturbed或diff主文件夹
        main_type_dir = os.path.abspath(os.path.join(output_dir, perturb_type))
        os.makedirs(main_type_dir, exist_ok=True)

        # 为每个系数创建子文件夹
        for m, folder_name in zip(multipliers, multiplier_folder_names):
            coeff_dir = os.path.abspath(os.path.join(main_type_dir, folder_name))
            os.makedirs(coeff_dir, exist_ok=True)
            dir_structure[perturb_type][m] = coeff_dir

            # 验证目录是否创建成功
            if not os.path.isdir(coeff_dir):
                raise OSError(f"无法创建目录: {coeff_dir}")

    # 打印目录结构
    print(f"输出根目录: {os.path.abspath(output_dir)}")
    for perturb_type in dir_structure:
        print(f"\n{perturb_type} 目录结构:")
        for m in dir_structure[perturb_type]:
            print(f"  系数 {m}: {dir_structure[perturb_type][m]}")

    # 2. 读取潜在空间矩阵
    print("\n读取潜在空间矩阵...")
    try:
        latent_df = pd.read_csv(latent_file, index_col=0)  # 假设第一列是行名
        print(f"潜在空间矩阵原始形状: {latent_df.shape}")
    except Exception as e:
        raise ValueError(f"读取潜在空间矩阵失败: {str(e)}")

    # 分离出cell_type列（非数值）和特征列（数值）
    if 'cell_type' in latent_df.columns:
        cell_type_column = latent_df['cell_type'].copy()  # 保存cell_type列
        feature_columns = [col for col in latent_df.columns if col != 'cell_type']  # 特征列
        numeric_df = latent_df[feature_columns].copy()  # 只处理数值列
        print(f"已分离出cell_type列，找到 {len(feature_columns)} 个特征列")
    else:
        print("未找到cell_type列，所有列都将作为特征处理")
        cell_type_column = None
        feature_columns = latent_df.columns.tolist()
        numeric_df = latent_df.copy()

    # 转换所有特征列到数值类型，无法转换的会设为NaN
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # 检查并处理NaN值
    if numeric_df.isnull().any().any():
        print("警告：潜在空间矩阵中存在无法转换为数值的值，已设为NaN并填充为0")
        numeric_df = numeric_df.fillna(0)

    print(f"将对每个特征应用以下扰动系数: {multipliers}")

    # 3. 对每个特征列单独处理
    for feature_idx, feature_name in enumerate(feature_columns, 1):
        # 清理特征名中的特殊字符，避免影响文件命名
        clean_feature_name = feature_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*',
                                                                                                         '_').replace(
            '?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

        print(f"\n处理特征 ({feature_idx}/{len(feature_columns)}): {feature_name} (清理后: {clean_feature_name})")

        # 对每个扰动系数分别处理
        for multiplier in multipliers:
            # 复制原始数值矩阵用于此特征的扰动
            perturbed_numeric = numeric_df.copy()

            # 应用扰动：只对当前特征列乘以扰动系数
            try:
                perturbed_numeric[feature_name] = perturbed_numeric[feature_name] * multiplier
            except Exception as e:
                print(f"扰动特征 {feature_name} 时出错: {str(e)}，将跳过此特征")
                continue

            # 新增：使用分类器预测扰动后的细胞类型
            try:
                # 模型预测概率
                pred_proba = model.predict(perturbed_numeric.values, verbose=0)
                # 获取预测的类别ID
                pred_ids = np.argmax(pred_proba, axis=1)
                # 转换为细胞类型名称
                pred_cell_types = le.inverse_transform(pred_ids)
                # 添加到结果中
                perturbed_numeric['Cell_Type'] = pred_cell_types
                print(f"已完成特征 {feature_name} 在系数 {multiplier} 下的细胞类型预测")
            except Exception as e:
                print(f"预测细胞类型时出错: {str(e)}")
                continue

            # 计算差值矩阵（扰动后 - 原始）
            try:
                # 原始数值矩阵不包含新添加的Cell_Type列，确保维度一致
                diff_numeric = perturbed_numeric.drop('Cell_Type', axis=1) - numeric_df
                # 将预测的细胞类型也添加到差值矩阵中
                diff_numeric['Cell_Type'] = pred_cell_types
            except Exception as e:
                print(f"计算差值矩阵时出错: {str(e)}")
                continue

            perturbed_result = perturbed_numeric
            diff_result = diff_numeric

            # 构造文件名
            perturbed_filename = f"feature_{clean_feature_name}_perturbed.csv"
            diff_filename = f"feature_{clean_feature_name}_diff.csv"

            # 获取当前系数对应的文件夹路径
            perturbed_path = os.path.join(dir_structure['perturbed'][multiplier], perturbed_filename)
            diff_path = os.path.join(dir_structure['diff'][multiplier], diff_filename)

            # 保存文件
            try:
                perturbed_result.to_csv(perturbed_path)
                diff_result.to_csv(diff_path)

                # 验证文件是否保存成功
                if os.path.exists(perturbed_path) and os.path.getsize(perturbed_path) > 0:
                    print(f"已保存扰动文件: {perturbed_path}")
                else:
                    print(f"警告: 扰动文件保存失败或为空: {perturbed_path}")

                if os.path.exists(diff_path) and os.path.getsize(diff_path) > 0:
                    print(f"已保存差值文件: {diff_path}")
                else:
                    print(f"警告: 差值文件保存失败或为空: {diff_path}")

            except Exception as e:
                print(f"保存文件时出错: {str(e)}")
                continue

    print("\n所有特征处理完成")

def batch_stack_csvs_and_add_celltype(csv_folder: str,
                                      h5ad_path: str,
                                      output_name: str = "batch_combined_with_celltype.csv",
                                      exclude_files: List[str] = None,
                                      check_duplicate_cells: bool = True) -> None:
    """
    批量读取文件夹下所有CSV文件并堆叠，再添加细胞类型信息

    参数说明：
        csv_folder: str，存储所有待堆叠CSV文件的文件夹路径
        h5ad_path: str，用于获取细胞类型的h5ad文件路径
        output_name: str，最终输出文件的名称（默认：batch_combined_with_celltype.csv）
        exclude_files: List[str]，需要排除的CSV文件名（如临时文件，默认：None）
        check_duplicate_cells: bool，是否检查并去除重复的细胞行（按行名去重，默认：True）
    """
    # 初始化参数默认值
    exclude_files = exclude_files or []
    all_csv_dfs = []  # 存储所有读取成功的CSV数据框
    processed_files = []  # 记录已处理的文件（用于日志）

    try:
        # 1. 检查CSV文件夹是否存在
        if not os.path.exists(csv_folder):
            raise FileNotFoundError(f"CSV文件夹不存在：{csv_folder}")

        # 2. 遍历文件夹，筛选出所有CSV文件
        for filename in os.listdir(csv_folder):
            # 跳过非CSV文件和需排除的文件
            if not filename.endswith(".csv") or filename in exclude_files:
                continue

            # 拼接完整文件路径
            csv_path = os.path.join(csv_folder, filename)
            # 跳过文件夹（防止误判为文件）
            if os.path.isdir(csv_path):
                continue

            # 3. 读取单个CSV文件
            print(f"正在读取CSV文件：{filename}")
            try:
                # 读取时指定行名为第0列（与原函数兼容）
                df = pd.read_csv(csv_path, index_col=0)
                all_csv_dfs.append(df)
                processed_files.append(filename)
                print(f"成功读取：{filename}（样本数：{len(df)}，特征数：{len(df.columns)}）")
            except Exception as e:
                print(f"警告：读取文件 {filename} 失败，跳过该文件 - 错误原因：{str(e)}")
                continue

        # 4. 检查是否有有效CSV文件
        if not all_csv_dfs:
            raise ValueError(f"在文件夹 {csv_folder} 中未找到可读取的CSV文件（已排除：{exclude_files}）")

        # 5. 检查所有CSV文件的列是否一致（确保堆叠后格式正确）
        first_columns = all_csv_dfs[0].columns
        for idx, df in enumerate(all_csv_dfs[1:], start=1):
            if not df.columns.equals(first_columns):
                raise ValueError(f"文件列不匹配：{processed_files[0]} 的列与 {processed_files[idx]} 的列不一致，无法堆叠")

        # 6. 堆叠所有CSV数据框（行方向堆叠，axis=0）
        print(f"\n开始堆叠 {len(all_csv_dfs)} 个CSV文件...")
        combined_df = pd.concat(all_csv_dfs, axis=0, ignore_index=False)  # ignore_index=False 保留原行名
        print(f"堆叠完成 - 总样本数：{len(combined_df)}，总特征数：{len(combined_df.columns)}")

        # 7. （可选）检查并去除重复细胞（按行名去重，保留第一次出现的行）
        if check_duplicate_cells:
            duplicate_count = combined_df.index.duplicated().sum()
            if duplicate_count > 0:
                print(f"检测到 {duplicate_count} 个重复细胞（按行名），已自动去重")
                combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
                print(f"去重后样本数：{len(combined_df)}")
            else:
                print("未检测到重复细胞")

        # 8. 保存临时堆叠文件（供add_celltype_to_embedding函数使用）
        temp_combined_path = os.path.join(csv_folder, "temp_batch_combined.csv")
        combined_df.to_csv(temp_combined_path, index=True, encoding="utf-8")
        print(f"\n临时堆叠文件已保存至：{temp_combined_path}")

        # 9. 调用函数添加细胞类型
        add_celltype_to_embedding(
            csv_path=temp_combined_path,
            h5ad_path=h5ad_path,
            output_csv_name=output_name
        )

        # 10. 清理临时文件（避免残留）
        if os.path.exists(temp_combined_path):
            os.remove(temp_combined_path)
            print(f"已删除临时堆叠文件：{temp_combined_path}")

        # 最终日志
        print(f"\n=== 批量处理完成 ===")
        print(f"已处理CSV文件：{processed_files}（共 {len(processed_files)} 个）")
        print(f"最终结果文件：training_set/{output_name}（样本数：{len(combined_df)}）")

    except Exception as e:
        print(f"\n批量处理过程中发生错误：{str(e)}")
        # 若出错，尝试清理临时文件
        temp_combined_path = os.path.join(csv_folder, "temp_batch_combined.csv")
        if os.path.exists(temp_combined_path):
            os.remove(temp_combined_path)
            print(f"错误后已清理临时文件：{temp_combined_path}")

def perturb_two_features(
        csv_path: str,  # 输入：细胞潜在向量CSV文件路径
        model: Sequential,  # 预测模型（输入64维，输出各细胞类型概率）
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        scaler=None,  # 数据标准化器
        index_col: str = "index"  # CSV中的索引列名称
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取CSV文件，对64维细胞潜在向量的每对特征施加20种强度扰动（0.5×~10×），
    两个特征使用相同扰动系数，记录各细胞类型预测占比
    """
    # 1. 从CSV读取数据
    cell_latents, feature_names, cell_types = load_cell_latent_csv(
        csv_path=csv_path,
        index_col=index_col
    )
    n_cells = cell_latents.shape[0]  # 细胞数量
    all_cell_types = sorted(cell_type_mapping.items(), key=lambda x: x[0])  # 排序后的细胞类型
    cell_type_ids = [cid for cid, _ in all_cell_types]
    cell_type_names = [cname for _, cname in all_cell_types]
    n_features = len(feature_names)  # 特征数量（64）

    # 2. 初始化存储结果的数据结构
    perturbed_records = []

    print(f"\n双特征扰动初始化完成：")
    print(f"  - 细胞数量：{n_cells}，特征维度：{n_features}（{feature_names[0]}~{feature_names[-1]}）")
    print(f"  - 细胞类型列表：{cell_type_names}（共{len(cell_type_names)}种）")
    print(f"  - 扰动计划：每对特征×20种强度（0.5× → 1.0× → ... → 10.0×）")
    print(f"  - 总组合数：{n_features * (n_features + 1) // 2}")  # 计算组合数

    # 3. 遍历每对特征+20种强度，生成扰动向量并预测
    print("\n开始生成双特征扰动向量并预测...")
    total_pairs = n_features * (n_features + 1) // 2
    processed_pairs = 0

    # 遍历所有特征对（i <= j，避免重复）
    for i in range(n_features):
        for j in range(i, n_features):
            feat_name_i = feature_names[i]
            feat_name_j = feature_names[j]
            processed_pairs += 1

            # 遍历20种扰动强度（1→20对应0.5×→10×，步长0.5）
            for strength_step in range(1, 21):
                strength_multiplier = strength_step * 0.5  # 当前扰动强度

                # 生成扰动后的潜在向量（同时修改两个特征）
                perturbed_vecs = np.copy(cell_latents)
                perturbed_vecs[:, i] *= strength_multiplier  # 扰动第一个特征
                perturbed_vecs[:, j] *= strength_multiplier  # 扰动第二个特征（与第一个相同强度）

                # 模型预测（匹配训练时的数据格式）
                if scaler is not None:
                    # 标准化：转为DataFrame（匹配scaler的特征名）
                    perturbed_df = pd.DataFrame(perturbed_vecs, columns=feature_names)
                    perturbed_scaled = scaler.transform(perturbed_df)
                else:
                    perturbed_scaled = perturbed_vecs

                # 预测概率→取最大概率对应的细胞类型ID
                pred_proba = model.predict(perturbed_scaled, verbose=0)  # [n_cells, n_cell_types]
                pred_cell_type_ids = np.argmax(pred_proba, axis=1)  # 每个细胞的预测类型

                # 统计各细胞类型占比
                cell_type_count = {cid: 0 for cid in cell_type_ids}
                for cid in pred_cell_type_ids:
                    if cid in cell_type_count:
                        cell_type_count[cid] += 1

                # 构建单条扰动记录
                record = {
                    "feature_index_1": i,  # 第一个特征索引
                    "feature_name_1": feat_name_i,  # 第一个特征名称
                    "feature_index_2": j,  # 第二个特征索引
                    "feature_name_2": feat_name_j,  # 第二个特征名称
                    "strength_multiplier": strength_multiplier,  # 扰动强度（0.5~10.0）
                    "total_cells": n_cells,  # 总细胞数
                    "original_cell_type_dist": cell_types.value_counts().to_dict()  # 原始细胞类型分布
                }

                # 添加各细胞类型的计数与占比
                for cid, cname in all_cell_types:
                    count = cell_type_count[cid]
                    ratio = count / n_cells if n_cells > 0 else 0.0
                    record[f"{cname}_count"] = count  # 细胞类型计数
                    record[f"{cname}_ratio"] = round(ratio, 4)  # 细胞类型占比（保留4位小数）

                perturbed_records.append(record)

            # 打印进度
            if processed_pairs % 10 == 0 or processed_pairs == total_pairs:
                print(f"已处理 {processed_pairs}/{total_pairs} 对特征（当前：{feat_name_i} & {feat_name_j}）")

    # 4. 整理结果为DataFrame并保存
    perturb_result_df = pd.DataFrame(perturbed_records)
    save_filename = "two_features_perturbation_results.csv"
    perturb_result_df.to_csv(save_filename, index=False, encoding="utf-8")
    print(f"\n所有双特征扰动结果已保存至：{os.path.abspath(save_filename)}")

    return perturb_result_df, cell_types

def save_two_features_perturbation_trends(
        perturb_result_df: pd.DataFrame,
        target_feature_idx1: int,  # 第一个目标特征索引
        target_feature_idx2: int,  # 第二个目标特征索引
        target_feature_name1: str,  # 第一个目标特征名称
        target_feature_name2: str,  # 第二个目标特征名称
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        total_cells: int,  # 总细胞数
        save_dir: str,  # 图像保存目录
) -> None:
    """
    绘制指定特征对的“扰动强度-各细胞类型占比”关系图并保存
    """
    # 1. 筛选目标特征对的扰动数据
    target_data = perturb_result_df[
        (perturb_result_df["feature_index_1"] == target_feature_idx1) &
        (perturb_result_df["feature_index_2"] == target_feature_idx2)
        ].copy()
    if target_data.empty:
        raise ValueError(f"未找到特征对({target_feature_idx1}, {target_feature_idx2})的扰动数据")

    # 按扰动强度排序
    target_data = target_data.sort_values("strength_multiplier").reset_index(drop=True)
    cell_type_names = [cell_type_mapping[ctid] for ctid in sorted(cell_type_mapping.keys())]

    # 2. 创建图像
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 支持中英文
    plt.figure(figsize=(14, 8))

    # 3. 为每种细胞类型绘制趋势曲线
    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_type_names)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'x', '+']

    for i, cname in enumerate(cell_type_names):
        ratio_col = f"{cname}_ratio"
        if ratio_col not in target_data.columns:
            print(f"警告：细胞类型{cname}无占比数据，跳过绘制")
            continue

        # 绘制曲线
        plt.plot(
            target_data["strength_multiplier"],
            target_data[ratio_col],
            label=cname,
            color=colors[i],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=7,
            alpha=0.8
        )

    # 4. 添加原始强度参考线（×1.0）
    plt.axvline(
        x=1.0,
        color='#FF4444',
        linestyle='--',
        linewidth=2,
        alpha=0.9,
        label='ori_strength(×1.0)'
    )

    # 5. 设置图表属性
    plt.title(
        f'The influence of features {target_feature_idx1}({target_feature_name1}) & '
        f'{target_feature_idx2}({target_feature_name2}) perturbation\n'
        f'({total_cells}cells)',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('perturbation intensity', fontsize=14, fontweight='bold')
    plt.ylabel('Cell_type distribution', fontsize=14, fontweight='bold')

    plt.xlim(0, 10.5)
    plt.ylim(-0.02, 1.02)

    plt.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)
    plt.legend(
        title='Cell type',
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        shadow=True
    )

    # 6. 保存图像
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名，确保不重复
    plot_filename = f"cell_type_perturb_trend_feat{target_feature_idx1}_{target_feature_idx2}_" \
                    f"{target_feature_name1}_vs_{target_feature_name2}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"双特征扰动趋势图已保存至：{os.path.abspath(plot_path)}")

    # 清除当前图像，释放内存
    plt.close()

def run_perturbation_two_features(
        csv_path: str,  # 输入：细胞潜在向量CSV文件路径
        model_path: str,  # 细胞类型分类器模型路径（.h5格式）
        cell_type_mapping: Dict[int, str],  # 细胞类型ID→名称映射
        scaler=None,  # 数据标准化器
        index_col: str = "index",  # CSV中的索引列名称
        start_feature_pair: Optional[Tuple[int, int]] = None,  # 起始特征对，用于断点续跑
        save_dir: str = "/visualizations/two_features_perturbation"  # 图像保存目录
) -> None:
    """
    端到端运行双特征扰动流程：读取CSV→生成扰动结果→绘制细胞类型趋势图
    """
    # 1. 加载分类器模型
    print("=" * 60)
    print("1. 加载细胞类型分类器模型")
    print("=" * 60)
    try:
        model = load_model(model_path)
        print(f"成功加载模型：{os.path.abspath(model_path)}")
    except Exception as e:
        raise FileNotFoundError(f"模型加载失败：{str(e)}")

    # 2. 执行双特征扰动分析
    print("\n" + "=" * 60)
    print("2. 读取CSV并执行双特征潜在向量扰动")
    print("=" * 60)
    perturb_result_df, cell_types = perturb_two_features(
        csv_path=csv_path,
        model=model,
        cell_type_mapping=cell_type_mapping,
        scaler=scaler,
        index_col=index_col
    )

    # 3. 准备可视化双特征扰动趋势图
    print("\n" + "=" * 60)
    print("3. 准备可视化双特征扰动趋势图")
    print("=" * 60)

    # 获取所有唯一的特征对
    feature_pairs = set()
    for _, row in perturb_result_df.iterrows():
        pair = (row["feature_index_1"], row["feature_index_2"])
        feature_pairs.add(pair)
    feature_pairs = sorted(list(feature_pairs))  # 排序确保处理顺序一致

    # 确定起始索引（用于断点续跑）
    start_idx = 0
    if start_feature_pair is not None:
        try:
            start_idx = feature_pairs.index(start_feature_pair)
            print(f"从特征对 {start_feature_pair} 开始处理（索引：{start_idx}）")
        except ValueError:
            print(f"未找到起始特征对 {start_feature_pair}，将从头开始处理")

    # 遍历所有特征对并绘图
    total_pairs = len(feature_pairs)
    for i, (feat_idx1, feat_idx2) in enumerate(feature_pairs[start_idx:]):
        current_idx = start_idx + i

        # 获取特征名称
        pair_data = perturb_result_df[
            (perturb_result_df["feature_index_1"] == feat_idx1) &
            (perturb_result_df["feature_index_2"] == feat_idx2)
            ]
        feat_name1 = pair_data["feature_name_1"].iloc[0]
        feat_name2 = pair_data["feature_name_2"].iloc[0]

        print(f"\n处理特征对 {current_idx + 1}/{total_pairs}："
              f"特征{feat_idx1}（{feat_name1}）和特征{feat_idx2}（{feat_name2}）")

        # 绘制并保存趋势图
        save_two_features_perturbation_trends(
            perturb_result_df=perturb_result_df,
            target_feature_idx1=feat_idx1,
            target_feature_idx2=feat_idx2,
            target_feature_name1=feat_name1,
            target_feature_name2=feat_name2,
            cell_type_mapping=cell_type_mapping,
            total_cells=cell_types.shape[0],
            save_dir=save_dir
        )

    print("\n所有双特征扰动分析完成！")

def split_matrix_by_2time_period(large_matrix_path, time1_path, time2_path, output1_path, output2_path):
    """
    根据两个时间阶段的索引拆分大矩阵CSV文件

    参数:
    large_matrix_path: 大矩阵CSV文件路径
    time1_path: 第一个时间阶段CSV文件路径
    time2_path: 第二个时间阶段CSV文件路径
    output1_path: 第一个输出CSV文件路径
    output2_path: 第二个输出CSV文件路径
    """
    try:
        # 读取大矩阵CSV，假设索引在第一列
        large_matrix = pd.read_csv(large_matrix_path, index_col=0)
        print(f"成功读取大矩阵，形状: {large_matrix.shape}")

        # 读取两个时间阶段的CSV，假设索引在第一列
        time1 = pd.read_csv(time1_path, index_col=0)
        time2 = pd.read_csv(time2_path, index_col=0)
        print(f"成功读取第一个时间阶段，索引数量: {len(time1.index)}")
        print(f"成功读取第二个时间阶段，索引数量: {len(time2.index)}")

        # 获取两个时间阶段的索引
        time1_index = time1.index
        time2_index = time2.index

        # 检查索引是否有重叠
        overlapping = set(time1_index) & set(time2_index)
        if overlapping:
            print(f"警告: 两个时间阶段的索引存在重叠，共 {len(overlapping)} 个索引")

        # 根据索引拆分大矩阵
        matrix1 = large_matrix.loc[large_matrix.index.isin(time1_index)]
        matrix2 = large_matrix.loc[large_matrix.index.isin(time2_index)]

        print(f"拆分后第一个矩阵形状: {matrix1.shape}")
        print(f"拆分后第二个矩阵形状: {matrix2.shape}")

        # 保存结果
        matrix1.to_csv(output1_path)
        matrix2.to_csv(output2_path)

        print(f"成功保存第一个矩阵到 {output1_path}")
        print(f"成功保存第二个矩阵到 {output2_path}")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except KeyError as e:
        print(f"错误: 索引错误 - {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def split_multiple_matrices_by_time_periods(large_matrices_dir, time_periods_dir, output_root_dir):
    """
    批量处理多个大矩阵，按多个时间阶段拆分并分别保存到对应文件夹

    参数:
    large_matrices_dir: 存放多个大矩阵CSV的文件夹路径
    time_periods_dir: 存放多个时间阶段CSV的文件夹路径
    output_root_dir: 输出根目录，将在此目录下创建各时间阶段的子文件夹
    """
    try:
        # 获取所有大矩阵文件
        large_matrix_files = glob.glob(os.path.join(large_matrices_dir, "*.csv"))
        if not large_matrix_files:
            print(f"错误: 在 {large_matrices_dir} 中未找到任何大矩阵CSV文件")
            return
        print(f"找到 {len(large_matrix_files)} 个大矩阵文件，准备处理...")

        # 获取所有时间阶段文件
        time_period_files = glob.glob(os.path.join(time_periods_dir, "*.csv"))
        if not time_period_files:
            print(f"错误: 在 {time_periods_dir} 中未找到任何时间阶段CSV文件")
            return
        print(f"找到 {len(time_period_files)} 个时间阶段文件，准备处理...")

        # 读取所有时间阶段的索引并创建对应输出文件夹
        time_periods = {}  # 格式: {时间阶段名: (索引, 输出文件夹路径)}
        for time_file in time_period_files:
            # 获取时间阶段名称（不含扩展名）
            time_name = os.path.splitext(os.path.basename(time_file))[0]
            # 读取索引
            time_df = pd.read_csv(time_file, index_col=0)
            time_indices = time_df.index
            # 创建对应的输出文件夹
            time_output_dir = os.path.join(output_root_dir, time_name)
            os.makedirs(time_output_dir, exist_ok=True)
            # 存储信息
            time_periods[time_name] = (time_indices, time_output_dir)
            print(f"加载时间阶段 {time_name}，索引数量: {len(time_indices)}，输出路径: {time_output_dir}")

        # 检查时间阶段之间的索引重叠
        time_names = list(time_periods.keys())
        for i in range(len(time_names)):
            name1 = time_names[i]
            indices1 = time_periods[name1][0]
            for j in range(i + 1, len(time_names)):
                name2 = time_names[j]
                indices2 = time_periods[name2][0]
                overlapping = set(indices1) & set(indices2)
                if overlapping:
                    print(f"警告: 时间阶段 {name1} 与 {name2} 存在 {len(overlapping)} 个重叠索引")

        # 处理每个大矩阵
        for matrix_file in large_matrix_files:
            # 读取大矩阵
            matrix_name = os.path.splitext(os.path.basename(matrix_file))[0]
            large_matrix = pd.read_csv(matrix_file, index_col=0)
            print(f"\n处理大矩阵 {matrix_name}，形状: {large_matrix.shape}")

            # 按每个时间阶段拆分
            for time_name, (time_indices, time_output_dir) in time_periods.items():
                # 拆分矩阵
                split_matrix = large_matrix.loc[large_matrix.index.isin(time_indices)]
                # 生成输出文件名（保留原大矩阵名称）
                output_file = os.path.join(time_output_dir, f"{matrix_name}.csv")
                # 保存文件
                split_matrix.to_csv(output_file)
                print(f"  拆分到 {time_name} 完成，形状: {split_matrix.shape}，保存至 {output_file}")

        print("\n所有大矩阵拆分完成！")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except KeyError as e:
        print(f"错误: 索引错误 - {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def filter_cell_types(input_file, output_file, target_types):
    """
    筛选出指定的cell_type类别并保存结果
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        target_types (list): 需要保留的cell_type类别列表
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查cell_type列是否存在
    if 'cell_type' not in df.columns:
        raise ValueError("CSV文件中未找到'cell_type'列")
    
    # 筛选指定类别
    filtered_df = df[df['cell_type'].isin(target_types)]
    
    # 保存筛选结果
    filtered_df.to_csv(output_file, index=False)
    print(f"筛选完成，共保留 {len(filtered_df)} 行数据，已保存至 {output_file}")

def map_cell_types(input_file, output_file, mapping_dict):
    """
    对cell_type进行自定义映射并保存结果
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        mapping_dict (dict): 映射字典，格式为{原始类别: 目标类别}
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查cell_type列是否存在
    if 'cell_type' not in df.columns:
        raise ValueError("CSV文件中未找到'cell_type'列")
    
    # 执行映射，未在映射字典中的类别将设为NaN
    df['cell_type'] = df['cell_type'].map(mapping_dict)
    
    # 检查是否有未映射的类别
    unmapped = df['cell_type'].isna().sum()
    if unmapped > 0:
        print(f"警告：存在 {unmapped} 个未在映射字典中定义的类别，已设为NaN")
    
    # 保存映射结果
    df.to_csv(output_file, index=False)
    print(f"映射完成，已保存至 {output_file}")

if __name__ == "__main__":
    # # 一、首先将输出的向量预处理：各个时间段堆叠，然后加入细胞的类型。
    # # 1. 配置文件路径（替换为你的实际路径）
    # CSV_FOLDER = "models_compare/CVAE/embeddings"  # 存储所有CSV的文件夹
    # H5AD_PATH = "UNAGI/data/DentateGyrus/Processed_2000_epoch_100/processed_simp_10X43_1_2000.h5ad"  # 参考h5ad文件
    # OUTPUT_NAME = "embed_with_celltype.csv"  # 最终输出文件名
    # EXCLUDE_FILES = ["temp_batch_combined.csv"]  # 排除临时文件（防止重复读取）
    
    # # 2. 执行批量堆叠和添加细胞类型
    # batch_stack_csvs_and_add_celltype(
    #     csv_folder=CSV_FOLDER,
    #     h5ad_path=H5AD_PATH,
    #     output_name=OUTPUT_NAME,
    #     exclude_files=EXCLUDE_FILES,
    #     check_duplicate_cells=True
    # )
    
    # # 3. 筛选指定细胞类型
    # input_csv = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/training_set/all_embeddings_with_celltype.csv"  # 替换为你的输入文件路径
    # output_filtered = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/training_set/Granule immature_emb.csv"  # 筛选结果输出路径
    # #target_cell_types = ["Ngn3 low EP", "Ngn3 high EP","Fev+","Beta","Alpha","Delta","Epsilon"]  # 需要保留的细胞类型
    # #target_cell_types = ["Ngn3 low EP", "Ngn3 high EP","Fev+","Beta"] 
    # #target_cell_types = ["Ngn3 low EP", "Ngn3 high EP","Fev+","Alpha"] 
    # #target_cell_types = ["Ngn3 low EP", "Ngn3 high EP","Fev+","Delta"]
    # #target_cell_types = ["Ngn3 low EP", "Ngn3 high EP","Fev+","Epsilon"]  
    # target_cell_types = ["Granule immature"]
    # filter_cell_types(input_csv, output_filtered, target_cell_types)
    
    # # 4. 细胞类型映射
    # input_csv = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/training_set/filtered_Epsilon_ling.csv"  # 替换为你的输入文件路径
    # output_mapped = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/training_set/mapped_Epsilon_ling.csv"  # 映射结果输出路径
    # cell_mapping = {
    #     "Ngn3 low EP": "initial",
    #     "Ngn3 high EP": "initial",
    #     "Fev+": "Fev+",
    #     "Beta": "Beta",
    #     "Alpha": "Alpha",
    #     "Delta": "Delta",
    #     "Epsilon": "Epsilon"
    # }
    # map_cell_types(input_csv, output_mapped, cell_mapping)

    #二、训练分类器，并将模型保存。
    training_Classifier("models_compare/CVAE/training_set/embed_with_celltype.csv")

    # # 三、对所有的向量进行系统地扰动，观察其输出的结果
    # # 1. 配置文件路径（替换为你的实际路径）
    # EMBEDDING_CSV_PATH = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/training_set/Granule_immature_emb.csv"#合并后的embedding
    # CLASSIFIER_MODEL_PATH = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/Classifi_models/cell_type_classifier_Dentate_2000_epoch_100.h5"  # 细胞类型分类器模型
    # H5AD_FILE_PATH = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/processed_simp_10X43_1_2000.h5ad"  # 你的h5ad文件路径（含obs['cluster']）
    
    # # 2. 从h5ad自动生成CELL_TYPE_MAPPING（无需手动定义）
    
    # CELL_TYPE_MAPPING = generate_cell_type_mapping_from_h5ad(
    #     h5ad_file_path=H5AD_FILE_PATH,
    #     cluster_col="clusters"  # 若cluster列名不同，此处修改（如"cell_cluster"）
    # )
    
    # #CELL_TYPE_MAPPING = generate_cell_type_mapping_from_csv("/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/7_Cell_type/training_set/mapped_all_embed.csv")
    
    # custom_colors = {
    # "Astrocytes": "#3ba458",
    # "Cajal Retzius": "#404040",
    # "Cck-Tox": "#7a7a7a",
    # "Endothelial": "#fda762",
    # "GABA": "#6950a3",
    # "Granule immature": "#2575b7",
    # "Granule mature": "#08306b",
    # "Microglia": "#e1bfb0",
    # "Mossy": "#e5d8bd",
    # "Neuroblast": "#79b5d9",
    # "OL": "#f14432",
    # "OPC": "#fc8a6a",
    # "Radial Glia-like": "#98d594",
    # "nIPC": "#d0e1f2",
    # }
    
    # # 3. 执行全流程（后续逻辑不变）
    # run_perturbation(
    #     csv_path=EMBEDDING_CSV_PATH,
    #     model_path=CLASSIFIER_MODEL_PATH,
    #     cell_type_mapping=CELL_TYPE_MAPPING,
    #     scaler=None,
    #     index_col="index",
    #     target_feature=54,
    #     save_dir_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/distribution_Cell_Type_Granule_immature",
    #     cell_type_colors=custom_colors
    # )

    # # 四、尝试两个两个特征进行干扰
    # # 2. 关键改进：从h5ad自动生成CELL_TYPE_MAPPING（无需手动定义）
    
    # CELL_TYPE_MAPPING = generate_cell_type_mapping(
    #     h5ad_file_path=H5AD_FILE_PATH,
    #     cluster_col="clusters"  # 若cluster列名不同，此处修改（如"cell_cluster"）
    # )
    
    # # 3. 执行全流程（后续逻辑不变）
    # run_perturbation_two_features(
    #     csv_path=EMBEDDING_CSV_PATH,
    #     model_path=CLASSIFIER_MODEL_PATH,
    #     cell_type_mapping=CELL_TYPE_MAPPING,
    #     scaler=None,
    #     index_col="index",
    #     start_feature_pair=(0,0),
    #     save_dir="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/7_types/distrib_Cell_type/distribution_two_feature"
    # )

    # # 四、选择特定的扰动方案，将作减法的向量保存下来
    # # 输入文件路径（请根据实际情况修改）
    # feature_csv_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/perturb_result/cell_latents_perturbation_results.csv"  # 包含feature和扰动系数的CSV
    # latent_csv_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/training_set/all_embeddings_with_celltype_GSE132188.csv"  # 潜在空间矩阵CSV
    # output_feat_pert_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/minus_embed_all_feat_with_predType"
    # Classifier_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/Classfi_models/cell_type_classifier_GSE132188.h5"
    # LabelEncoder_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE132188/100epochs/Classfi_models/label_encoder.pkl"
    # # 执行处理流程，结果会保存在当前目录的perturbation_results文件夹中
    # #minus_emb_each_cell_type(feature_csv_path, latent_csv_path)
    
    # #process_feature_pert:将原有的数据的细胞类型添加入结果中
    # #process_feature_perturbations(feature_csv_path, latent_csv_path,output_feat_pert_path)
    
    # #
    # process_feature_perturbations_add_predType(feature_csv_path,latent_csv_path,
    #                                            Classifier_path,LabelEncoder_path,
    #                                            output_feat_pert_path)

    #
    # 时间阶段的划分：设置文件夹路径
    # # 存放所有大矩阵CSV的文件夹
    # large_matrix_folder = r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/minus_embed_all_feat_with_predType/diff/2p5/all"
    # # 时间阶段原先的扰动向量的路径（固定不变）
    # # 主要是为了读取细胞索引，指示时间阶段的分类。
    # time1_path = r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/embedding/latent_embeddings_time0.csv"
    # time2_path = r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/embedding/latent_embeddings_time1.csv"
    # # 输出文件夹（固定不变）
    # output_time0_folder = r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/minus_embed_all_feat_with_predType/diff/2p5/time0"
    # output_time1_folder = r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/minus_embed_all_feat_with_predType/diff/2p5/time1"

    # # 获取文件夹中所有CSV文件
    # csv_files = glob.glob(os.path.join(large_matrix_folder, "*.csv"))
    
    # if not csv_files:
    #     print(f"在文件夹 {large_matrix_folder} 中未找到任何CSV文件")
    # else:
    #     print(f"找到 {len(csv_files)} 个CSV文件，开始批量处理...\n")
    
    #     # 循环处理每个CSV文件
    #     for csv_file in csv_files:
    #         # 获取文件名（不含路径和扩展名）
    #         file_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    #         # 构造输出文件路径
    #         output1_path = os.path.join(output_time0_folder, f"{file_name}_Time0.csv")
    #         output2_path = os.path.join(output_time1_folder, f"{file_name}_Time1.csv")
    
    #         # 调用拆分函数
    #         split_matrix_by_time_periods(
    #             csv_file,  # 当前处理的大矩阵文件
    #             time1_path,  # 时间阶段1文件（固定）
    #             time2_path,  # 时间阶段2文件（固定）
    #             output1_path,  # 输出Time0文件
    #             output2_path  # 输出Time1文件
    #         )
    
    #     print("所有文件处理完成")
    
    # # 存放所有大矩阵CSV的文件夹
    # large_matrix_folder = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/minus_embed_all_feat_with_predType/perturbed/multi_2p5/all"
    # # 时间阶段原先的扰动向量的路径（固定不变）
    # # 主要是为了读取细胞索引，指示时间阶段的分类。
    # time_index_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/embedding"
    # # 输出文件夹（固定不变）
    # output_folder = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/minus_embed_all_feat_with_predType/perturbed/multi_2p5"
    
    # split_multiple_matrices_by_time_periods(
    #     large_matrices_dir=large_matrix_folder,  # 包含64个大矩阵的文件夹
    #     time_periods_dir=time_index_path,      # 包含6个时间阶段的文件夹
    #     output_root_dir=output_folder      # 输出根目录
    #     )