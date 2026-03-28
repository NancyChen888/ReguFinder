import os
import re
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# 【核心配置区】所有需要修改的参数都集中在这里，一站式调整
# ==============================================================================
CONFIG = {
    # 输入路径配置
    "input_folder": "UNAGI/data/haniffa_covid/recons_matrx/moderate",  # CSV文件所在文件夹
    "h5ad_path": "UNAGI/data/haniffa_covid/haniffa_hvg_addSim.h5ad",  # h5ad文件路径
    
    # 列名配置
    "cell_type_column": "Cell_Type",        # 细胞类型列名（CSV中）
    "cell_index_column": "covid_index",           # 细胞索引列名（CSV中）
    "h5ad_cell_type_col": "initial_clustering",       # h5ad中细胞类型列名
    "target_cell_type": "Severe",          # 新增：目标细胞类型（请修改为你需要的细胞类型名称）
    "sort_by_abs_value": True,              # 新增：是否按绝对值排序（True=按绝对值，False=按原始值）
    
    # 输出路径配置（统一指定保存文件夹）
    "output_folder": "UNAGI/data/haniffa_covid/result_plot",  # 根输出文件夹
    # 注意：这里不再直接定义带变量的子文件夹名，改为后续动态生成
    "csv_subfolder_prefix": "heatmap_csv_",  # CSV子文件夹前缀
    "plot_subfolder_prefix": "heatmap_",     # 热图子文件夹前缀
    
    # 分析参数配置
    "top_percent": 0.15,                   # 筛选前X%的基因（当前12.5%）
    "dpi": 800,                             # 热图保存DPI
    "figsize": (10, 15),                    # 热图尺寸
    "font_scale": 1,                        # seaborn字体缩放
    "ytick_fontsize": 7,                    # y轴字体大小
    
    # 基因过滤配置（核糖体/线粒体基因关键词，支持正则）
    "filter_gene_patterns": [
        r'^MT-',    # 线粒体基因（MT开头）
        r'^MRP',    # 线粒体相关基因
        r'^RPL',    # 核糖体大亚基基因
        r'^RPS',    # 核糖体小亚基基因
        r'^ribo',   # 核糖体相关（小写开头）
        r'^mito',   # 线粒体相关（小写开头）
        r'ribosomal',# 核糖体（英文全称）
        r'mitochondria' # 线粒体（英文全称）
    ]
}

# ==============================================================================
# 工具函数：过滤核糖体/线粒体基因
# ==============================================================================
def filter_ribo_mito_genes(gene_list, filter_patterns):
    """
    过滤掉核糖体/线粒体相关基因
    :param gene_list: 原始基因列表
    :param filter_patterns: 过滤的正则表达式列表
    :return: 过滤后的基因列表
    """
    filtered_genes = []
    for gene in gene_list:
        # 检查基因名是否匹配任何过滤模式（忽略大小写）
        if not any(re.search(pattern, str(gene), re.IGNORECASE) for pattern in filter_patterns):
            filtered_genes.append(gene)
    return filtered_genes

# ==============================================================================
# 动态生成带目标细胞类型的输出文件夹路径（核心修改）
# ==============================================================================
# 获取目标细胞类型
target_cell_type = CONFIG["target_cell_type"]
# 清理细胞类型名称中的特殊字符（避免文件夹名称非法）
clean_target_cell_type = re.sub(r'[<>:"/\\|?*]', '_', target_cell_type)

# 动态拼接子文件夹名称
csv_subfolder = CONFIG["csv_subfolder_prefix"] + clean_target_cell_type
plot_subfolder = CONFIG["plot_subfolder_prefix"] + clean_target_cell_type

# 构建完整的输出文件夹路径
csv_output_dir = os.path.join(CONFIG["output_folder"], csv_subfolder)
plot_output_dir = os.path.join(CONFIG["output_folder"], plot_subfolder)

# 自动创建输出文件夹（无需手动创建）
os.makedirs(csv_output_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

print(f"\n📌 本次分析的目标细胞类型：{target_cell_type}")
print(f"📁 CSV文件保存路径：{csv_output_dir}")
print(f"📁 热图文件保存路径：{plot_output_dir}")

# ==============================================================================
# 遍历输入文件夹中的CSV文件并处理
# ==============================================================================
for filename in os.listdir(CONFIG["input_folder"]):
    if filename.endswith('.csv'):
        # 构建完整的CSV文件路径
        csv_path = os.path.join(CONFIG["input_folder"], filename)
        print(f"\n正在处理文件: {csv_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        file_basename = os.path.basename(csv_path)

        # --------------------------
        # 校验目标细胞类型是否存在于当前文件中
        # --------------------------
        cell_type_list = df[CONFIG["cell_type_column"]].unique()
        if target_cell_type not in cell_type_list:
            print(f"目标细胞类型 '{target_cell_type}' 不存在于文件 {file_basename} 中，文件中的细胞类型：{cell_type_list}")
            continue
        
        # --------------------------
        # 计算各细胞类型的平均基因分数
        # --------------------------
        average_list = []
        for cell_type in cell_type_list:
            rslt_df = df[df[CONFIG["cell_type_column"]] == cell_type]
            # 排除索引列和细胞类型列，计算平均值
            average_list.append(rslt_df.drop(columns=[CONFIG["cell_index_column"], CONFIG["cell_type_column"]]).mean())

        # 整理为DataFrame（行=基因，列=细胞类型）
        making_dict = {cell_type: average for cell_type, average in zip(cell_type_list, average_list)}
        making = pd.DataFrame(making_dict)
        
        # --------------------------
        # 过滤核糖体/线粒体基因
        # --------------------------
        # 获取原始基因列表（DataFrame的行索引）
        original_genes = making.index.tolist()
        # 过滤基因
        filtered_genes = filter_ribo_mito_genes(original_genes, CONFIG["filter_gene_patterns"])
        # 筛选出非核糖体/线粒体基因的行
        making_filtered = making.loc[filtered_genes]
        
        print(f"原始基因数: {len(original_genes)}")
        print(f"过滤核糖体/线粒体后基因数: {len(filtered_genes)}")
        print(f"过滤掉的基因数: {len(original_genes) - len(filtered_genes)}")

        # --------------------------
        # 按目标细胞类型列排序
        # --------------------------
        # 选择目标细胞类型列
        target_column = making_filtered[target_cell_type]
        
        # 决定是否按绝对值排序
        if CONFIG["sort_by_abs_value"]:
            # 按目标细胞类型列的绝对值降序排序
            sorted_idx = target_column.abs().sort_values(ascending=False).index
        else:
            # 按目标细胞类型列的原始值降序排序
            sorted_idx = target_column.sort_values(ascending=False).index
        
        # 按排序后的索引重新排列DataFrame
        making_sorted = making_filtered.loc[sorted_idx]

        # --------------------------
        # 筛选top指定百分比的基因
        # --------------------------
        # 计算top基因数量（至少1个）
        top_num = max(1, int(len(making_sorted) * CONFIG["top_percent"]))
        top_genes = making_sorted.iloc[:top_num]
        print(f"按 {target_cell_type} 列筛选前{CONFIG['top_percent']*100}%的基因数: {len(top_genes)}")

        # --------------------------
        # 保存结果文件（保存到动态生成的文件夹）
        # --------------------------
        csv_save_path = os.path.join(csv_output_dir, f"top250_{file_basename}")
        top_genes.to_csv(csv_save_path)
        print(f"已保存按 {target_cell_type} 排序的Top基因CSV: {csv_save_path}")

        # --------------------------
        # 绘制并保存热图（保存到动态生成的文件夹）
        # --------------------------
        sns.set_theme(font_scale=CONFIG["font_scale"])
        plt.figure(figsize=CONFIG["figsize"])
        
        # 计算热图的最大/最小值
        Vmin = top_genes.min().min()
        Vmax = top_genes.max().max()
        
        # 绘制热图
        sns.heatmap(
            top_genes,
            cmap="vlag",
            vmin=Vmin,
            vmax=Vmax
        )
        
        # 设置字体大小并调整布局
        plt.yticks(fontsize=CONFIG["ytick_fontsize"])
        plt.title(f"Top Genes by {target_cell_type} (Top {CONFIG['top_percent']*100}%)", fontsize=12)
        plt.tight_layout()
        
        # 保存热图
        plot_save_path = os.path.join(plot_output_dir, f"top250_{file_basename}_heatmap.pdf")
        plt.savefig(plot_save_path, dpi=CONFIG["dpi"])
        plt.close()  # 关闭画布释放内存
        print(f"已保存按 {target_cell_type} 排序的热图文件: {plot_save_path}")
        print("-" * 80)

print("\n所有文件处理完成！")