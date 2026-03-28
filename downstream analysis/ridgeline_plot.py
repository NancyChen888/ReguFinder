import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

# ----------------------
# 1. 设置全局参数和自定义配置
# ----------------------
# 设置全局绘图风格（顶刊级别的简洁风格）
sns.set_theme(style="white", rc={
    "axes.facecolor": (0, 0, 0, 0),
    "font.family": "Arial",       # 顶刊常用字体
    "font.size": 10,              # 基础字体大小
    "axes.linewidth": 1.2         # 坐标轴线条粗细
})

# 自定义配置 - 请根据你的需求修改
H5AD_FILE_PATH = "UNAGI/data/GSE171993_Hep/Hep_fil_addSim.h5ad"  # 替换为你的h5ad文件路径
TARGET_GENES = ["Junb", "Jun", "Tcf3", "Polr2d", "Fos", "Ncl"]  # 替换为你的6个目标基因
OUTPUT_FILE = "UNAGI/data/GSE171993_Hep/plot_for_figure/ridgeline_plot/gene_ridge_plot2_800dpi.png"  # 输出图片路径
DPI = 800  # 分辨率

# ----------------------
# 2. 读取h5ad文件并提取基因表达数据
# ----------------------
def load_gene_expression_from_h5ad(file_path, target_genes):
    """
    从h5ad文件中提取指定基因的表达数据
    
    参数:
        file_path: h5ad文件路径
        target_genes: 目标基因列表
    
    返回:
        DataFrame: 包含Gene和Expression列的数据集
    """
    # 读取h5ad文件
    adata = ad.read_h5ad(file_path)
    
    # 检查基因是否存在
    missing_genes = [gene for gene in target_genes if gene not in adata.var_names]
    if missing_genes:
        raise ValueError(f"以下基因在h5ad文件中不存在: {', '.join(missing_genes)}")
    
    # 提取指定基因的表达矩阵 (cells x genes)
    expr_matrix = adata[:, target_genes].X
    
    # 处理稀疏矩阵（如果是稀疏格式）
    if hasattr(expr_matrix, "toarray"):
        expr_matrix = expr_matrix.toarray()
    
    # 转换为DataFrame
    df_list = []
    for i, gene in enumerate(target_genes):
        expr_values = expr_matrix[:, i]
        # 过滤掉0表达（可选，根据需求调整）
        expr_values = expr_values[expr_values > 0]
        df_list.append(pd.DataFrame({
            "Gene": gene,
            "Expression": expr_values
        }))
    
    return pd.concat(df_list, ignore_index=True)

# 加载数据
df = load_gene_expression_from_h5ad(H5AD_FILE_PATH, TARGET_GENES)

# ----------------------
# 3. 绘制顶刊级别的山脊图
# ----------------------
# 顶刊配色方案（Nature/Cell风格的渐变配色）
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# 初始化FacetGrid
g = sns.FacetGrid(
    df, 
    row="Gene", 
    hue="Gene", 
    aspect=15,  # 宽高比
    height=0.8, # 每行高度
    palette=palette
)

# 绘制密度曲线（填充+轮廓）
# 主密度曲线（填充）
g.map(sns.kdeplot, "Expression",
      bw_adjust=0.6,  # 平滑度，可根据数据调整
      clip_on=False,
      fill=True,
      alpha=0.85,
      linewidth=1.5)

# 白色轮廓线（增强层次感）
g.map(sns.kdeplot, "Expression",
      clip_on=False,
      color="white",
      lw=2,
      bw_adjust=0.6)

# 添加基线
g.refline(y=0, linewidth=1.5, linestyle="-", color="black", clip_on=False)

# ----------------------
# 4. 美化图表（顶刊级别细节）
# ----------------------
# 添加基因标签
def add_gene_label(x, color, label):
    ax = plt.gca()
    # 调整标签位置和样式
    ax.text(
        0.02, 0.5, label,
        fontweight="bold",
        fontsize=11,
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes
    )

g.map(add_gene_label, "Expression")

# 调整子图间距（形成山脊重叠效果）
g.figure.subplots_adjust(hspace=-0.35)

# 移除多余元素
g.set_titles("")  # 移除自动生成的标题
g.set(
    yticks=[],        # 移除y轴刻度
    ylabel="",        # 移除y轴标签
    xlabel="Gene Expression Level"  # x轴标签
)
# 自定义x轴标签样式
g.set_xlabels(fontsize=12, fontweight="bold")

# 移除边框
g.despine(bottom=True, left=True)

# 添加整体标题（可选）
plt.suptitle("Gene Expression Distribution", y=1.02, fontsize=14, fontweight="bold")

# ----------------------
# 5. 保存高清图片
# ----------------------
plt.savefig(
    OUTPUT_FILE,
    dpi=DPI,
    bbox_inches="tight",  # 紧凑布局
    facecolor="white",    # 背景色
    edgecolor="none"      # 无边框
)
plt.close()  # 关闭画布释放内存

print(f"山脊图已保存至: {OUTPUT_FILE} (分辨率: {DPI} DPI)")