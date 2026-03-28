import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os

# -------------------------- 配置参数（已保留你的原始配置） --------------------------
# CSV文件路径
csv_file1 = "UNAGI/data/GSE171993_Hep/result_plot/heatmap_csv_Hep-D21/top250_processed_recons_feature_latent_dim_59_diff.csv"  # 第一个基因列表CSV文件
csv_file2 = "UNAGI/data/GSE171993_Hep/result_plot/heatmap_csv_Hep-D56/top250_processed_recons_feature_latent_dim_17_diff.csv"  # 第二个基因列表CSV文件
# 保存图片和CSV的文件夹路径
save_dir = "UNAGI/data/GSE171993_Hep/plot_for_figure/Venn"
# 图片名称（可自定义）
img_name = "gene_venn_diagram.png"
# 新增：共同基因CSV文件名（可自定义）
common_gene_csv_name = "common_regulators.csv"
# 韦恩图中两个集合的标签（可自定义）
label1 = "Regulators from dim59 perturbation"
label2 = "Regulators from dim17 perturbation"
# 分辨率设置（800 dpi）
dpi = 800

# -------------------------- 核心代码（新增共同基因保存功能） --------------------------
def draw_gene_venn(csv1, csv2, save_dir, img_name, common_gene_csv_name, label1, label2, dpi=800):
    """
    绘制两个基因集合的韦恩图并保存，同时保存共同基因到CSV文件
    
    参数:
    csv1: 第一个基因列表CSV文件路径
    csv2: 第二个基因列表CSV文件路径
    save_dir: 图片和CSV保存文件夹
    img_name: 图片文件名（含后缀，如.png）
    common_gene_csv_name: 共同基因CSV文件名（含后缀，如.csv）
    label1: 第一个集合的标签
    label2: 第二个集合的标签
    dpi: 图片分辨率
    """
    # 1. 读取CSV文件，提取第一列的基因名称（跳过列头）
    try:
        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)
    except FileNotFoundError as e:
        print(f"错误：找不到指定的CSV文件 - {e}")
        return
    except Exception as e:
        print(f"读取CSV文件时出错 - {e}")
        return
    
    # 获取第一列的列名，并提取基因列表（去重+去除空值）
    col1 = df1.columns[0]
    col2 = df2.columns[0]
    genes1 = set(df1[col1].dropna().unique())  # 转为集合去重
    genes2 = set(df2[col2].dropna().unique())
    
    # 新增：计算共同基因并保存为CSV
    common_genes = sorted(list(genes1 & genes2))  # 交集基因排序，便于查看
    common_genes_df = pd.DataFrame({
        "Common_Regulators": common_genes  # 列名贴合你的研究场景
    })
    # 创建保存文件夹（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 共同基因CSV保存路径
    common_csv_path = os.path.join(save_dir, common_gene_csv_name)
    # 保存CSV（不保留索引，编码为utf-8避免乱码）
    common_genes_df.to_csv(common_csv_path, index=False, encoding="utf-8")
    print(f"共同调控因子已保存到：{common_csv_path}")
    
    # 2. 拼接韦恩图保存路径
    save_path = os.path.join(save_dir, img_name)
    
    # 3. 设置绘图样式
    plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体（避免中文乱码）
    plt.figure(figsize=(8, 6), dpi=dpi)  # 设置画布大小和初始dpi
    
    # 4. 绘制韦恩图
    venn = venn2([genes1, genes2], (label1, label2))
    
    # 美化韦恩图（可选）
    for text in venn.set_labels:
        text.set_fontsize(12)  # 设置集合标签字体大小
    for text in venn.subset_labels:
        if text:  # 避免空值报错
            text.set_fontsize(10)  # 设置数字标签字体大小
    
    # 5. 保存图片（指定dpi）
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')  # bbox_inches避免标签被裁剪
    plt.close()  # 关闭画布释放内存
    
    # 打印更详细的统计信息
    print(f"韦恩图已成功保存到：{save_path}")
    print(f"{label1} 数量：{len(genes1)}")
    print(f"{label2} 数量：{len(genes2)}")
    print(f"共同调控因子数量：{len(common_genes)}")
    if len(common_genes) > 0:
        print(f"前5个共同调控因子：{common_genes[:5]}")  # 快速核对结果

# 执行绘图函数（新增common_gene_csv_name参数）
if __name__ == "__main__":
    draw_gene_venn(
        csv1=csv_file1,
        csv2=csv_file2,
        save_dir=save_dir,
        img_name=img_name,
        common_gene_csv_name=common_gene_csv_name,  # 新增参数
        label1=label1,
        label2=label2,
        dpi=dpi
    )