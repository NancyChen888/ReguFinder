import cospar as cs
import numpy as np
import scanpy as sc

# 打印Cospar版本信息
cs.logging.print_version()

# 设置日志详细程度
cs.settings.verbosity = 2

# 设置数据和图表的保存路径
cs.settings.data_path = "UNAGI/data/Larry_hematopoiesis/LARRY_data"  # 保存数据的相对路径，如果不存在会自动创建
cs.settings.figure_path = "UNAGI/data/Larry_hematopoiesis/LARRY_figure"  # 保存图表的相对路径，如果不存在会自动创建

# 设置图表参数
cs.settings.set_figure_params(
    format="png", 
    figsize=[4, 3.5], 
    dpi=400, 
    fontsize=14, 
    pointsize=2
)

# 测试用的路径设置（当前注释掉）
# cs.settings.data_path='data_cospar'
# cs.settings.figure_path='fig_cospar'
# adata_orig=cs.datasets.hematopoiesis_subsampled()
# adata_orig.uns['data_des']=['LARRY_sp500_ranking1']

# 加载数据
print("Loading data...")
#adata_orig = cs.datasets.hematopoiesis()
adata_orig= sc.read("/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/Larry_hematopoiesis/LARRY_data/LARRY_adata_preprocessed.h5ad")

# 打印数据信息
print("AnnData object information:")
print(f"n_obs × n_vars = {adata_orig.n_obs} × {adata_orig.n_vars}")
print("obs:", list(adata_orig.obs.keys()))
print("uns:", list(adata_orig.uns.keys()))
print("obsm:", list(adata_orig.obsm.keys()))

# 绘制嵌入图并按state_info着色
print("Generating embedding plot...")
#cs.pl.embedding(adata_orig, color="state_info")
plot_filename = "embedding_time_info.png"
# 绘制并保存图像，save=True会保存到cs.settings.figure_path指定的路径
cs.pl.embedding(
    adata_orig, 
    color="time_info", 
    save=plot_filename  # 保存图像，会自动添加设置的格式后缀
)

# 检查可用选项
print("Checking available choices...")
cs.hf.check_available_choices(adata_orig)

print(f"Analysis completed successfully.Embedding plot saved to {cs.settings.figure_path}/{plot_filename}")
