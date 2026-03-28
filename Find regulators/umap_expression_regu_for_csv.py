import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_gene_expression_on_dimred(
    h5ad_path, 
    csv_path, 
    output_dir,
    dimred_type='emb',  # 新增 'emb' 选项，默认使用X_emb
    figsize=(6, 5), 
    dpi=750,
    n_pcs=50,  # PCA降维的主成分数（仅UMAP/TSNE有效）
    n_neighbors=15,  # UMAP的邻近点数（仅UMAP有效）
    force_use_existing_dimred=True  # 对emb类型强制使用已有数据（无则报错）
):
    """
    批量绘制高变基因在降维空间中的表达情况并保存。
    
    参数:
    h5ad_path: 输入的h5ad文件路径
    csv_path: 包含高变基因的csv文件路径，第一列为基因名
    output_dir: 图片保存的目标文件夹
    dimred_type: 降维类型，可选 'emb'（X_emb）、'umap' 或 'tsne'，默认 'emb'
    figsize: 每个子图的尺寸，默认(6, 5)
    dpi: 保存图片的分辨率，默认750
    n_pcs: PCA主成分数（仅UMAP/TSNE有效），默认50
    n_neighbors: UMAP邻近点数（仅UMAP有效），默认15
    force_use_existing_dimred: 手动指定是否使用已有降维数据
                               - True: 强制使用已有数据（无则报错）
                               - False: 强制重新计算（仅UMAP/TSNE有效）
                               - None: 自动检测（仅UMAP/TSNE有效）
    """
    # 1. 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 读取数据
    adata = sc.read(h5ad_path)
    df_genes = pd.read_csv(csv_path)
    # 获取csv第一列的基因名
    var_genes = df_genes.iloc[:, 0].tolist()
    
    # 3. 定义关键变量
    dimred_key = f'X_{dimred_type.lower()}'
    plot_func = None
    use_existing_dimred = False
    adata_for_plot = adata  # 用于绘图的最终数据对象

    # 4. 核心修改：单独处理X_emb逻辑
    if dimred_type.lower() == 'emb':
        # 检查X_emb是否存在
        if 'X_emb' not in adata.obsm:
            raise ValueError(f"adata.obsm中未找到X_emb数据！请确认h5ad文件中包含X_emb")
        
        # 检查X_emb维度（需要是2维才能画图）
        emb_data = adata.obsm['X_emb']
        if emb_data.ndim != 2 or emb_data.shape[1] != 2:
            # 如果是高维X_emb（如64维），自动用UMAP降维到2维
            print(f"⚠️ X_emb是{emb_data.shape[1]}维，自动降维到2维用于绘图...")
            # 临时复制数据做UMAP降维
            adata_emb = adata.copy()
            # 用X_emb替换X，基于emb做UMAP
            adata_emb.X = emb_data
            sc.pp.scale(adata_emb, zero_center=True, max_value=10)
            sc.tl.pca(adata_emb, n_comps=min(50, emb_data.shape[1]), svd_solver='arpack')
            sc.pp.neighbors(adata_emb, n_neighbors=n_neighbors, n_pcs=min(n_pcs, emb_data.shape[1]))
            sc.tl.umap(adata_emb)
            # 将降维后的UMAP作为X_emb的绘图数据
            adata.obsm['X_emb_2d'] = adata_emb.obsm['X_umap']
            dimred_key = 'X_emb_2d'
            adata_for_plot = adata
        else:
            # X_emb已是2维，直接使用
            print(f"✅ 检测到2维X_emb数据，直接用于绘图...")
        
        # 自定义X_emb的绘图函数（核心：用matplotlib绘制，兼容scanpy风格）
        def plot_emb(adata, color, color_map='viridis', title='', show=False, return_fig=True):
            # 获取X_emb数据
            if dimred_key == 'X_emb_2d':
                emb_coords = adata.obsm['X_emb_2d']
            else:
                emb_coords = adata.obsm['X_emb']
            
            # 创建画布
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # 获取基因表达值
            expr_values = adata[:, color].X.toarray() if hasattr(adata[:, color].X, 'toarray') else adata[:, color].X
            expr_values = expr_values.flatten()
            
            # 绘制散点图（模仿scanpy的viridis配色）
            scatter = ax.scatter(
                emb_coords[:, 0], emb_coords[:, 1],
                c=expr_values, cmap=color_map,
                s=1, alpha=0.8, edgecolors='none'
            )
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(f'{color} expression', fontsize=12)
            
            # 设置标题和样式
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Emb Dimension 1', fontsize=12)
            ax.set_ylabel('Emb Dimension 2', fontsize=12)
            ax.axis('off')  # 隐藏坐标轴（模仿scanpy的UMAP图）
            plt.tight_layout()
            
            if show:
                plt.show()
            if return_fig:
                return fig
            return ax
        
        plot_func = plot_emb
        use_existing_dimred = True
        print(f"📌 成功加载X_emb数据，准备绘图...")
    
    # 5. 原有UMAP/TSNE逻辑（保留）
    else:
        if force_use_existing_dimred is not None:
            if force_use_existing_dimred:
                if dimred_key not in adata.obsm:
                    raise ValueError(f"强制使用已有{dimred_type.upper()}数据，但adata中未找到{dimred_key}！")
                use_existing_dimred = True
                print(f"✅ 强制使用已有 {dimred_type.upper()} 降维数据（{dimred_key}）...")
            else:
                use_existing_dimred = False
                print(f"🔄 强制重新计算 {dimred_type.upper()} 降维数据（忽略已有数据）...")
        else:
            if dimred_key in adata.obsm:
                use_existing_dimred = True
                print(f"📌 检测到已有 {dimred_type.upper()} 降维数据（{dimred_key}），直接使用...")
            else:
                use_existing_dimred = False
                print(f"📌 未检测到 {dimred_type.upper()} 降维数据（{dimred_key}），开始自动降维...")
        
        if use_existing_dimred:
            if dimred_type.lower() == 'umap':
                plot_func = sc.pl.umap
            else:
                plot_func = sc.pl.tsne
        else:
            adata_hv = adata.copy()
            sc.pp.scale(adata_hv, zero_center=True, max_value=10)
            sc.tl.pca(adata_hv, n_comps=n_pcs, svd_solver='arpack')
            
            if dimred_type.lower() == 'umap':
                sc.pp.neighbors(adata_hv, n_neighbors=n_neighbors, n_pcs=n_pcs)
                sc.tl.umap(adata_hv)
                plot_func = sc.pl.umap
            elif dimred_type.lower() == 'tsne':
                sc.tl.tsne(adata_hv, n_pcs=n_pcs)
                plot_func = sc.pl.tsne
            else:
                raise ValueError("dimred_type 仅支持 'emb'、'umap' 或 'tsne'")
            
            adata_for_plot = adata_hv
    
    # 6. 筛选csv中在数据集中存在的基因
    valid_genes = [gene for gene in var_genes if gene in adata_for_plot.var_names]
    if not valid_genes:
        raise ValueError("在提供的h5ad文件中没有找到任何匹配的基因！")
    print(f"✅ 共找到 {len(valid_genes)} 个有效基因，开始批量绘图...")
    
    # 7. 设置绘图风格
    sc.set_figure_params(figsize=figsize, dpi=dpi, fontsize=12)
    
    # 8. 批量绘制并保存每个基因的表达图
    for gene in valid_genes:
        try:
            # 绘制表达图（兼容emb/umap/tsne）
            fig = plot_func(
                adata_for_plot,
                color=gene,
                color_map='viridis',
                title=gene,
                show=False,
                return_fig=True
            )
            # 保存图片（bbox_inches='tight' 去除多余白边）
            save_filename = f'{gene}_{dimred_type}_expression.png'
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
            print(f"📁 已保存：{save_path}")
        except Exception as e:
            print(f"❌ 绘制基因 {gene} 失败，错误信息：{str(e)}")

# --- 示例调用（使用X_emb绘图）---
if __name__ == "__main__":
    # 请根据你的实际文件路径修改以下参数
    h5ad_file = "UNAGI/data/haniffa_covid/haniffa_hvg_addSim.h5ad"
    csv_file = "UNAGI/data/haniffa_covid/result_plot/heatmap_csv_Critical/top250_processed_recons_feature_latent_dim_17_diff.csv"
    output_folder = "UNAGI/data/haniffa_covid/plot_for_figure/umap_gene_exp/severe_2_critical"
    
    # 核心修改：使用X_emb绘图
    plot_gene_expression_on_dimred(
        h5ad_path=h5ad_file,
        csv_path=csv_file,
        output_dir=output_folder,
        dimred_type='umap',  # 指定使用X_emb
        figsize=(6, 5),
        dpi=850,
        force_use_existing_dimred=True  # 强制使用已有X_emb数据
    )