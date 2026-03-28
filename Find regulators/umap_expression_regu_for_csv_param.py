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
    force_use_existing_dimred=True,  # 对emb类型强制使用已有数据（无则报错）
    umap_params=None  # 新增：UMAP参数配置字典
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
    force_use_existing_dimred: 手动指定是否使用已有降维数据
                               - True: 强制使用已有数据（无则报错）
                               - False: 强制重新计算（仅UMAP/TSNE有效）
                               - None: 自动检测（仅UMAP/TSNE有效）
    umap_params: UMAP参数配置字典，支持所有scanpy兼容的UMAP参数，默认值：
                 {
                     'n_components': 2,
                     'n_neighbors': 15,
                     'min_dist': 0.01,
                     'metric': 'cosine',
                     'random_state': 42
                 }
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

    # 4. 初始化UMAP参数（新增核心逻辑）
    default_umap_params = {
        'n_components': 2,
        'n_neighbors': 15,
        'min_dist': 0.01,
        'metric': 'cosine',
        'random_state': 42
    }
    # 合并用户传入的参数和默认参数（用户参数优先级更高）
    if umap_params is None:
        umap_params = default_umap_params
    else:
        # 补充默认值，避免参数缺失
        for key, val in default_umap_params.items():
            if key not in umap_params:
                umap_params[key] = val
    print(f"📌 使用UMAP参数：{umap_params}")

    # 5. 核心修改：单独处理X_emb逻辑
    if dimred_type.lower() == 'emb':
        # 检查X_emb是否存在
        if 'X_emb' not in adata.obsm:
            raise ValueError(f"adata.obsm中未找到X_emb数据！请确认h5ad文件中包含X_emb")
        
        # 检查X_emb维度（需要是2维才能画图）
        emb_data = adata.obsm['X_emb']
        if emb_data.ndim != 2 or emb_data.shape[1] != 2:
            # 如果是高维X_emb（如64维），自动用UMAP降维到2维（适配scanpy版本）
            print(f"⚠️ X_emb是{emb_data.shape[1]}维，自动用自定义UMAP参数降维到2维用于绘图...")
            # 临时复制数据做UMAP降维
            adata_emb = adata.copy()
            # 用X_emb替换X，基于emb做UMAP
            adata_emb.X = emb_data
            sc.pp.scale(adata_emb, zero_center=True, max_value=10)
            sc.tl.pca(adata_emb, n_comps=min(n_pcs, emb_data.shape[1]), svd_solver='arpack')
            
            # 关键修复：metric参数放在neighbors中（适配旧版scanpy）
            sc.pp.neighbors(
                adata_emb, 
                n_neighbors=umap_params['n_neighbors'],
                n_pcs=min(n_pcs, emb_data.shape[1]),
                metric=umap_params['metric']  # metric参数移到这里
            )
            # sc.tl.umap只传支持的参数
            sc.tl.umap(
                adata_emb,
                n_components=umap_params['n_components'],
                min_dist=umap_params['min_dist'],
                random_state=umap_params['random_state']
            )
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
    
    # 6. 原有UMAP/TSNE逻辑（适配自定义UMAP参数，修复metric报错）
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
                # 关键修复：metric参数移到sc.pp.neighbors中（适配所有scanpy版本）
                sc.pp.neighbors(
                    adata_hv, 
                    n_neighbors=umap_params['n_neighbors'],
                    n_pcs=n_pcs,
                    metric=umap_params['metric']  # metric参数放在这里
                )
                # sc.tl.umap只传它支持的参数（移除metric）
                sc.tl.umap(
                    adata_hv,
                    n_components=umap_params['n_components'],
                    min_dist=umap_params['min_dist'],
                    random_state=umap_params['random_state']
                )
                plot_func = sc.pl.umap
            elif dimred_type.lower() == 'tsne':
                sc.tl.tsne(adata_hv, n_pcs=n_pcs)
                plot_func = sc.pl.tsne
            else:
                raise ValueError("dimred_type 仅支持 'emb'、'umap' 或 'tsne'")
            
            adata_for_plot = adata_hv
    
    # 7. 筛选csv中在数据集中存在的基因
    valid_genes = [gene for gene in var_genes if gene in adata_for_plot.var_names]
    if not valid_genes:
        raise ValueError("在提供的h5ad文件中没有找到任何匹配的基因！")
    print(f"✅ 共找到 {len(valid_genes)} 个有效基因，开始批量绘图...")
    
    # 8. 设置绘图风格
    sc.set_figure_params(figsize=figsize, dpi=dpi, fontsize=12)
    
    # 9. 批量绘制并保存每个基因的表达图
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
            save_path = os.path.join(output_dir, f'{gene}_{dimred_type}_expression.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
            print(f"📁 已保存：{save_path}")
        except Exception as e:
            print(f"❌ 绘制基因 {gene} 失败，错误信息：{str(e)}")

# --- 示例调用（使用自定义UMAP参数）---
if __name__ == "__main__":
    # 请根据你的实际文件路径修改以下参数
    h5ad_file = "UNAGI/data/GSE171993_Hep/Hep_fil_addSim.h5ad"
    csv_file = "UNAGI/data/GSE171993_Hep/result_plot/Hep_D21/heatmap_0.1_csv/top0.1_processed_recons_feature_latent_dim_17_diff.csv_making.csv"
    output_folder = "UNAGI/data/GSE171993_Hep/plot_for_figure/Hep_D21"
    
    # 自定义UMAP参数（你可以根据需要修改）
    umap_params = {
        'n_components': 2,
        'n_neighbors': 10,
        'min_dist': 0.01,
        'metric': 'cosine',
        'random_state': 42
    }
    
    # 核心修改：传入自定义UMAP参数
    plot_gene_expression_on_dimred(
        h5ad_path=h5ad_file,
        csv_path=csv_file,
        output_dir=output_folder,
        dimred_type='umap',  # 指定使用UMAP
        figsize=(6, 5),
        dpi=850,
        n_pcs=50,
        force_use_existing_dimred=False,  # 强制重新计算UMAP（使用自定义参数）
        umap_params=umap_params  # 传入自定义UMAP参数
    )