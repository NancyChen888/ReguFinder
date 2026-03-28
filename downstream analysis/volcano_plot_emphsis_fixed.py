import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import os
import random

# 设置随机种子，保证每次抽样结果可复现
random.seed(42)
np.random.seed(42)

# -------------------------- 全局字体配置（适配低版本matplotlib） --------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'  # 全局字体加粗（基础）
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗

def filter_ribo_mt_genes(genes_list):
    """
    筛选剔除核糖体基因和线粒体基因
    :param genes_list: 原始候选基因列表
    :return: 剔除后的非核糖体/线粒体基因列表
    """
    # 定义核糖体基因和线粒体基因的特征前缀（支持大小写，覆盖人/小鼠命名）
    ribo_prefixes = ['RPS', 'RPL', 'MRPS', 'MRPL']  # 核糖体基因核心前缀
    mt_prefixes = ['MT-', 'mt-', 'Mt-']             # 线粒体基因前缀（人类MT-，小鼠mt-）
    
    # 初始化筛选结果和剔除记录
    filtered_genes = []
    ribo_removed = []
    mt_removed = []
    
    for gene in genes_list:
        gene_upper = gene.upper()  # 这里是单个字符串，可用upper()
        # 判断是否为核糖体基因
        is_ribo = any(gene_upper.startswith(p) for p in ribo_prefixes)
        # 判断是否为线粒体基因
        is_mt = any(gene_upper.startswith(p) for p in mt_prefixes)
        
        if is_ribo:
            ribo_removed.append(gene)
        elif is_mt:
            mt_removed.append(gene)
        else:
            filtered_genes.append(gene)
    
    # 打印筛选日志，便于核对
    print(f"\n===== 基因筛选日志 =====")
    print(f"原始候选基因数量：{len(genes_list)}")
    print(f"剔除核糖体基因数量：{len(ribo_removed)} | 示例：{ribo_removed[:5]}")
    print(f"剔除线粒体基因数量：{len(mt_removed)} | 示例：{mt_removed[:5]}")
    print(f"最终候选基因数量：{len(filtered_genes)}")
    print("========================\n")
    
    return filtered_genes

def run_diff_analysis_between_cell_types(adata, cell_type_col, source_cell, target_cell, method='wilcoxon'):
    """
    仅针对源细胞类型和目标细胞类型执行差异表达分析
    :param adata: scanpy的AnnData对象
    :param cell_type_col: 细胞类型列名（如cell_type，需在adata.obs中）
    :param source_cell: 源细胞类型名称（如'Neuron'，需匹配adata.obs[cell_type_col]中的值）
    :param target_cell: 目标细胞类型名称（如'Astrocyte'，需匹配adata.obs[cell_type_col]中的值）
    :param method: 差异分析方法（wilcoxon/ t-test，推荐wilcoxon）
    :return: 补充了log2FC和pvalue的AnnData对象
    """
    # 1. 检查细胞类型列是否存在
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"h5ad的adata.obs中缺少 {cell_type_col} 列！请确保该列存储了细胞类型标注")
    
    # 2. 检查源/目标细胞类型是否存在
    cell_types = adata.obs[cell_type_col].unique()
    if source_cell not in cell_types:
        raise ValueError(f"源细胞类型 {source_cell} 不存在！可选类型：{list(cell_types)}")
    if target_cell not in cell_types:
        raise ValueError(f"目标细胞类型 {target_cell} 不存在！可选类型：{list(cell_types)}")
    
    # 3. 筛选仅源细胞和目标细胞的子集（减少计算量，提高准确性）
    adata_subset = adata[adata.obs[cell_type_col].isin([source_cell, target_cell])].copy()
    print(f"筛选出 {source_cell} 和 {target_cell} 类型的细胞，共 {adata_subset.shape[0]} 个细胞")
    
    # 4. 执行差异分析（对比源细胞vs目标细胞）
    print(f"开始差异分析：{source_cell} vs {target_cell}，方法={method}")
    sc.tl.rank_genes_groups(
        adata_subset,
        groupby=cell_type_col,
        reference=source_cell,  # 源细胞为参考组，对比目标细胞
        method=method,
        n_genes=adata_subset.shape[1],  # 输出所有基因的结果
        pts=True
    )
    
    # 5. 提取差异分析结果到adata.var中（匹配原始adata的基因顺序）
    diff_results = sc.get.rank_genes_groups_df(adata_subset, group=target_cell)  # 仅取目标细胞vs源细胞的结果
    # 重命名列并整理
    diff_results.rename(columns={
        'names': 'gene', 
        'logfoldchanges': 'log2FC', 
        'pvals': 'pvalue'
    }, inplace=True)
    # 按基因名去重，保留目标结果
    diff_results = diff_results.drop_duplicates(subset=['gene'], keep='first')
    diff_results.set_index('gene', inplace=True)
    
    # 6. 合并到原始adata.var中（确保所有基因都有值）
    adata.var = adata.var.join(diff_results[['log2FC', 'pvalue']], how='left')
    # 填充缺失值（无差异的基因设为0/log2FC，1/pvalue）
    adata.var['log2FC'] = adata.var['log2FC'].fillna(0)
    adata.var['pvalue'] = adata.var['pvalue'].fillna(1.0)
    
    print(f"差异分析完成：{source_cell} vs {target_cell}，已补充log2FC和pvalue列")
    return adata

def load_h5ad_data(h5ad_path, cell_type_col, source_cell, target_cell):
    """
    从h5ad文件读取数据 + 针对指定细胞类型做差异分析 + 提取全基因的log2FC和pvalue
    :param h5ad_path: h5ad文件路径
    :param cell_type_col: 细胞类型列名（如cell_type）
    :param source_cell: 源细胞类型名称
    :param target_cell: 目标细胞类型名称
    :return: 包含gene, log2FC, pvalue的DataFrame（全基因）
    """
    # 读取h5ad
    adata = sc.read_h5ad(h5ad_path)
    print(f"成功读取h5ad文件，数据维度：{adata.shape}")
    
    # 强制执行指定细胞类型的差异分析（覆盖原有结果，确保精准）
    adata = run_diff_analysis_between_cell_types(
        adata, 
        cell_type_col=cell_type_col,
        source_cell=source_cell,
        target_cell=target_cell
    )
    
    # 提取全基因数据并整理
    df_all_genes = adata.var[['log2FC', 'pvalue']].reset_index()
    df_all_genes.rename(columns={'index': 'gene'}, inplace=True)
    
    # 处理P值：0值替换为极小值（避免-log10(0)报错）
    df_all_genes['pvalue'] = df_all_genes['pvalue'].replace(0, 1e-300)
    # 处理NaN值（极少数情况）
    df_all_genes['pvalue'] = df_all_genes['pvalue'].fillna(1.0)
    df_all_genes['log2FC'] = df_all_genes['log2FC'].fillna(0)
    
    # 修复核心错误：Series需用.str.upper()，而非直接upper()
    df_all_genes['gene'] = df_all_genes['gene'].astype(str).str.strip().str.upper()
    return df_all_genes

def load_csv_genes(csv_path):
    """
    从CSV读取候选高变基因列表 + 自动剔除核糖体/线粒体基因
    :param csv_path: CSV文件路径
    :return: 筛选后的候选高变基因列表
    """
    # 读取CSV第一列，忽略列头
    df_key_genes = pd.read_csv(csv_path, usecols=[0])  # 仅读第一列
    raw_genes = df_key_genes.iloc[:, 0].dropna().unique().tolist()  # 去重、去空值
    
    # 统一基因名格式（和h5ad中的基因名对齐）
    raw_genes = [str(gene).strip().upper() for gene in raw_genes if str(gene).strip() != '']
    
    if len(raw_genes) == 0:
        raise ValueError("CSV第一列未读取到有效基因名！请检查CSV格式")
    print(f"从CSV读取到 {len(raw_genes)} 个候选高变基因（未筛选）")
    
    # 核心修改：调用筛选函数剔除核糖体/线粒体基因
    filtered_genes = filter_ribo_mt_genes(raw_genes)
    
    return filtered_genes

def plot_validate_volcano(df_all_genes, key_genes, source_cell, target_cell,
                          log2fc_cutoff=1, pvalue_cutoff=0.05,
                          save_dir='./', save_name='validate_volcano.png'):
    """
    绘制验证型火山图：仅高亮显著区域的候选基因，增大字号和点大小
    非显著区域的HVG统一用灰色，不强调、不标注
    按垂直距离（y轴）分层抽样10个，保证分散、不集中
    新增：同时保存PNG和PDF格式到同一文件夹
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    
    # 生成PDF保存路径（替换后缀为pdf，保持文件名前缀一致）
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    
    # 1. 计算 -log10(P)
    df_all_genes['neg_log10_pvalue'] = -np.log10(df_all_genes['pvalue'])
    
    # 2. 标记显著/不显著
    df_all_genes['category'] = 'not significant'
    df_all_genes.loc[(df_all_genes['log2FC'] > log2fc_cutoff) & 
                     (df_all_genes['pvalue'] < pvalue_cutoff), 'category'] = 'up (target vs source)'
    df_all_genes.loc[(df_all_genes['log2FC'] < -log2fc_cutoff) & 
                     (df_all_genes['pvalue'] < pvalue_cutoff), 'category'] = 'down (target vs source)'
    
    # 3. 标记是否是CSV里的HVG
    df_all_genes['is_key_gene'] = df_all_genes['gene'].isin(key_genes)
    # 核心：只保留【显著 + HVG】的交集
    key_significant = df_all_genes[df_all_genes['is_key_gene'] & (df_all_genes['category'] != 'not significant')].copy()

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = {
        'up (target vs source)': '#e74c3c',
        'down (target vs source)': '#3498db',
        'not significant': '#95a5a6'
    }

    # ===================== 画所有背景点 =====================
    for cat in ['not significant', 'up (target vs source)', 'down (target vs source)']:
        subset = df_all_genes[(df_all_genes['category'] == cat) & (~df_all_genes['is_key_gene'])]
        ax.scatter(subset['log2FC'], subset['neg_log10_pvalue'],
                   c=colors[cat], alpha=0.6, s=85)
    
    # ===================== 画显著HVG（黄色+黑圈）=====================
    ax.scatter(key_significant['log2FC'], key_significant['neg_log10_pvalue'],
               c='gold', s=90, edgecolors='black', linewidth=2, alpha=0.95,
               label='Significant HVGs')

    # ===================== 核心：按垂直距离分层抽10个，保证分散 =====================
    np.random.seed(42)
    n_select = 10

    if len(key_significant) == 0:
        print("没有显著HVG可标注")
    else:
        # 按 y 轴从大到小排序（从上到下）
        ks = key_significant.sort_values('neg_log10_pvalue', ascending=False).reset_index(drop=True)
        total = len(ks)

        # 等间距取索引，保证垂直方向均匀
        # 例子：共50个 → 取 0,5,10,15,20,25,30,35,40,45 → 均匀分布
        step = max(1, total // n_select)
        selected_idx = np.arange(0, total, step)[:n_select]
        selected = ks.iloc[selected_idx].copy()

        print(f"均匀挑选的 {len(selected)} 个基因（按y轴分散）:")
        print(selected['gene'].tolist())

        # 标注：智能左右对齐，不重叠
        offset = 0.2
        for i, (_, row) in enumerate(selected.iterrows()):
            if row['log2FC'] < 0:
                ha = 'right'   # 左边点 → 文字放右侧
                x_pos=row['log2FC']-0.5
            else:
                ha = 'left'  # 右边点 → 文字放左侧
                x_pos=row['log2FC']+0.5

            # 轻微上下错开，避免同高度重叠
            y_pos = row['neg_log10_pvalue'] + (i % 2 - 0.5) * offset

            ax.text(x_pos, y_pos, row['gene'],
                    fontsize=14, fontweight='bold', ha=ha, va='center', color='black')

    # ===================== 样式 =====================
    ax.set_xlabel('log2(Fold Change) (target vs source)', fontsize=18, fontweight='bold')
    ax.set_ylabel('-log10(P Value)', fontsize=18, fontweight='bold')
    ax.set_title(f'Validation of HVGs: {source_cell} vs {target_cell}',
                 fontsize=20, pad=25, fontweight='bold')

    for label in ax.get_xticklabels():
        label.set_weight('bold')
        label.set_fontsize(16)
    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_fontsize(16)

    ax.axvline(x=log2fc_cutoff, color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=-log2fc_cutoff, color='gray', linestyle='--', linewidth=2)
    ax.axhline(y=-np.log10(pvalue_cutoff), color='gray', linestyle='--', linewidth=2)

    legend = ax.legend(fontsize=14, frameon=True, loc='upper right')
    for text in legend.get_texts():
        text.set_weight('bold')

    ax.grid(True, alpha=0.2, lw=1)
    plt.tight_layout()
    
    # 保存PNG（原有逻辑不变）
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    print(f"\nPNG格式火山图已保存至: {save_path}")
    
    # 新增：保存PDF格式（矢量图，无失真）
    plt.savefig(save_path_pdf, bbox_inches='tight', format='pdf')
    print(f"PDF格式火山图已保存至: {save_path_pdf}")
    
    plt.show()     

# -------------------------- 主程序（只需修改以下参数） --------------------------
if __name__ == '__main__':
    # ========== 请修改以下6个核心参数 ==========
    # 1. 文件路径
    h5ad_file = "UNAGI/data/DentateGyrus/Processed_2000_epoch_100/processed_simp_10X43_1_2000.h5ad"       # 你的h5ad文件路径
    csv_file = "UNAGI/data/DentateGyrus/Processed_2000_epoch_100/result_plot/noNormaliz_2p5/heatmap_csv_0.05/top0.05_processed_recons_feature_latent_dim_23_diff_Time0.csv_making.csv"         # 你的CSV文件路径
    save_dir = "UNAGI/data/DentateGyrus/Processed_2000_epoch_100/plot_for_figure/volcano_plots"    # 图片保存文件夹
    
    # 2. 细胞类型相关（核心！需根据你的h5ad修改）
    cell_type_col = "clusters"  # h5ad中存储细胞类型的列名（如cell_type/louvain/leiden）
    source_cell = "nIPC"       # 源细胞类型名称（如'神经元'/'0'，需匹配h5ad中的值）
    target_cell = "Neuroblast"    # 目标细胞类型名称（如'星形胶质细胞'/'1'，需匹配h5ad中的值）
    
    # 3. 差异阈值（可选调整）
    log2fc_cutoff = 1      # log2FC阈值（1=2倍差异，1.5=1.5倍）
    pvalue_cutoff = 0.05   # P值阈值
    
    # ========== 执行流程 ==========
    # 1. 读取h5ad并针对指定细胞类型做差异分析
    df_all = load_h5ad_data(
        h5ad_path=h5ad_file,
        cell_type_col=cell_type_col,
        source_cell=source_cell,
        target_cell=target_cell
    )
    # 2. 读取CSV候选高变基因 + 自动剔除核糖体/线粒体基因
    key_genes = load_csv_genes(csv_file)
    # 3. 绘制验证型火山图
    plot_validate_volcano(
        df_all_genes=df_all,
        key_genes=key_genes,
        source_cell=source_cell,
        target_cell=target_cell,
        log2fc_cutoff=log2fc_cutoff,
        pvalue_cutoff=pvalue_cutoff,
        save_dir=save_dir,
        save_name='hvg_validation_source_vs_target_dim_23_emphasis_fixed.png'  # 文件名标注修复版
    )