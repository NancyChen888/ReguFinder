import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import anndata as ad
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 尝试导入CUDA加速库（优先GPU计算）
# ----------------------
try:
    import cupy as cp
    from cupyx.scipy.stats import spearmanr as cuda_spearmanr
    CUDA_AVAILABLE = True
    print("✅ CUDA可用，将使用GPU加速Spearman计算")
except ImportError:
    from scipy.stats import spearmanr
    CUDA_AVAILABLE = False
    print("⚠️ CUDA不可用，使用CPU计算Spearman")

# ----------------------
# 1. 配置与文件路径（请修改为你的实际路径）
# ----------------------
# 核心修改：适配6列TF-target格式（小鼠/人类切换只需改这个路径）
KEY_DIM="intersection"
TF_TARGET_TSV_PATH = "UNAGI/TF_target_DB/mouse_TF_Target.txt"  # ← 改你的6列数据路径
GENE_CSV_PATH = f"UNAGI/data/GSE171993_Hep/plot_for_figure/Venn/common_regulators.csv"
H5AD_PATH = "UNAGI/data/GSE171993_Hep/Hep_fil_addSim.h5ad"
OUTPUT_FOLDER = "UNAGI/data/GSE171993_Hep/plot_for_figure/gene_network_plots"

# 文件名区分小鼠/人类（可自定义）
SPECIES = "mouse"  # ← 切换为"human"即可适配人类数据
PDF_FILENAME = f"dim_{KEY_DIM}_GRN_top30_no_selfloop.pdf"
PNG_FILENAME = f"dim_{KEY_DIM}_GRN_top30_no_selfloop.png"
# 新增：边信息保存文件名
EDGE_INFO_FILENAME = f"dim_{KEY_DIM}_GRN_top30_no_selfloop.csv"

# 配色优化：按Target_type（TF/Gene）区分节点颜色
COLOR_PALETTE = {
    'tf': '#E64B35',          # 源TF节点（红色）
    'target_tf': '#9B2226',   # 靶基因是TF（深红外）
    'target_gene': '#4DBBD5', # 靶基因是普通Gene（蓝色）
    'edge_default': '#28A745' # 边颜色（绿色，6列格式无调控方向，统一配色）
}

FILTER_KEYWORDS = ['^MT-', '^MRP', '^RPL', '^RPS', 'mitochondria', 'ribosomal']

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
PDF_SAVE_PATH = os.path.join(OUTPUT_FOLDER, PDF_FILENAME)
PNG_SAVE_PATH = os.path.join(OUTPUT_FOLDER, PNG_FILENAME)
# 新增：边信息保存路径
EDGE_INFO_SAVE_PATH = os.path.join(OUTPUT_FOLDER, EDGE_INFO_FILENAME)

# ----------------------
# 2. 关键基因过滤
# ----------------------
def filter_ribo_mito_genes(gene_list):
    import re
    filtered = []
    for g in gene_list:
        if not any(re.search(p, g, re.IGNORECASE) for p in FILTER_KEYWORDS):
            filtered.append(g)
    return filtered

raw_key_genes = pd.read_csv(GENE_CSV_PATH).iloc[:,0].tolist()
filtered_key_genes = filter_ribo_mito_genes(raw_key_genes)

print(f"原始基因：{len(raw_key_genes)}")
print(f"过滤核糖体/线粒体后：{len(filtered_key_genes)}")

# ----------------------
# 3. 读取6列TF-target数据（核心适配）
# ----------------------
# 读取6列数据，指定列名（匹配标准格式）
edges_df = pd.read_csv(
    TF_TARGET_TSV_PATH, 
    sep="\t", 
    header=None,  # 6列数据无表头
    names=['TF', 'TF_ID', 'Target', 'Target_ID', 'TF_Type', 'Target_Type']  # 6列对应名称
)

# 核心筛选1：只保留关键基因的调控关系
edges_df = edges_df[
    (edges_df['TF'].isin(filtered_key_genes)) & 
    (edges_df['Target'].isin(filtered_key_genes))
]

# 核心筛选2：排除自环（TF和Target相同的情况）
edges_df = edges_df[edges_df['TF'] != edges_df['Target']]
print(f"排除自环后调控关系数：{len(edges_df)}")

# 数据清洗：过滤空值、重复边
edges_df = edges_df.dropna(subset=['TF', 'Target'])
edges_df = edges_df.drop_duplicates(subset=['TF', 'Target'])  # 按TF-Target去重

print(f"筛选后有效调控关系数：{len(edges_df)}")
print(f"靶基因类型分布：\n{edges_df['Target_Type'].value_counts()}")

if len(edges_df) == 0:
    raise ValueError("无有效调控关系，请检查基因列表或数据路径！")

# ----------------------
# 4. 读取表达矩阵（与原代码一致，保留稳定性修复）
# ----------------------
adata = ad.read_h5ad(H5AD_PATH)
adata.var_names = adata.var_names.astype(str)
valid_genes = [g for g in filtered_key_genes if g in adata.var_names]

if len(valid_genes) == 0:
    raise ValueError("无有效基因在表达矩阵中！")

# 提取表达矩阵并处理异常值
expr_mat = adata[:, valid_genes].X
expr_mat = expr_mat.toarray() if hasattr(expr_mat, 'toarray') else expr_mat
expr_mat = np.nan_to_num(expr_mat, nan=0.0, posinf=0.0, neginf=0.0)
expr_df = pd.DataFrame(expr_mat, columns=valid_genes)

# ----------------------
# 5. 计算Spearman权重（保留CUDA加速+异常处理）
# ----------------------
def calc_spearman(tf, target, df):
    try:
        x = df[tf].values
        y = df[target].values
        
        # 过滤常量基因对（避免计算错误）
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return 0.1
        
        if CUDA_AVAILABLE:
            cx, cy = cp.array(x), cp.array(y)
            corr, _ = cuda_spearmanr(cx, cy)
            w = float(cp.abs(corr))
        else:
            corr, _ = spearmanr(x, y)
            w = abs(corr)
        
        # 替换NaN/无穷大
        w = 0.1 if np.isnan(w) or np.isinf(w) else w
        return w
    except Exception as e:
        print(f"计算{tf}-{target}权重出错：{e}，使用默认值0.1")
        return 0.1

# 批量计算权重
tqdm.pandas(desc="计算边权重（Spearman相关系数）")
edges_df['weight'] = edges_df.progress_apply(
    lambda r: calc_spearman(r.TF, r.Target, expr_df), axis=1
)

# 核心筛选3：按权重降序排列，取前30条
edges_df = edges_df.sort_values(by='weight', ascending=False).head(30)
print(f"筛选权重前30条调控关系后：{len(edges_df)}")

# 权重归一化（防除0）
weight_min = edges_df['weight'].min()
weight_max = edges_df['weight'].max()
if weight_max - weight_min < 1e-6:
    edges_df['weight_norm'] = 0.5
else:
    edges_df['weight_norm'] = (edges_df.weight - weight_min) / (weight_max - weight_min)

# ----------------------
# 6. 构建有向图（保留Target_Type属性）
# ----------------------
G = nx.from_pandas_edgelist(
    edges_df,
    source='TF',
    target='Target',
    edge_attr=['weight', 'weight_norm'],
    create_using=nx.DiGraph()
)

# 给节点添加Target_Type属性（用于配色）
node_type_dict = {}
for _, row in edges_df.iterrows():
    # 源节点标记为TF
    node_type_dict[row['TF']] = 'source_tf'
    # 靶节点标记为对应的类型（TF/Gene）
    node_type_dict[row['Target']] = row['Target_Type']

# 过滤孤立节点
G.remove_nodes_from(list(nx.isolates(G)))
print(f"过滤孤立节点后，剩余节点数：{len(G.nodes())}")

if len(G.nodes()) < 2:
    raise ValueError("节点数过少，无法绘制网络图！")

# ----------------------
# 新增：准备并保存边信息（from/to/weight）
# ----------------------
# 提取from、to、weight三列数据
edge_info_df = edges_df[['TF', 'Target', 'weight']].copy()
# 重命名列名为from、to、weight
edge_info_df.columns = ['Source', 'Target', 'Weight']
# 保存为CSV文件
edge_info_df.to_csv(EDGE_INFO_SAVE_PATH, index=False, encoding='utf-8')
print(f"\n📄 边信息已保存至：{EDGE_INFO_SAVE_PATH}")
print(f"📈 保存的边信息行数：{len(edge_info_df)}")

# ----------------------
# 7. 绘图（适配6列格式的节点/边配色）
# ----------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 画布参数（保留稳定性修复）
fig, ax = plt.subplots(figsize=(12, 10), dpi=200)

# 布局优化（避免数值溢出）
np.random.seed(42)
pos = nx.spring_layout(
    G, 
    k=0.8,
    iterations=50,
    seed=42,
    scale=10
)

# 修复布局坐标异常值
for node in pos:
    pos[node] = np.nan_to_num(pos[node], nan=0.0, posinf=10.0, neginf=-10.0)

# 节点配色：按类型区分（核心适配）
node_color = []
for node in G.nodes():
    if node_type_dict.get(node) == 'source_tf':
        # 源TF节点
        node_color.append(COLOR_PALETTE['tf'])
    elif node_type_dict.get(node) == 'TF':
        # 靶节点是TF
        node_color.append(COLOR_PALETTE['target_tf'])
    else:
        # 靶节点是普通Gene
        node_color.append(COLOR_PALETTE['target_gene'])

node_size = 1500

# 边配置：6列格式无调控方向，统一配色+按权重控粗细
edge_colors = [COLOR_PALETTE['edge_default'] for _ in G.edges()]
edge_widths = []
for u, v, d in G.edges(data=True):
    width = d['weight_norm'] * 3 + 0.5
    width = np.clip(width, 0.5, 4.0)
    edge_widths.append(width)

# 绘制边（保留稳定性修复）
nx.draw_networkx_edges(
    G, pos, ax=ax,
    arrowstyle='->',
    arrowsize=15,
    edge_color=edge_colors,
    width=edge_widths,
    alpha=0.8,
    min_source_margin=5,
    min_target_margin=5
)

# 绘制节点
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_color,
    node_size=node_size,
    edgecolors='white',
    linewidths=2.5,
    alpha=0.98
)

# 绘制标签
nx.draw_networkx_labels(
    G, pos, ax=ax,
    font_size=10,
    font_weight='bold',
    bbox=dict(facecolor='white', alpha=0.9, pad=1.5)
)

# ----------------------
# 图例（适配6列格式的节点类型）
# ----------------------
legend_elements = [
    Line2D([0],[0],marker='o',color='w',markersize=12,label='Source TF (转录因子)',
           markerfacecolor=COLOR_PALETTE['tf'], markeredgecolor='white', markeredgewidth=2.5),
    Line2D([0],[0],marker='o',color='w',markersize=12,label='Target TF (靶TF)',
           markerfacecolor=COLOR_PALETTE['target_tf'], markeredgecolor='white', markeredgewidth=2.5),
    Line2D([0],[0],marker='o',color='w',markersize=12,label='Target Gene (普通靶基因)',
           markerfacecolor=COLOR_PALETTE['target_gene'], markeredgecolor='white', markeredgewidth=2.5),
    Line2D([0],[0],color=COLOR_PALETTE['edge_default'], lw=4, label='Regulatory Edge (调控边)', alpha=0.8),
    Line2D([0],[0],color='gray', lw=1, label='Edge Width: Normalized Spearman Correlation'),
    Line2D([0],[0],color='gray', lw=5, label='(0=Thin, 1=Thick)', alpha=0.5)
]

ax.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=10,
    frameon=True,
    facecolor='white',
    framealpha=0.98
)

# 隐藏坐标轴
ax.axis('off')
plt.tight_layout(pad=1.0)

# ----------------------
# 8. 保存文件（保留稳定性修复）
# ----------------------
plt.savefig(
    PDF_SAVE_PATH,
    format='pdf',
    bbox_inches='tight',
    dpi=300,
    facecolor='white',
    edgecolor='none'
)

plt.savefig(
    PNG_SAVE_PATH,
    format='png',
    bbox_inches='tight',
    dpi=800,
    facecolor='white',
    edgecolor='none'
)

plt.show()

# ----------------------
# 9. 统计信息（适配6列格式）
# ----------------------
print("\n📊 最终网络统计（6列TF-target格式，TOP30权重+无自环）：")
print(f"- 物种：{SPECIES}")
print(f"- 总节点数：{len(G.nodes())}")
print(f"- 总调控边数：{len(G.edges())}")
print(f"- 节点类型分布：")
type_count = {}
for node in G.nodes():
    t = node_type_dict.get(node)
    type_count[t] = type_count.get(t, 0) + 1
for t, cnt in type_count.items():
    if t == 'source_tf':
        print(f"  → 源TF节点：{cnt}")
    elif t == 'TF':
        print(f"  → 靶TF节点：{cnt}")
    else:
        print(f"  → 普通靶基因节点：{cnt}")
print(f"- Spearman权重范围：{edges_df['weight'].min():.3f} ~ {edges_df['weight'].max():.3f}")
print(f"- PDF文件路径：{PDF_SAVE_PATH}")
print(f"- PNG文件路径：{PNG_SAVE_PATH}")
print(f"- 边信息文件路径：{EDGE_INFO_SAVE_PATH}")