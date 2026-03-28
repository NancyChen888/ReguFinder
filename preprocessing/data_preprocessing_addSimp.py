import scanpy as sc

# 读取数据
adata = sc.read('UNAGI/data/DentateGyrus/Processed_1944_ep_100/10X43_1_filtered.h5ad')

# 打印原始信息
print("原始 AnnData 对象信息:")
print(adata)
print("\n原始观测值列名:", adata.obs.columns.tolist())
print(adata.obs['clusters'])
print("\n原始细胞类别:", adata.obs['clusters'])

# 检查并添加'name.simple'字段
if 'name.simple' not in adata.obs.columns:
    adata.obs['name.simple'] = adata.obs['clusters'].copy()
    print("已添加 'name.simple' 字段")
else:
    print("'name.simple' 字段已存在")

# # 检查并添加'name.simple'字段
# if 'stages' not in adata.obs.columns:
#     adata.obs['stages'] = adata.uns['Phenotype'].copy()
#     print("已添加 'stages' 字段")
# else:
#     print("'stages' 字段已存在")

# 指定保存路径和文件名（可根据需要修改）
output_path = 'UNAGI/data/DentateGyrus/Processed_1944_ep_100/10X43_1_filtered_addSim.h5ad'

# 保存为新的h5ad文件
adata.write(output_path)
print(f"已将处理后的adata保存至: {output_path}")