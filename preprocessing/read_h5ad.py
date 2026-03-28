import scanpy as sc

def read_and_print_h5ad(file_path):
    """
    读取h5ad文件并打印其内容
    
    参数:
    file_path (str): h5ad文件的路径
    """
    try:
        # 读取h5ad文件
        adata = sc.read_h5ad(file_path)
        
        print((adata.obsm['X_pca'].shape))
        print("成功读取h5ad文件!")
        print("\n===== AnnData对象基本信息 =====")
        print(adata)
        
        print("\n===== 观测值(obs)信息 =====")
        print("观测值数量:", adata.n_obs)
        print("观测值列名:", list(adata.obs.columns))
        print("前5行观测值数据:")
        print(adata.obs.head())
        print(adata.obs.index)
        
        print("\n===== 特征值(var)信息 =====")
        print("特征值数量:", adata.n_vars)
        print("特征值列名:", list(adata.var.columns))
        print("前5行特征值数据:")
        print(adata.var.head())
        print(adata.obs['time_info'])
        
        print("\n===== 主要数据矩阵(X)信息 =====")
        print("数据矩阵形状:", adata.X.shape)
        print("数据矩阵类型:", type(adata.X))
            
        # 如果有观测值的注释信息，也打印出来
        if adata.obs_names is not None:
            print("\n前5个观测值名称:")
            print(adata.obs_names[:5].tolist())
            
        # 如果有特征值的注释信息，也打印出来
        if adata.var_names is not None:
            print("\n前5个特征值名称:")
            print(adata.var_names[:5].tolist())
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")

if __name__ == "__main__":
    # 替换为你的h5ad文件路径
    h5ad_file_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/Larry_hematopoiesis/LARRY_data/LARRY_adata_preprocessed.h5ad"
    read_and_print_h5ad(h5ad_file_path)
    