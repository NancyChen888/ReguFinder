import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pathlib
import torch

# 1. 路径配置（保持原逻辑不变）
current_script_path = os.path.abspath(__file__)
location_script_path=os.path.dirname(current_script_path)
pyscript_dir = os.path.dirname(location_script_path)
tutorials_dir = os.path.dirname(pyscript_dir)
root_dir = os.path.dirname(tutorials_dir)
sys.path.append(root_dir)
sys.path.append("/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI")

# 使用pathlib获取项目根目录
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))
print(f"项目根目录: {project_root}")
print(f"当前 sys.path: {sys.path[:5]}")  # 打印前5个路径，避免输出过长


import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pathlib
import torch
import scanpy as sc
from UNAGI import UNAGI  # 导入原始UNAGI类
from UNAGI.utils.attribute_utils import split_dataset_into_stage
import pandas as pd

# 1. 重写UNAGI类，修改setup_data方法以支持跳过目录检查
class UNAGIWithResume(UNAGI):
    def setup_data(self, data_path, stage_key, total_stage, 
                  gcn_connectivities=False, neighbors=50, threads=20,
                  skip_existing_dirs=False):  # 新增参数：是否跳过已存在目录检查
        """
        重写setup_data方法，添加skip_existing_dirs参数
        当skip_existing_dirs=True时，跳过目录存在性检查，不创建新目录
        """
        if total_stage < 2:
            raise ValueError('The total number of stages should be larger than 1')
        
        if os.path.isfile(data_path):
            self.data_folder = os.path.dirname(data_path)
        else:
            self.data_folder = data_path
        self.stage_key = stage_key

        # 检查数据是否已按阶段拆分
        if os.path.exists(os.path.join(self.data_folder, '0.h5ad')):
            temp = sc.read(os.path.join(self.data_folder, '0.h5ad'))
            self.input_dim = temp.shape[1]
            if 'gcn_connectivities' not in list(temp.obsp.keys()):
                gcn_connectivities = False
            else:
                gcn_connectivities = True
        else:
            print('The dataset is not splited into stages, splitting now...')
            self.data_path = data_path
            self.input_dim = split_dataset_into_stage(self.data_path, self.data_folder, self.stage_key)
            gcn_connectivities = False

        if os.path.isfile(data_path):
            self.data_path = os.path.dirname(data_path)
        else:
            self.data_path = data_path
        self.data_path = os.path.join(self.data_path, '0.h5ad')
        self.data_folder = os.path.dirname(self.data_path)
        self.ns = total_stage

        # 关键修改：仅当不跳过检查且目录存在时才报错
        if not skip_existing_dirs:
            if os.path.exists(os.path.join(self.data_folder, '0')):
                raise ValueError('The iteration 0 folder is already existed, please remove it or use skip_existing_dirs=True')
            if os.path.exists(os.path.join(self.data_folder, '0/stagedata')):
                raise ValueError('The iteration 0/stagedata folder is already existed, please remove it or use skip_existing_dirs=True')
            if os.path.exists(os.path.join(self.data_folder, 'model_save')):
                raise ValueError('The model_save folder is already existed, please remove it or use skip_existing_dirs=True')
            
            # 创建目录
            dir1 = os.path.join(self.data_folder, '0')
            dir2 = os.path.join(self.data_folder, '0/stagedata')
            dir3 = os.path.join(self.data_folder, 'model_save')
            os.makedirs(dir1, exist_ok=True)
            os.makedirs(dir2, exist_ok=True)
            os.makedirs(dir3, exist_ok=True)
        else:
            # 跳过检查时，仅确认目录存在（如果不存在则创建，避免后续报错）
            print("Skipping directory existence check (skip_existing_dirs=True)")
            os.makedirs(os.path.join(self.data_folder, '0'), exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, '0/stagedata'), exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, 'model_save'), exist_ok=True)

        # 计算细胞图（如果需要）
        if not gcn_connectivities:
            print(f'Calculating cell graphs with K={neighbors}, threads={threads}')
            self.calculate_neighbor_graph(neighbors, threads)
        else:
            print('Cell graphs found, skipping calculation')

# 2. 路径配置
current_script_path = os.path.abspath(__file__)
pyscript_dir = os.path.dirname(current_script_path)
tutorials_dir = os.path.dirname(pyscript_dir)
root_dir = os.path.dirname(tutorials_dir)
sys.path.append(root_dir)
sys.path.append("/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI")

current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 3. 配置参数
DATA_FOLDER = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100"
IDREM_DIR = "/home/lijunxian/NancyRentedPlace/UNAGI/idrem/"
MAX_ITER = 3
TOTAL_STAGE = 2

# 4. 判断是否需要跳过训练
def is_training_completed(data_folder, max_iter):
    required_dirs = [
        os.path.join(data_folder, "model_save"),
        os.path.join(data_folder, f"{max_iter-1}/stagedata")
    ]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Missing directory: {dir_path}")
            return False
    return True

# 5. 初始化UNAGI实例（使用重写的类）
unagi = UNAGIWithResume()

# 6. 调用setup_data，根据是否训练完成决定是否跳过目录检查
training_completed = is_training_completed(DATA_FOLDER, MAX_ITER)
unagi.setup_data(
    data_path=DATA_FOLDER,
    total_stage=TOTAL_STAGE,
    stage_key='age(days)',
    gcn_connectivities=True,  # 如果细胞图已计算，设为True
    neighbors=50,
    threads=20,
    skip_existing_dirs=training_completed  # 关键：训练完成则跳过目录检查
)

# 7. 配置训练参数
unagi.setup_training(
    task='DentateGyrus',
    dist='ziln',
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    GPU=torch.cuda.is_available(),
    epoch_iter=5,
    epoch_initial=1,
    max_iter=MAX_ITER,
    BATCHSIZE=560
)

# 8. 执行训练（已完成则跳过）
if training_completed:
    print("Training already completed, skipping...")
    unagi.run_UNAGI(
        idrem_dir=IDREM_DIR,
        CPO=True,
        resume=True,
        resume_iteration=MAX_ITER
    )
else:
    print("Starting training...")
    unagi.run_UNAGI(
        idrem_dir=IDREM_DIR,
        CPO=True,
        resume=False
    )


# # 9. 执行后续函数
# minus_matrix_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/Tocons_matrix.csv"
# unagi.matrix_subtraction(
#     ori_emb_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/embeddings/latent_embeddings_time1.csv",
#     perturbed_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/Perturb_output/12h_to_96h/feat_54_mul_7.0_embed.csv",
#     output_path=minus_matrix_path
# )

#----------------------------------------------
# 利用原有的重构矩阵的函数进行测试
#----------------------------------------------
# save_path_unagi_dentate=r"/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/recons_matrx_exiFunc1.csv"
# # 调用函数处理处理后的文件
# unagi.test_for_recons_matrx(
#     ori_adata_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/1.h5ad",
#     iteration=2,
#     save_path=save_path_unagi_dentate
# )
# print(f"完成处理: {save_path_unagi_dentate}\n")

#----------------------------------------------
# 将原本的潜在空间输入，查看重构之后的矩阵
#----------------------------------------------
# # 保存重构结果的基本文件夹
# base_save_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/test_for_recons/recons_exiFun_noAdj"
# os.makedirs(base_save_path, exist_ok=True)

# # 临时文件夹用于保存移除Cell_type列后的文件
# temp_dir = os.path.join(base_save_path, "temp_processed")
# os.makedirs(temp_dir, exist_ok=True)

# # 指定要处理的单个CSV文件路径
# # 请将下面的文件路径替换为你需要处理的具体文件
# target_file = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/training_set/all_embeddings_with_celltype.csv"

# # 检查文件是否存在且是CSV文件
# if not os.path.exists(target_file):
#     raise FileNotFoundError(f"指定的文件不存在: {target_file}")
# if not target_file.endswith('.csv'):
#     raise ValueError(f"指定的文件不是CSV文件: {target_file}")

# # 读取CSV文件
# df = pd.read_csv(target_file)

# # 提取文件名（不包含路径）
# filename = os.path.basename(target_file)

# # 检查并移除Cell_type列
# if "cell_type" in df.columns:
#     df = df.drop(columns=["cell_type"])
#     print(f"已移除文件 {filename} 中的cell_type列")
# else:
#     print(f"文件 {filename} 中未找到cell_type列，无需移除")

# # 保存处理后的临时文件
# temp_file_path = os.path.join(temp_dir, filename)
# df.to_csv(temp_file_path, index=False)

# # 构建保存路径和文件名
# save_filename = f"recons_{os.path.splitext(filename)[0]}.csv"
# save_path = os.path.join(base_save_path, save_filename)

# print(f"处理文件: {filename}")
# print(f"保存路径: {save_path}")

# # 调用函数处理处理后的文件
# unagi.save_perturbed_reconstruction_withGCN(
#     ori_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/processed_simp_10X43_1_2000.h5ad",
#     adata_path=temp_file_path,  # 使用处理后的临时文件
#     iteration=2,
#     save_path=save_path
# )

# # # 调用函数处理处理后的文件
# # unagi.test_for_recons_matrx(
# #     ori_adata_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/processed_simp_10X43_1_2000.h5ad",
# #     iteration=2,
# #     save_path=save_path
# # )

# print(f"完成处理: {filename}\n")
    

#----------------------------------------------
# 循环处理整个文件夹下的潜在空间向量
#----------------------------------------------

# # 保存重构结果的基本文件夹
# base_save_path = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/recons_matrx/2p5/time1_GCN"
# os.makedirs(base_save_path, exist_ok=True)

# # 临时文件夹用于保存移除Cell_type列后的文件
# temp_dir = os.path.join(base_save_path, "temp_processed")
# os.makedirs(temp_dir, exist_ok=True)

# minus_matrix_savedir = "/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/minus_embed_all_feat/diff/multiplier_2p5/2p5_time1"

# for filename in os.listdir(minus_matrix_savedir):
#     if filename.endswith('.csv'):
#         # 构建完整的CSV文件路径
#         adata_path = os.path.join(minus_matrix_savedir, filename)
        
#         # try:
#         # 读取CSV文件
#         df = pd.read_csv(adata_path)
        
#         # 检查并移除Cell_type列
#         if "cell_type" in df.columns:
#             df = df.drop(columns=["cell_type"])
#             print(f"已移除文件 {filename} 中的cell_type列")
#         else:
#             print(f"文件 {filename} 中未找到cell_type列，无需移除")
        
#         # 保存处理后的临时文件
#         temp_file_path = os.path.join(temp_dir, filename)
#         df.to_csv(temp_file_path, index=False)
        
#         # 构建保存路径和文件名
#         save_filename = f"recons_{os.path.splitext(filename)[0]}.csv"
#         save_path = os.path.join(base_save_path, save_filename)
        
#         print(f"处理文件: {filename}")
#         print(f"保存路径: {save_path}")
        
#         # 调用函数处理处理后的文件
#         recons_matrx=unagi.save_perturbed_reconstruction_withGCN(
#             ori_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/1.h5ad",
#             adata_path=temp_file_path,  # 使用处理后的临时文件
#             iteration=2,
#             save_path=save_path
#         )
#         # # 调用函数处理处理后的文件
#         # unagi.test_for_recons_matrx(
#         #     ori_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/processed_simp_10X43_1_2000.h5ad",
#         #     iteration=2,
#         #     save_path=save_path
#         # )
#         print(f"完成处理: {filename}\n")
            
#         # except Exception as e:
#         #     print(f"处理文件 {filename} 时出错: {str(e)}")


#----------------------------------------------
# 循环处理整个文件夹下的潜在空间向量(修复index,CellType, 列名等问题)
#----------------------------------------------
# 保存重构结果的基本文件夹
base_save_path = "models_compare/UNAGI/construct/time0"
os.makedirs(base_save_path, exist_ok=True)

# 临时文件夹用于保存移除Cell_type列后的文件
temp_dir = os.path.join(base_save_path, "temp_processed")
os.makedirs(temp_dir, exist_ok=True)

minus_matrix_savedir = "models_compare/UNAGI/embedding/time0"

# 读取原始数据以获取基因名（假设从h5ad文件中读取）
import scanpy as sc
print("读取原始数据以获取基因名称...")
ori_adata_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/DentateGyrus/Processed_2000_epoch_100/0.h5ad"
ori_adata = sc.read(ori_adata_path)
gene_names = ori_adata.var_names.tolist()  # 获取基因名称列表
print(f"成功获取 {len(gene_names)} 个基因名称")

for filename in os.listdir(minus_matrix_savedir):
    if filename.endswith('.csv'):
        # 构建完整的CSV文件路径
        adata_path = os.path.join(minus_matrix_savedir, filename)
        
        # 读取CSV文件
        df = pd.read_csv(adata_path)
        
        # 保存Cell_Type列（如果存在）以便后续恢复
        # 同时保存其原始索引，确保后续能正确对齐
        if 'Cell_Type' in df.columns:
            cell_type_column = df[['index', 'Cell_Type']].copy()  # 同时保留index列用于对齐
            print(f"文件 {filename} 中找到Cell_Type列，已保存")
        else:
            cell_type_column = None
            print(f"警告: 文件 {filename} 中未找到Cell_Type列")
        
        # 检查并移除Cell_type列和可能存在的其他不需要的列
        columns_to_drop = []
        if "cell_type" in df.columns:
            columns_to_drop.append("cell_type")
        if "Cell_Type" in df.columns:
            columns_to_drop.append("Cell_Type")
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"已移除文件 {filename} 中的 {columns_to_drop} 列")
        else:
            print(f"文件 {filename} 中未找到需要移除的列")
        
        # 保存处理后的临时文件
        temp_file_path = os.path.join(temp_dir, filename)
        df.to_csv(temp_file_path, index=False)
        
        # 构建保存路径和文件名
        save_filename = f"recons_{os.path.splitext(filename)[0]}.csv"
        save_path = os.path.join(base_save_path, save_filename)
        
        print(f"处理文件: {filename}")
        print(f"保存路径: {save_path}")
        
        # 调用函数处理处理后的文件
        recons_matrx = unagi.save_perturbed_reconstruction_withGCN(
            ori_data_path=ori_adata_path,
            adata_path=temp_file_path,  # 使用处理后的临时文件
            iteration=2,
            save_path=save_path
        )
        
        # 处理返回的重构矩阵
        if recons_matrx is not None:
            # 1. 将重构矩阵转换为DataFrame
            recons_df = pd.DataFrame(recons_matrx)
            
            # 2. 使用原始CSV文件中的index作为索引
            original_df = pd.read_csv(adata_path)
            recons_df.index = original_df["index"]  # 仅设置索引，不添加index列
            
            # 3. 将列名替换为基因名
            if len(gene_names) == recons_df.shape[1]:  # 不需要减去index列了
                recons_df.columns = gene_names
                print(f"已将 {filename} 的重构矩阵列名替换为基因名")
            else:
                print(f"警告: 基因名数量({len(gene_names)})与重构矩阵列数({recons_df.shape[1]})不匹配，无法替换列名")
            
            # 4. 将之前保存的Cell_Type列加回来（使用索引匹配）
            if cell_type_column is not None:
                # 将cell_type_column也设置为index索引，然后用join合并
                cell_type_column = cell_type_column.set_index('index')
                recons_df = recons_df.join(cell_type_column, how='left')
                print(f"已为 {filename} 的重构矩阵添加Cell_Type列")
            
            # 5. 保存处理后的重构矩阵
            final_save_path = os.path.join(base_save_path, f"processed_{save_filename}")
            recons_df.to_csv(final_save_path)  # 保存时会自动包含索引
            print(f"已保存处理后的重构矩阵: {final_save_path}")
        else:
            print(f"警告: 未获取到 {filename} 的重构矩阵，跳过后续处理")
        
        print(f"完成处理: {filename}\n")

# # 10. 保存扰动重构结果
# unagi.save_perturbed_reconstruction(
#     iteration=MAX_ITER - 1,
#     ori_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/Stage_divided_h5ad_cells_genes/1.h5ad",
#     adata_path=minus_matrix_path,
#     save_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/construction_matrix/recons_matrix.csv"
# )

print("All tasks completed!")
