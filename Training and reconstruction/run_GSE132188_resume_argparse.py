import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pathlib
import torch
import scanpy as sc
import pandas as pd
import argparse  # 用于解析命令行参数
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
from UNAGI import UNAGI
from UNAGI.utils.attribute_utils import split_dataset_into_stage

# 1. 重写UNAGI类（保持原逻辑不变）
class UNAGIWithResume(UNAGI):
    def setup_data(self, data_path, stage_key, total_stage, 
                  gcn_connectivities=False, neighbors=50, threads=20,
                  skip_existing_dirs=False):
        if total_stage < 2:
            raise ValueError('The total number of stages should be larger than 1')
        
        if os.path.isfile(data_path):
            self.data_folder = os.path.dirname(data_path)
        else:
            self.data_folder = data_path
        self.stage_key = stage_key

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

        if not skip_existing_dirs:
            if os.path.exists(os.path.join(self.data_folder, '0')):
                raise ValueError('The iteration 0 folder is already existed, please remove it or use skip_existing_dirs=True')
            if os.path.exists(os.path.join(self.data_folder, '0/stagedata')):
                raise ValueError('The iteration 0/stagedata folder is already existed, please remove it or use skip_existing_dirs=True')
            if os.path.exists(os.path.join(self.data_folder, 'model_save')):
                raise ValueError('The model_save folder is already existed, please remove it or use skip_existing_dirs=True')
            
            dir1 = os.path.join(self.data_folder, '0')
            dir2 = os.path.join(self.data_folder, '0/stagedata')
            dir3 = os.path.join(self.data_folder, 'model_save')
            os.makedirs(dir1, exist_ok=True)
            os.makedirs(dir2, exist_ok=True)
            os.makedirs(dir3, exist_ok=True)
        else:
            print("Skipping directory existence check (skip_existing_dirs=True)")
            os.makedirs(os.path.join(self.data_folder, '0'), exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, '0/stagedata'), exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, 'model_save'), exist_ok=True)

        if not gcn_connectivities:
            print(f'Calculating cell graphs with K={neighbors}, threads={threads}')
            self.calculate_neighbor_graph(neighbors, threads)
        else:
            print('Cell graphs found, skipping calculation')

# 2. 命令行参数解析函数（新增--ori-adata-path参数）
def parse_args():
    parser = argparse.ArgumentParser(description="UNAGI Training & Reconstruction (Command Line Version)")
    
    # 核心路径参数（必传）
    parser.add_argument('--data-folder', required=True, help='Path to data folder (e.g., /home/.../GSE132188/100epochs)')
    parser.add_argument('--idrem-dir', required=True, help='Path to IDREM directory (e.g., /home/.../UNAGI/idrem/)')
    parser.add_argument('--minus-matrix-dir', required=True, help='Path to folder with minus matrix CSV files (e.g., /home/.../multi_1p5/time0)')
    parser.add_argument('--recons-save-dir', required=True, help='Path to save reconstruction results (e.g., /home/.../recons_mat/1p5/time0)')
    parser.add_argument('--ori-adata-path', required=True, help='Path to original h5ad file (e.g., /home/.../GSE132188/100epochs/0.h5ad)')  # 新增参数
    
    # 训练配置参数（可选，有默认值）
    parser.add_argument('--max-iter', type=int, default=3, help='Max iteration for training (default: 3)')
    parser.add_argument('--total-stage', type=int, default=4, help='Total number of stages (default: 4)')
    parser.add_argument('--stage-key', default='age(days)', help='Stage key for dataset splitting (default: "age(days)")')
    parser.add_argument('--neighbors', type=int, default=50, help='Number of neighbors for cell graph (default: 50)')
    parser.add_argument('--threads', type=int, default=20, help='Number of threads for cell graph calculation (default: 20)')
    parser.add_argument('--batchsize', type=int, default=560, help='Batch size for training (default: 560)')
    parser.add_argument('--epoch-iter', type=int, default=100, help='Number of epochs per iteration (default: 100)')
    parser.add_argument('--gcn-connectivities', action='store_true', help='Whether cell graphs are pre-calculated (add this flag if yes)')
    
    return parser.parse_args()

# 3. 辅助函数（保持原逻辑不变）
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

# 4. 主函数（使用--ori-adata-path参数）
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 路径配置
    current_file = pathlib.Path(__file__).resolve()
    project_root = current_file.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.append("/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI")
    print(f"项目根目录: {project_root}")
    print(f"当前 sys.path: {sys.path[:5]}")

    # 初始化UNAGI实例
    unagi = UNAGIWithResume()

    # 检查训练是否完成，决定是否跳过目录检查
    training_completed = is_training_completed(args.data_folder, args.max_iter)
    unagi.setup_data(
        data_path=args.data_folder,
        total_stage=args.total_stage,
        stage_key=args.stage_key,
        gcn_connectivities=args.gcn_connectivities,
        neighbors=args.neighbors,
        threads=args.threads,
        skip_existing_dirs=training_completed
    )

    # 配置训练参数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    unagi.setup_training(
        task='GSE132188',
        dist='ziln',
        device=device,
        GPU=torch.cuda.is_available(),
        epoch_iter=args.epoch_iter,
        epoch_initial=1,
        max_iter=args.max_iter,
        BATCHSIZE=args.batchsize
    )

    # 执行训练或恢复
    if training_completed:
        print("Training already completed, skipping...")
        unagi.run_UNAGI(
            idrem_dir=args.idrem_dir,
            CPO=True,
            resume=True,
            resume_iteration=args.max_iter
        )
    else:
        print("Starting training...")
        unagi.run_UNAGI(
            idrem_dir=args.idrem_dir,
            CPO=True,
            resume=False
        )

    # 重构矩阵处理（使用--ori-adata-path参数）
    os.makedirs(args.recons_save_dir, exist_ok=True)
    temp_dir = os.path.join(args.recons_save_dir, "temp_processed")
    os.makedirs(temp_dir, exist_ok=True)

    # 读取原始数据获取基因名（使用命令行传入的路径）
    ori_adata_path = args.ori_adata_path  # 关键修改：直接使用参数
    print(f"读取原始数据: {ori_adata_path}")
    ori_adata = sc.read(ori_adata_path)
    gene_names = ori_adata.var_names.tolist()
    print(f"成功获取 {len(gene_names)} 个基因名称")

    # 循环处理减法矩阵CSV文件
    for filename in os.listdir(args.minus_matrix_dir):
        if filename.endswith('.csv'):
            adata_path = os.path.join(args.minus_matrix_dir, filename)
            print(f"\n处理文件: {filename}")

            # 读取并预处理CSV
            df = pd.read_csv(adata_path)
            cell_type_column = df[['index', 'Cell_Type']].copy() if 'Cell_Type' in df.columns else None
            columns_to_drop = [col for col in ['cell_type', 'Cell_Type'] if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                print(f"已移除列: {columns_to_drop}")

            # 保存临时文件
            temp_file_path = os.path.join(temp_dir, filename)
            df.to_csv(temp_file_path, index=False)

            # 调用重构函数（使用命令行传入的ori_adata_path）
            save_filename = f"recons_{os.path.splitext(filename)[0]}.csv"
            save_path = os.path.join(args.recons_save_dir, save_filename)
            recons_matrx = unagi.save_perturbed_reconstruction_withGCN(
                ori_data_path=ori_adata_path,  # 关键修改：使用参数
                adata_path=temp_file_path,
                iteration=args.max_iter - 1,
                save_path=save_path
            )

            # 后处理重构结果
            if recons_matrx is not None:
                recons_df = pd.DataFrame(recons_matrx)
                original_df = pd.read_csv(adata_path)
                recons_df.index = original_df["index"]

                if len(gene_names) == recons_df.shape[1]:
                    recons_df.columns = gene_names
                    print(f"已替换列名为基因名")
                else:
                    print(f"警告: 基因名数量({len(gene_names)})与列数({recons_df.shape[1]})不匹配")

                if cell_type_column is not None:
                    cell_type_column = cell_type_column.set_index('index')
                    recons_df = recons_df.join(cell_type_column, how='left')
                    print(f"已恢复Cell_Type列")

                final_save_path = os.path.join(args.recons_save_dir, f"processed_{save_filename}")
                recons_df.to_csv(final_save_path)
                print(f"已保存最终结果: {final_save_path}")
            else:
                print(f"警告: 未获取到重构矩阵，跳过后续处理")

    print("\nAll tasks completed!")

# 5. 入口函数
if __name__ == "__main__":
    main()