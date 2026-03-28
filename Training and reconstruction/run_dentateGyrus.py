import warnings
warnings.filterwarnings('ignore')
import sys
import os

# 获取当前脚本路径（text2code_vllm.py）
current_script_path = os.path.abspath(__file__)
# evaluation 目录（当前脚本的父目录）
training_and_recons_dir = os.path.dirname(current_script_path)
print(training_and_recons_dir)
pyscripts_dir = os.path.dirname(training_and_recons_dir)
print(pyscripts_dir)
# selfcodealign-main 根目录（evaluation的父目录）
tutorials_dir = os.path.dirname(pyscripts_dir)
print(tutorials_dir)
root_dir = os.path.dirname(tutorials_dir)
print(root_dir)

# 将 src 目录添加到 sys.path（关键修正）
sys.path.append(root_dir)
print("已添加到sys.path的目录：", root_dir)
sys.path.append("/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI")
import pathlib

# 使用 pathlib 获取更可靠的路径
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent  # 根据你的项目结构调整

# 添加到 sys.path
sys.path.insert(0, str(project_root))
print(f"项目根目录: {project_root}")
print(f"当前 sys.path: {sys.path}")

from UNAGI import UNAGI
unagi = UNAGI()
print(dir(unagi))
#如果时间段没有设为0，1这样的自然数，就需要先跑setup_data,然后将对应的h5ad名字修改后再接着跑setup_training
unagi.setup_data('UNAGI/data/DentateGyrus/Processed_1944_ep_100/10X43_1_filtered_addSim.h5ad',total_stage=2,stage_key='age(days)')
unagi.setup_training(task='DentateGyrus',dist='ziln',device='cuda:0',GPU=True,epoch_iter=100,epoch_initial=1,max_iter=3,BATCHSIZE=560)
unagi.run_UNAGI(idrem_dir = '/home/lijunxian/NancyRentedPlace/UNAGI/idrem/')

#获得所有时间阶段的摘要向量 get embeddings from all time stages
unagi.train_for_classifier(total_stage=2,
                           h5ad_folder="UNAGI/data/DentateGyrus/Processed_1944_ep_100/2/stagedata",
                           embedding_save_folder="UNAGI/data/DentateGyrus/Processed_1944_ep_100/embedding")

##设置相减的矩阵的保存路径
#minus_matrix_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/Tocons_matrix.csv"

# #原始潜在向量与扰动后的潜在向量作减法
# #发生读取多值的问题
# unagi.matrix_subtraction(ori_emb_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/embeddings/latent_embeddings_time1.csv",
#                          perturbed_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/Perturb_output/12h_to_96h/feat_54_mul_7.0_embed.csv",
#                          output_path=minus_matrix_path)

#reconstruction of perturbed embeddings
#adata_path指的是扰动的向量它的路径
# unagi.save_perturbed_reconstruction(ori_data_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/Stage_divided_h5ad_cells_genes/1.h5ad",
#                                     adata_path=minus_matrix_path,
#                                     iteration=2,
#                                     save_path="/home/lijunxian/NancyRentedPlace/MyGCN/UNAGI/data/GSE75748/construction_matrix/recons_matrix.csv")
