#Introduction
Cell-state transition regulators have garnered substantial attention in regenerative and preventive medicine due to their pivotal role in dictating cellular trajectories during complex biological processes. However, conventional methods often overlook intercellular contextual influences and fail to delineate stage-specific regulatory factors underlying sequential cell-state transitions. To address these limitations, we introduce ReguFinder, a deep learning framework based on a Cell Graph Convolutional Layer-Variational Autoencoder (CGCL-VAE) with generative adversarial network, designed to identify regulators of cell-state transitions. Specifically, by integrating time-stage–specific graph neural network autoencoder modeling, systematic perturbation of latent dimensions, and decoder-driven prioritization score calculation, ReguFinder enables the identification of core regulatory dimensions and their associated regulators governing cell-state transitions, which are functionally shown to be implicated in the corresponding transition processes. Additionally, it allows the identification of gene regulatory modules that mediate cell-state transitions through the coordinated interactions among these regulators. ReguFinder has successfully applied to five distinct real-world datasets, including dentate gyrus development, pancreatic endocrinogenesis, liver development and maturation, hematopoiesis, and COVID-19, where it consistently demonstrated higher accuracy than conventional methods. Consequently, ReguFinder provides a computational framework for prioritizing candidate regulators of cell-state transitions through systematic in silico perturbation, offering mechanistic hypotheses for downstream experimental validation.


##Preprocessing:
Firstly, it is highly advised that you should select some highly varible genes to reduce 计算量, and add the 'name.simple' to your h5ad, which is the stages you concerned
you can use data_preprocessing_addSimp.py in preprocessing 文件夹。

##Training
Then,you can start your training of the CGCL-VAE model using run_dentateGyrus.py

##perturbation
Using the Emb_Cell_type_Classifier_Dentate_11.py to first add the cell type to embedding from each time, and use them to train the classifier.Then start the systematically perturbation.
it also has the function that can produce the difference matrix between unperturbed embedding and perturbed embedding, and combine the matrices from all time stages.

##Find our Regulators
use run_GSE132188_resume_argparse.py to reconstruct the priorization scores matrix and find regulators using run_regus_for_allType

