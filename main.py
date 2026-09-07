import os
import torch
import pandas as pd
import scanpy as sc
import pickle
import numpy as np
from preprocess import permutation

# Environment configuration. MultiGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# the location of R, which is required for the 'mclust' clustering algorithm. Please replace the path below with local R installation path
#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'
os.environ['R_HOME'] = '/home/andy/miniconda3/envs/MultiGlue/lib/R'

# read data
#file_fold = '/home/andy/miniconda3/work/MultiGlue/data/mouse_thymus/' #please replace 'file_fold' with the download path
#file_fold = '/home/andy/miniconda3/work/MultiGlue/data/mouse_spleen/'
#file_fold = '/home/andy/miniconda3/work/MultiGlue/data/human_hippocampus/human_brain/'

#file_fold = '/home/andy/miniconda3/work/MultiGlue/data/data_202060129/human_brain/'  # RNA-ATAC

file_fold = '/home/andy/miniconda3/work/MultiGlue/data/RNA_ADT_20260331/Mouse_Spleen/' #RNA-ADT

#file_fold = '/home/andy/miniconda3/work/MultiGlue/data/V11L12_038_A1/' #RNA-META    

with open(file_fold + 'adj_full_17.pkl','rb') as f:
    adj_graph = pickle.load(f)    
    
#adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
#adata_omics2 = sc.read_h5ad(file_fold + 'adata_peaks_normalized.h5ad')

adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA1.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_Pro1.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

print('RNA:', adata_omics1)
print('ADT:', adata_omics2)

#RNA-ADT
data_type = 'Stereo-CITE-seq'

#RNA-ATAC
#data_type = 'Spatial-epigenome-transcriptome'

#RNA-META
#data_type = 'rna_meta'


# Fix random seed
from preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)

# construct graph
from preprocess import construct_neighbor_graph
adata_omics1, adata_omics2 = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

from preprocess import clr_normalize_each_cell, lsi, pca

if data_type in ['SPOTS','Stereo-CITE-seq', '10x']:
  # RNA
  #sc.pp.filter_genes(adata_omics1, min_cells=10)
  #sc.pp.filter_cells(adata_omics1, min_genes=80)

  #sc.pp.filter_genes(adata_omics2, min_cells=50)
  adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()

  sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
  sc.pp.normalize_total(adata_omics1, target_sum=1e4)
  sc.pp.log1p(adata_omics1)

  adata_omics1 =  adata_omics1[:, adata_omics1.var['highly_variable']]
  adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=adata_omics2.n_vars-1)
  
  adata_omics1.obsm['feat_a'] = permutation(adata_omics1.obsm['feat'].copy())

  # Protein
  adata_omics2 = clr_normalize_each_cell(adata_omics2)
  adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
  
  adata_omics2.obsm['feat_a'] = permutation(adata_omics2.obsm['feat'].copy())

  # obtain contextual feature matrices
  from preprocess import neighbor_context_features
  adata_omics1.obsm['feat_context'] = neighbor_context_features(adata_omics1)
  adata_omics2.obsm['feat_context'] = neighbor_context_features(adata_omics2)
  
  data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adj_graph': adj_graph}
  
elif data_type in ['Spatial-epigenome-transcriptome']:
  # RNA
  #sc.pp.filter_genes(adata_omics1, min_cells=10)
  #sc.pp.filter_cells(adata_omics1, min_genes=200)

  #sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
  sc.pp.normalize_total(adata_omics1, target_sum=1e4)
  sc.pp.log1p(adata_omics1)
  sc.pp.scale(adata_omics1)

  adata_omics1 =  adata_omics1[:, adata_omics1.var['highly_variable']]
  adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50)
  
  adata_omics1.obsm['feat_a'] = permutation(adata_omics1.obsm['feat'].copy())
  
  # ATAC
  adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
  if 'X_lsi' not in adata_omics2.obsm.keys():
     #sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
     lsi(adata_omics2, use_highly_variable=False, n_components=51)  
  
  adata_omics2 =  adata_omics2[:, adata_omics2.var['highly_variable']]  
  adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
  
  adata_omics2.obsm['feat_a'] = permutation(adata_omics2.obsm['feat'].copy())
  
  #adata_omics2.obsm['feat'] = pca(adata_omics2, use_reps='X_epiagent', n_comps=50)

  # obtain contextual feature matrices
  from preprocess import neighbor_context_features
  adata_omics1.obsm['feat_context'] = neighbor_context_features(adata_omics1)
  adata_omics2.obsm['feat_context'] = neighbor_context_features(adata_omics2)
  
  print('adj_graph:', adj_graph.shape)
  print(adata_omics1)
  print(adata_omics2)
  
  data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adj_graph': adj_graph}
  
elif data_type in ['rna_atac']:
   
  # RNA  
  adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50) 
  
  # ATAC
  lsi(adata_omics2, use_highly_variable=False, n_components=51)
  ###adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)
  adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

  
  #RNA
  #adata_omics1.obsm['feat'] = pca(adata_omics1, use_reps='X_scGPT', n_comps=50)
  #ATAC
  #adata_omics2.obsm['feat'] = pca(adata_omics2, use_reps='X_EpiAgent', n_comps=50)
  
  # obtain contextual feature matrices
  from preprocess import neighbor_context_features
  adata_omics1.obsm['feat_context'] = neighbor_context_features(adata_omics1)
  adata_omics2.obsm['feat_context'] = neighbor_context_features(adata_omics2)
  
  data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adj_graph': adj_graph}
  
  
elif data_type in ['rna_meta']:
  sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
  sc.pp.normalize_total(adata_omics1, target_sum=1e4)
  sc.pp.log1p(adata_omics1)
  sc.pp.scale(adata_omics1)

  adata_omics1 =  adata_omics1[:, adata_omics1.var['highly_variable']]
  adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50)
  
  adata_omics1.obsm['feat_a'] = permutation(adata_omics1.obsm['feat'].copy())
  
  # META  
  adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50) 
  adata_omics2.obsm['feat_a'] = permutation(adata_omics2.obsm['feat'].copy())
  
  # obtain contextual feature matrices
  from preprocess import neighbor_context_features
  adata_omics1.obsm['feat_context'] = neighbor_context_features(adata_omics1)
  adata_omics2.obsm['feat_context'] = neighbor_context_features(adata_omics2)
  
  print('adj_graph:', adj_graph.shape)
  print(adata_omics1)
  print(adata_omics2)
  
  data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adj_graph': adj_graph}
  
# define model
from MultiGlue import Train_MultiGlue
model = Train_MultiGlue(data, datatype=data_type, device=device)

# train model
print('Starting training...')
output = model.train()

adata = adata_omics1.copy()
adata.obsm['MultiGlue'] = output['MultiGlue']
adata.obsm['emb_recon_omics1'] = output['emb_recon_omics1']
adata.obsm['emb_recon_omics2'] = output['emb_recon_omics2']
adata.obsm['pred_omics1'] = output['pred_omics1']
adata.obsm['pred_omics2'] = output['pred_omics2']

# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from utils import clustering
num_cluster = 5
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='MultiGlue', add_key='MultiGlue', n_clusters=num_cluster, method=tool, use_pca=True)
clustering(adata, key='pred_omics1', add_key='Omics1', n_clusters=num_cluster, method=tool, use_pca=True)
clustering(adata, key='pred_omics2', add_key='Omics2', n_clusters=num_cluster, method=tool, use_pca=True)

# visualization
import matplotlib.pyplot as plt
#E13_mouse_embryo
#adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
#adata.obsm['spatial'][:,0] = -1*adata.obsm['spatial'][:,0]
#adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

# p22_mouse_brain
#adata.obsm['spatial'][:,0] = -1*adata.obsm['spatial'][:,0]
#adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

# mouse_embryo
#adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

# mouse_thymus
#adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

# mouse_spleen
adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

res = 40 #40

fig, ax_list = plt.subplots(2, 2, figsize=(8, 7))
sc.pp.neighbors(adata, use_rep='MultiGlue', n_neighbors=30)
sc.tl.umap(adata, min_dist=0.02)

sc.pl.umap(adata, color='MultiGlue', ax=ax_list[0, 0], title='MultiGlue', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='MultiGlue', ax=ax_list[0, 1], title='MultiGlue', s=res, show=False)

sc.pl.embedding(adata, basis='spatial', color='Omics1', ax=ax_list[1, 0], title='Omics1', s=res, show=False)
sc.pl.embedding(adata, basis='spatial', color='Omics2', ax=ax_list[1, 1], title='Omics2', s=res, show=False)

print(adata)

#adata.write(file_fold + 'MultiGlue.h5ad')

# save gene-peak interaction scores
#gene_names = adata_omics1.var_names
#peak_names = adata_omics2.var_names
#scores = output['scores']

#df_scores = pd.DataFrame(data=scores, index=gene_names, columns=peak_names)
#df_scores.to_csv(file_fold + 'gene_peak_scores.csv')

plt.tight_layout(w_pad=0.3)
plt.show()

