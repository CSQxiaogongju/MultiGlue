import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from model import Encoder_overall
from scipy.sparse import issparse
from loss import graph_recon_loss
from torch import nn
from preprocess import adjacent_matrix_preprocessing, preprocess_graph, pca_, add_contrastive_label, permutation
from utils import cosine_similarity_matrix

class Train_MultiGlue:
    def __init__(self, 
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64, #64
        weight_factors = [1, 5, 1, 1]
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.  
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        
        # adding contrastive labels
        add_contrastive_label(self.adata_omics1)
        add_contrastive_label(self.adata_omics2)
        
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        #self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        #self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        # build neighbor graph (dense)
        graph_neigh = self.adj['graph_neighbor']
        self.graph_neigh = torch.FloatTensor(graph_neigh + np.eye(graph_neigh.shape[0])).to(self.device) 
        
        # gene-peak interactions
        self.adj_graph_ = self.data['adj_graph']  
        self.adj_graph = preprocess_graph(self.adj_graph_).to(self.device)
        #self.adj_graph = self.adj_graph_
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        # define features for corrupted graph
        self.features_omics1_a = torch.FloatTensor(self.adata_omics1.obsm['feat_a'].copy()).to(self.device)
        self.features_omics2_a = torch.FloatTensor(self.adata_omics2.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata_omics1.obsm['label_CSL']).to(self.device)
        
        if issparse(self.adata_omics1.X):
           raw_data_omics1 = self.adata_omics1.X.toarray()
        else:
           raw_data_omics1 = self.adata_omics1.X
           
        if issparse(self.adata_omics2.X):
           raw_data_omics2 = self.adata_omics2.X.toarray()
        else:
           raw_data_omics2 = self.adata_omics2.X   
           
        # obtain gene and peak expression features by transposing raw data
        #raw_data_gene = raw_data_omics1.T
        #raw_data_peak = raw_data_omics2.T
        
        # dimension reduction
        #features_raw_gene = pca_(raw_data_gene, n_comps=50)
        #features_raw_peak = pca_(raw_data_peak, n_comps=50)
           
        self.features_raw_omics1 = torch.FloatTensor(raw_data_omics1).to(self.device)
        self.features_raw_omics2 = torch.FloatTensor(raw_data_omics2).to(self.device)
        
        #self.features_raw_gene = torch.FloatTensor(features_raw_gene).to(self.device)
        #self.features_raw_peak = torch.FloatTensor(features_raw_peak).to(self.device)
        
        self.features_context_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat_context'].copy()).to(self.device)
        self.features_context_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat_context'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.n_feat_omics1 = self.adata_omics1.n_vars #3000 #3000 #2716 #2392 #self.adata_omics1.n_vars
        self.n_feat_omics2 = self.adata_omics2.n_vars #6942 #6942 #13443 #4909 #self.adata_omics2.n_vars
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        if self.datatype == 'SPOTS':
           self.epochs = 600 #600 
           self.weight_factors = [1,5,1,1]
           
        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 600 # mouse_thymus->3000 mouse_spleen->800 human_lymph_node 800
           self.weight_factors = [1,10,1,10]
           
        elif self.datatype == '10x':
           self.epochs = 200
           self.weight_factors = [1,5,1,10]
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 1600  #mouse embryo->2500 human_brain->1600 E13_mouse_embryo->2000 p22_mouse_brain->3000
           self.weight_factors = [1,5,1,1]
        elif self.datatype == 'rna_atac':
           self.epochs = 2000 #2000
        elif self.datatype == 'rna_meta':
           self.epochs = 3000 #2000   
    
    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, 
                                     self.dim_input2, self.dim_output2,
                                     self.n_feat_omics1, self.n_feat_omics2,
                                     self.graph_neigh).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            
            self.features_omics1_a = permutation(self.features_omics1)
            self.features_omics2_a = permutation(self.features_omics2)
            
            results = self.model(self.features_omics1, self.features_omics2,
                                 self.features_omics1_a, self.features_omics2_a,
                                 self.features_context_omics1, self.features_context_omics2, 
                                 self.adj_spatial_omics1, self.adj_spatial_omics2,
                                 self.adj_graph)
            
            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            
            # graph reconstruction loss
            #adj_orig = self.adj_graph_[0:self.n_feat_omics1, self.n_feat_omics1:(self.n_feat_omics1 + self.n_feat_omics2)]
            #adj_orig = self.adj_graph_
            #adj_orig = torch.FloatTensor(adj_orig).to(self.device)
            #adj_recon_logits = emb_graph_omics1 @ emb_graph_omics2.t()
            #adj_recon_logits = emb_graph @ emb_graph.t()
            #mask = adj_orig == 1
            #self.loss_recon_graph = F.binary_cross_entropy_with_logits(adj_recon[mask], torch.ones_like(adj_orig[mask]))
            #self.loss_recon_graph = graph_recon_loss(adj_recon_logits, adj_orig, epoch)
            
            # expression reconstruction loss
            emb_latent_omics1, emb_latent_omics2 = results['emb_orig_omics1'], results['emb_orig_omics2']
            
            emb_graph = results['emb_graph']
            emb_graph_omics1, emb_graph_omics2 = emb_graph[0:self.n_feat_omics1], emb_graph[self.n_feat_omics1:]
               
            pred_omics1 = emb_latent_omics1 @ emb_graph_omics1.t()  # 64
            pred_omics2 = emb_latent_omics2 @ emb_graph_omics2.t()  # 64
            
            self.loss_recon_exp1 = F.mse_loss(self.features_raw_omics1, pred_omics1)  #[2500, 15061]
            self.loss_recon_exp2 = F.mse_loss(self.features_raw_omics2, pred_omics2)
            
            # contrastive loss
            ret_omics1 = results['ret_omics1']
            ret_omics1_a = results['ret_omics1_a']
            ret_omics2 = results['ret_omics2']
            ret_omics2_a = results['ret_omics2_a']
            
            self.loss_sl_1_orig = self.loss_CSL(ret_omics1, self.label_CSL)
            self.loss_sl_1_a = self.loss_CSL(ret_omics1_a, self.label_CSL)
            self.loss_sl_1 = self.loss_sl_1_orig + self.loss_sl_1_a
            
            self.loss_sl_2_orig = self.loss_CSL(ret_omics2, self.label_CSL)
            self.loss_sl_2_a = self.loss_CSL(ret_omics2_a, self.label_CSL)
            self.loss_sl_2 = self.loss_sl_2_orig + self.loss_sl_2_a
            
            self.loss_sl = self.loss_sl_1 +self.loss_sl_2
            
            ### RNA-ATAC
            #loss = self.loss_recon_exp1 + 5*self.loss_recon_exp2 + 0.3*self.loss_recon_omics1 + self.loss_recon_omics2 + 0.3*self.loss_sl # p22_mouse_brain
            #loss = self.loss_recon_exp1 + self.loss_recon_exp2 + 0.1*self.loss_recon_omics1 + self.loss_recon_omics2 + 0.1*self.loss_sl  # E13_mouse_embryo
            #loss = self.loss_recon_exp1 + 4*self.loss_recon_exp2 + 0.2*self.loss_recon_omics1 + self.loss_recon_omics2 + 0.5*self.loss_sl # human_brain
            #loss = self.loss_recon_exp1 + 0.5*self.loss_recon_exp2 + 0.2*self.loss_recon_omics1 + 2*self.loss_recon_omics2 + 0.1*self.loss_sl  # mouse_embryo
            
            ### RNA-ADT
            #loss = 5*self.loss_recon_exp1 + 0.2*self.loss_recon_exp2 + self.loss_recon_omics1 + 20*self.loss_recon_omics2 + 0.1*self.loss_sl  # mouse_thymus
            #loss = self.loss_recon_omics1 + 50*self.loss_recon_omics2 + 0.3*self.loss_sl   # mouse_spleen
            #loss = 0.9*self.loss_recon_omics1 + 20*self.loss_recon_omics2 + + 0.5*self.loss_sl  # human_lymph_node
            
            #loss = self.loss_recon_exp1 + self.loss_recon_exp2 + 0.1*self.loss_recon_omics1 + self.loss_recon_omics2 + 0.1*self.loss_sl
            #loss = self.loss_recon_exp1 + 0.5*self.loss_recon_exp2 + 0.2*self.loss_recon_omics1 + 2*self.loss_recon_omics2 + 0.1*self.loss_sl 
            #loss = 4*self.loss_recon_exp1 + self.loss_recon_exp2
            
            #loss = self.loss_recon_exp1 + 0.1*self.loss_recon_exp2 + 0.1*self.loss_recon_omics1 + 0.1*self.loss_recon_omics2 + 0.1*self.loss_sl
            
            #loss = 2*self.loss_recon_exp1 + self.loss_recon_exp2 #+ self.loss_recon_omics1 + 40*self.loss_recon_omics2 + 0.1*self.loss_sl
            loss = self.loss_recon_omics1 + 40*self.loss_recon_omics2 + 2*self.loss_recon_exp1 + self.loss_recon_exp2 #+ 0.1*self.loss_sl
            
            print(f"""loss: {loss:.4f}, 
                      loss_omics1: {2*self.loss_recon_exp1:.4f}, 
                      loss_omics2: {self.loss_recon_exp2:.4f},
                      loss_recon_omics1: {self.loss_recon_omics1:.4f}, 
                      loss_recon_omics2: {40*self.loss_recon_omics2:.4f},
                      loss_sl: {0.1*self.loss_sl:.4f},
                      """)
                      
            #print(f"""loss: {loss:.4f}, 
            #          loss_recon_omics1: {0.2*self.loss_recon_omics1:.4f}, 
            #          loss_recon_omics2: {self.loss_recon_omics2:.4f},
            #          loss_sl: {0.2*self.loss_sl:.4f},
            #          """)           
                      
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, 
                               self.features_omics1_a, self.features_omics2_a,
                               self.features_context_omics1, self.features_context_omics2,
                               self.adj_spatial_omics1, self.adj_spatial_omics2,
                               self.adj_graph)
 
        #emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        #emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        
        ## normalization for RNA-ATAC data
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        emb_recon_omics1 = F.normalize(results['emb_recon_omics1'], p=2, eps=1e-12, dim=1)
        emb_recon_omics2 = F.normalize(results['emb_recon_omics2'], p=2, eps=1e-12, dim=1)
        
        ## non-normalization for RNA-Protein data
        #emb_combined = results['emb_latent_combined']
        #emb_recon_omics1 = results['emb_recon_omics1']
        #emb_recon_omics2 = results['emb_recon_omics2']
        
        emb_graph = results['emb_graph']
        emb_graph_omics1, emb_graph_omics2 = emb_graph[0:self.n_feat_omics1], emb_graph[self.n_feat_omics1:]
        pred_omics1 = emb_latent_omics1 @ emb_graph_omics1.t()  # 64
        pred_omics2 = emb_latent_omics2 @ emb_graph_omics2.t()  # 64
        
        pred_omics1 = F.normalize(pred_omics1, p=2, eps=1e-12, dim=1)
        pred_omics2 = F.normalize(pred_omics2, p=2, eps=1e-12, dim=1)
        
        # output gene-peak interaction scores
        scores = cosine_similarity_matrix(emb_graph_omics1, emb_graph_omics2)
        
        output = {'MultiGlue': emb_combined.detach().cpu().numpy(),
                  'emb_recon_omics1': emb_recon_omics1.detach().cpu().numpy(),
                  'emb_recon_omics2': emb_recon_omics2.detach().cpu().numpy(),
                  'pred_omics1': pred_omics1.detach().cpu().numpy(),
                  'pred_omics2': pred_omics2.detach().cpu().numpy(),
                  'scores': scores.detach().cpu().numpy()
                 }
        
        return output
    
    
    
        
    
    
      

    
        
    
    
