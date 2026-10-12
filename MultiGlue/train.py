import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from .model import Encoder_overall
from scipy.sparse import issparse
from torch import nn
from .preprocess import adjacent_matrix_preprocessing, preprocess_graph, add_contrastive_label, permutation

class Train_MultiGlue:
    def __init__(self, 
        data,
        datatype = 'Spatial-epigenome-transcriptome',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=3000, 
        dim_input=3000,
        dim_output=64, 
        weight_factors = [1, 1, 1, 1, 1] # by default
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'Spatial-proteome-transcriptome' and 'Spatial-epigenome-transcriptome'. We plan to extend our model for more data types in the future.  
            The default is 'Spatial-epigenome-transcriptome'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 3000.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different losses on model training.
    
        Returns
        -------
        The integrated representation 'MultiGlue',
        The reconstructed omics-specific representations 'emb_recon_omics1' and 'emb_recon_omics2',
        The generated modal-specific omics distributions 'pred_omics1' and 'pred_omics2'.

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
        
        # Preprocessed omics-specific data
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        
        # Adding contrastive labels
        add_contrastive_label(self.adata_omics1)
        add_contrastive_label(self.adata_omics2)
        
        # Preprocessing adjacent matrices
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        
        # Preprocessing neighbor graph
        graph_neigh = self.adj['graph_neighbor']
        self.graph_neigh = torch.FloatTensor(graph_neigh + np.eye(graph_neigh.shape[0])).to(self.device) 
        
        # Preprocessing molecular adjacent matrix
        self.adj_graph_ = self.data['adj_graph']  
        self.adj_graph = preprocess_graph(self.adj_graph_).to(self.device)
        
        # PCA-reduced omics features
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        # PCA-reduced omics features (corrupted)
        self.features_omics1_a = torch.FloatTensor(self.adata_omics1.obsm['feat_a'].copy()).to(self.device)
        self.features_omics2_a = torch.FloatTensor(self.adata_omics2.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata_omics1.obsm['label_CSL']).to(self.device)
        
        # Raw omics features
        if issparse(self.adata_omics1.X):
           raw_data_omics1 = self.adata_omics1.X.toarray()
        else:
           raw_data_omics1 = self.adata_omics1.X
           
        if issparse(self.adata_omics2.X):
           raw_data_omics2 = self.adata_omics2.X.toarray()
        else:
           raw_data_omics2 = self.adata_omics2.X   
           
        self.features_raw_omics1 = torch.FloatTensor(raw_data_omics1).to(self.device)
        self.features_raw_omics2 = torch.FloatTensor(raw_data_omics2).to(self.device)
        
        self.features_context_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat_context'].copy()).to(self.device)
        self.features_context_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat_context'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.n_feat_omics1 = self.adata_omics1.n_vars 
        self.n_feat_omics2 = self.adata_omics2.n_vars 
        
        # Dimension of input features
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
           
        if self.datatype == 'Spatial-proteome-transcriptome':
           self.epochs = 3500 
           self.weight_factors = [5,0.2,1,25,0.1]
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 3000
           self.weight_factors = [0,0,0.3,0.8,0.2]
        else:
           raise NotImplementedError(f"Unknown data type {self.datatype}")  
    
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
            
            # Reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            
            # Prediction loss (Optional)
            emb_latent_omics1, emb_latent_omics2 = results['emb_orig_omics1'], results['emb_orig_omics2']
            
            emb_graph = results['emb_graph']
            emb_graph_omics1, emb_graph_omics2 = emb_graph[0:self.n_feat_omics1], emb_graph[self.n_feat_omics1:]
              
            pred_omics1 = emb_latent_omics1 @ emb_graph_omics1.t()  
            pred_omics2 = emb_latent_omics2 @ emb_graph_omics2.t()  
            
            self.loss_pred_omics1 = F.mse_loss(self.features_raw_omics1, pred_omics1) 
            self.loss_pred_omics2 = F.mse_loss(self.features_raw_omics2, pred_omics2)
            
            # Contrastive loss
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
            
            loss = self.weight_factors[0]*self.loss_pred_omics1 + self.weight_factors[1]*self.loss_pred_omics2 + \
                  self.weight_factors[2]*self.loss_recon_omics1 + self.weight_factors[3]*self.loss_recon_omics2 + self.weight_factors[4]*self.loss_sl
            
            print(f"""loss: {loss:.4f}""")
                      
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
 
        if self.datatype == 'Spatial-proteome-transcriptome':
           emb_combined = results['emb_latent_combined']
           emb_recon_omics1 = results['emb_recon_omics1']
           emb_recon_omics2 = results['emb_recon_omics2']
           
        elif self.datatype == 'Spatial-epigenome-transcriptome':   
           emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
           emb_recon_omics1 = F.normalize(results['emb_recon_omics1'], p=2, eps=1e-12, dim=1)
           emb_recon_omics2 = F.normalize(results['emb_recon_omics2'], p=2, eps=1e-12, dim=1)

        pred_omics1, pred_omics2 = results['pred_omics1'], results['pred_omics2']
        
        output = {'MultiGlue': emb_combined.detach().cpu().numpy(),
                  'emb_recon_omics1': emb_recon_omics1.detach().cpu().numpy(),
                  'emb_recon_omics2': emb_recon_omics2.detach().cpu().numpy(),
                  'pred_omics1': pred_omics1.mean.detach().cpu().numpy(),
                  'pred_omics2': pred_omics2.mean.detach().cpu().numpy(),
                 }
        
        return output
    
    
    
        
    
    
      

    
        
    
    
