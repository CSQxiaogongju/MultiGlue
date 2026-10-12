import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .model_pro import ZINBDataDecoder, BerDataDecoder, NBDataDecoder, PoisDataDecoder
    
class Encoder_overall(Module):
      
    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2. 
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    n_feat_omics1 : int
        Number of molecules (e.g., gene) in omics1 data.
    n_feat_omics2 : int
        Number of molecules (e.g., proteins) in omics2 data.
    graph_neigh : int
        Adjacent matrix for molecular interaction (e.g., gene x protein) graph. 
    data_type : str
        Data type of input data, such as 'Spatial-proteome-transcriptome' and 'Spatial-epignome-transcriptome'.
    dim_graph_in : int
        Dimension of original molecule features.
    dim_graph_output : int
        Dimension of latent molecule representations.    
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    results: a dictionary including various representations.

    """
     
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, 
                 dim_in_feat_omics2, dim_out_feat_omics2, 
                 n_feat_omics1, n_feat_omics2, 
                 graph_neigh, datatype,
                 dim_graph_in=256, dim_graph_output=64, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.n_feat_omics1 = n_feat_omics1
        self.n_feat_omics2 = n_feat_omics2
        self.graph_neigh = graph_neigh
        self.datatype = datatype
        self.dim_graph_in = dim_graph_in
        self.dim_graph_output = dim_graph_output
        self.dropout = dropout
        self.act = act
        
        self.n_feat_node = self.n_feat_omics1 + self.n_feat_omics2
        self.out_features = self.dim_out_feat_omics1
        self.l = torch.as_tensor(1.0) 
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.encoder_graph = Encoder_Graph(self.n_feat_node, self.dim_graph_in, self.dim_graph_output)
        
        self.generator_omics1 = NBDataDecoder(self.n_feat_omics1) # for omics1 data
        
        if self.datatype == 'Spatial-proteome-transcriptome': # for omics2 data
           self.generator_omics2 = PoisDataDecoder(self.n_feat_omics2)
        elif self.datatype == 'Spatial-epignome-transcriptome':   
           self.generator_omics2 = NBDataDecoder(self.n_feat_omics2) 
        else:
           raise NotImplementedError(f"Unknown data type {self.datatype}") 
        
        self.decoder_recon_omics1 = nn.Sequential(
             nn.Linear(self.dim_out_feat_omics1, 32),
             #nn.LayerNorm(32),
             nn.Linear(32, self.dim_in_feat_omics1)
             )
        
        self.decoder_recon_omics2 = nn.Sequential(
             nn.Linear(self.dim_out_feat_omics2, 32),
             #nn.LayerNorm(32),
             nn.Linear(32, self.dim_in_feat_omics2)
             )
        
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
   
    def forward(self, features_omics1, features_omics2, 
                features_omics1_a, features_omics2_a,
                features_context_omics1, features_context_omics2, 
                adj_omics1, adj_omics2, adj_graph):
        
        # Omics1
        emb_orig_omics1 = self.encoder_omics1(features_omics1, adj_omics1)  
        emb_orig_omics1_a = self.encoder_omics1(features_omics1_a, adj_omics1)
        
        # Omics2
        emb_orig_omics2 = self.encoder_omics2(features_omics2, adj_omics2)
        emb_orig_omics2_a = self.encoder_omics2(features_omics2_a, adj_omics2)
        
        # Between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_orig_omics1, emb_orig_omics2)
        
        # Reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_recon_omics1(emb_latent_combined)  # reconstructing omics-specific using combined embedding
        emb_recon_omics2 = self.decoder_recon_omics2(emb_latent_combined)
        
        # Encoding molecular heterogeneous network
        emb_graph = self.encoder_graph(adj_graph)
        
        # Generating unimodal data
        emb_omics1, emb_omics2 = emb_graph[0:self.n_feat_omics1], emb_graph[self.n_feat_omics1:]
        
        pred_omics1 = self.generator_omics1(emb_latent_combined, emb_omics1, self.l)
        pred_omics2 = self.generator_omics2(emb_latent_combined, emb_omics2, self.l)
        
        # Constrastive learning within modality 1
        g_omics1 = self.read(self.act(emb_orig_omics1), self.graph_neigh)
        g_omics1 = self.sigm(g_omics1)
        
        g_omics1_a = self.read(self.act(emb_orig_omics1_a), self.graph_neigh)
        g_omics1_a =self.sigm(g_omics1_a)       
       
        ret_omics1 = self.disc(g_omics1, emb_orig_omics1, emb_orig_omics1_a)  
        ret_omics1_a = self.disc(g_omics1_a, emb_orig_omics1_a, emb_orig_omics1)
        
        # Constrastive learning within modality 2
        g_omics2 = self.read(self.act(emb_orig_omics2), self.graph_neigh)
        g_omics2 = self.sigm(g_omics2)
        
        g_omics2_a = self.read(self.act(emb_orig_omics2_a), self.graph_neigh)
        g_omics2_a =self.sigm(g_omics2_a)       
       
        ret_omics2 = self.disc(g_omics2, emb_orig_omics2, emb_orig_omics2_a)  
        ret_omics2_a = self.disc(g_omics2_a, emb_orig_omics2_a, emb_orig_omics2)
        
        results = {'emb_orig_omics1': emb_orig_omics1,
                   'emb_orig_omics2': emb_orig_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'ret_omics1':ret_omics1,
                   'ret_omics1_a':ret_omics1_a,
                   'ret_omics2':ret_omics2,
                   'ret_omics2_a':ret_omics2_a,
                   'emb_graph': emb_graph,
                   'pred_omics1': pred_omics1,
                   'pred_omics2': pred_omics2,
                   }
        
        return results     

class Encoder(Module): 
    
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Latent representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x
    
class Discriminator(nn.Module):
    
    """\
    Discriminator for distinguishing postive samples from negative samples.

    Parameters
    ----------
    n_h: int
        Dimension of features in the bi-linear layer.    

    Inputs
    ----------
    c : torch.Tensor
        Context vector, shape: [batch_size, n_h]
    h_pl : torch.Tensor
        Positive sample embedding, shape: [batch_size, n_h]
    h_mi : torch.Tensor
        Negative sample embedding, shape: [batch_size, n_h]
    s_bias1 : torch.Tensor, optional
        Bias term for positive score, default=None
    s_bias2 : torch.Tensor, optional
        Bias term for negative score, default=None

    Returns
    -------
    logits : torch.Tensor
        Concatenated logits of positive and negative matching scores.
        shape: [batch_size, 2]
    """
    
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    """\
        Readout function.
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1)     
    
    
class Encoder_Graph(Module): 
    
    """\
    Graph GNN encoder for modeling molecular interaction network.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Latent molecular representation.

    """
    
    def __init__(self, num_feat_node, dim_graph=256, dim_out=64, dropout=0.0, act=F.relu):
        super(Encoder_Graph, self).__init__()
        self.num_feat_node = num_feat_node
        self.dim_graph = dim_graph
        self.dim_out = dim_out
        self.feature = Parameter((torch.FloatTensor(self.num_feat_node, self.dim_graph)))
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.dim_graph, self.dim_out))
        #self.weight2 = Parameter(torch.FloatTensor(self.dim_out, self.dim_out))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.feature)
        
    def forward(self, adj):
        x = torch.mm(self.feature, self.weight)
        x = torch.spmm(adj, x)
        
        #x = torch.mm(x, self.weight2)
        #x = torch.spmm(adj, x)
        
        return x    
    
class Decoder(Module):
    
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x                  

class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha      
