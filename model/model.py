import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.RRP import *
import math
from utils import prepare_device

class Embedding(BaseModel):
    """
    This class implements the Embedding layer that projects input time series data into a high-dimensional vector space.
    It utilizes a linear projection, layer normalization, and dropout(Not used in practice) for feature embedding.

    Args:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        drop_prob (float): Dropout probability for regularization.
        device (str): Device for computation ('cpu' or 'cuda').

    Attributes:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        drop_prob (float): Dropout probability for regularization (Not used in practice).
        device (str): Device for computation ('cpu' or 'cuda').
        feature_emb (nn.Linear): Linear projection layer for feature embedding.
        norm (nn.LayerNorm): Layer normalization for the embedded vectors.
        drop_out (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, drop_prob, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.device = device

        self.feature_emb = Linear(in_features=self.input_window*self.feature_dim, out_features=self.d_model, bias=True)
        # He Initialization
        self.feature_emb.weight.data.normal_(0, math.sqrt(2.0/(self.input_window*self.feature_dim+self.d_model)))
        self.norm = LayerNorm(self.d_model)
        self.drop_out = Dropout(self.drop_prob)

    def forward(self, x):
        # [batch_size, series_num, input_window, feature_dim]
        x = x.reshape(-1, self.series_num, self.input_window*self.feature_dim)
        embedding = self.feature_emb(x)
        embedding = self.norm(embedding)
        embedding = self.drop_out(embedding)
        return embedding
        # [batch_size, series_num, d_model]

class CausalConv(BaseModel):
    """
    This class implements the Multi-kernel Causal Convolution block.
    It is designed to adapt the causality-aware transformer to temporal causal discovery scenarios.

    Args:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        n_head (int): Number of attention heads. h in paper.
        device (str): Device for computation ('cpu' or 'cuda').

    Attributes:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        n_head (int): Number of attention heads. h in paper.
        device (str): Device for computation ('cpu' or 'cuda').
        wgt (None): Placeholder for kernel weights.
        grad (None): Placeholder for gradients.
        rel (None): Placeholder for relevance.
        K (nn.Parameter): Learnable convolution kernel parameter.
        mul (torch.einsum): Einstein summation for convolution.
        base (torch.Tensor): Tensor used to scale the convolution result.
    """

    def __init__(self, series_num, input_window, n_head, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.n_head = n_head
        self.device = device

        self.wgt = None
        self.grad = None
        self.rel = None

        self.K = nn.Parameter(torch.ones((self.n_head, self.series_num, self.series_num, self.input_window), dtype=torch.float))
        self.register_parameter("kernel", self.K)
        self.mul = einsum('hxyji,bxif->bhxyjf')
        self.base = torch.tensor([i for i in range(1, self.input_window+1)]).reshape(1,1,1,1,-1,1).to(self.device)

    def get_wgt(self):
        return self.wgt

    def save_wgt(self, wgt):
        self.wgt = wgt

    def get_grad(self):
        return self.grad
    
    def save_grad(self, grad):
        self.grad = grad
    
    def get_rel(self):
        return self.rel
    
    def save_rel(self, rel):
        self.rel = rel

    def forward(self, x):
        # [batch_size, series_num, input_window, hidden_dim]
        kernel = []
        for i in range(self.input_window):
            # right rolling i unit
            shifted = torch.roll(self.K, i+1, dims=3)
            kernel.append(shifted)
        kernel = torch.stack(kernel)
        kernel = kernel.permute(1, 2, 3, 0, 4)
        kernel = torch.tril(kernel, diagonal=0)
        kernel.requires_grad_()

        self.save_wgt(kernel)
        kernel.register_hook(self.save_grad)

        x = self.mul([kernel ,x])
        x = x / self.base # due to plenty of padding, the base (number of non-zero elements) revise should be employed.
        # considering instantaneous causality, rolling is needed to hidden temporal self information for each series.
        for i in range(self.series_num):
            x[:,:,i,i,:,:] = x[:,:,i,i,:,:].roll(1, dims=2)
            x[:,:,i,i,0,:] *= torch.zeros_like(x[:,:,i,i,0,:])
        return x
        # [batch_size, head, series_num(data source), series_num(data user), input_window, hidden_dim]
    
    def regularization(self):
        return torch.mean(torch.norm(self.K, dim=-1, p=1))

    def relprop(self, rel):
        for i in range(self.series_num):
            rel[:,:,i,i,:,:] = rel[:,:,i,i,:,:].roll(-1, dims=2)
        rel = rel * self.base
        rel_k, rel_x = self.mul.relprop(rel)
        self.save_rel(rel_k)
        return rel_x

class MultiVariateCausalAttention(BaseModel):
    """
    This class implements the multi-variateCausal attention mechanism described in the paper.
    It computes the multi-variate causal attention by applying softmax to the scaled dot product of query and key matrices.

    Args:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        tau (float): Temperature hyperparameter for softmax.
        device (str): Device for computation ('cpu' or 'cuda').

    Attributes:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        d_tensor (int): Dimension of each tensor within an attention head.
        tau (float): Temperature hyperparameter for softmax.
        device (str): Device for computation ('cpu' or 'cuda').
        wgt (None): Placeholder for kernel weights.
        grad (None): Placeholder for gradients.
        rel (None): Placeholder for relevance.
        qk_mul (torch.einsum): Einstein summation for query-key multiplication.
        softmax (torch.nn.Softmax): Softmax function.
        mask (nn.Parameter): Learnable attention mask for sparsity adjustment.
        hardmard_product (torch.einsum): Einstein summation for element-wise multiplication.
        mul (torch.einsum): Einstein summation for attention matrix-value multiplication.
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, tau, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_head = n_head
        self.d_tensor = self.d_model // self.n_head
        self.tau = tau
        self.device = device
        
        self.wgt = None
        self.grad = None
        self.rel = None

        self.qk_mul = einsum('bhid,bhdj->bhij')
        self.softmax = Softmax(dim=-1)
        self.mask = nn.Parameter(torch.ones((self.n_head, self.series_num, self.series_num), dtype=torch.float))
        self.register_parameter("mask", self.mask)
        self.hardmard_product = einsum('hij,bhij->bhij')
        self.mul = einsum('bhij,bhjitf->bhitf')

    def get_wgt(self):
        return self.wgt

    def save_wgt(self, satt):
        self.wgt = satt

    def get_grad(self):
        return self.grad
    
    def save_grad(self, grad):
        self.grad = grad
    
    def get_rel(self):
        return self.rel
    
    def save_rel(self, rel):
        self.rel = rel

    def forward(self, q, k, v):
        # q, k: [batch_size, head, series_num, d_tensor]
        # v: [batch_size, head, series_num(data source), series_num(data user), input_window, feature_dim]

        # 1. dot product Q with K^T to compute similarity
        k_t = k.transpose(2, 3) # transpose
        score = self.qk_mul([q,k_t])/math.sqrt(self.input_window*self.d_tensor) # scaled dot product

        # 2. apply masking
        A = self.hardmard_product([self.mask, score])

        # 3. pass them softmax to make [0, 1] range
        A = self.softmax(A/self.tau)
        A.requires_grad_()
        self.save_wgt(A)
        A.register_hook(self.save_grad)

        # 4. multiply with Value
        out = self.mul([A, v])
        return out
        # [batch_size, head, series_num, input_window, feature_dim]
    
    def regularization(self):
        return torch.mean(torch.norm(self.mask, dim=-1, p=1))
    
    def relprop(self, rel):
        rel_A, rel_v = self.mul.relprop(rel)
        self.save_rel(rel_A)
        rel_score = self.softmax.relprop(rel_A)
        rel_mask, rel_score = self.hardmard_product.relprop(rel_score)
        rel_score *= math.sqrt(self.input_window * self.d_tensor)
        rel_q, rel_k = self.qk_mul.relprop(rel_score)
        rel_k = rel_k.transpose(2, 3)
        return rel_q, rel_k, rel_v

class MultiHeadAttention(BaseModel):
    """
    This class implements the Multi-Head Attention mechanism described in the paper.
    It duplicates the multi-variate causal attention for multiple heads and aggregates their outputs.

    Args:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        tau (float): Temperature hyperparameter for softmax.
        device (str): Device for computation ('cpu' or 'cuda').

    Attributes:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        tau (float): Temperature hyperparameter for softmax.
        device (str): Device for computation ('cpu' or 'cuda').
        attention (MultiVariateCausalAttention): Instance of the MultiVariateCausalAttention class.
        Wq (nn.Linear): Linear projection layer for queries.
        Wk (nn.Linear): Linear projection layer for keys.
        Wv (CausalConv): Causal convolution layer for values.
        w_concat (nn.Linear): Linear projection layer for concatenating outputs of attention heads.
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, tau, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_head = n_head
        self.tau = tau
        self.device = device
        
        self.attention = MultiVariateCausalAttention(self.series_num, self.input_window, self.feature_dim, self.d_model, self.n_head, self.tau, self.device)
        self.Wq = Linear(in_features=self.d_model, out_features=self.d_model, bias=True)
        # He Initialization
        self.Wq.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.d_model)))
        self.Wk = Linear(in_features=self.d_model, out_features=self.d_model, bias=True)
        # He Initialization
        self.Wk.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.d_model)))
        self.Wv = CausalConv(self.series_num, self.input_window, self.n_head, self.device)
        self.w_concat = Linear(in_features=self.n_head * self.feature_dim, out_features=self.feature_dim, bias=False)
        # He Initialization
        self.w_concat.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.d_model)))

    def forward(self, q, k, v):
        # q, k: [batch_size, series_num, d_model]
        # v: [batch_size, head, series_num, input_window, feature_dim]

        # 1. dot product with weight matrices
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        # q, k: [batch_size, series_num, d_model]
        # v: [batch_size, head, series_num(data source), series_num(data user), input_window, feature_dim]

        # 2. split tensor by number of heads
        q, k = self.split(q), self.split(k)
        # q, k: [batch_size, head, series_num, d_tensor]
        
        # 3. do scale dot product to compute similarity
        out = self.attention(q, k, v)
        # out: [batch_size, head, series_num, input_window, feature_dim]
        
        # 4. concat and pass to linear layer
        out = out.reshape(-1, self.n_head, self.series_num * self.input_window, self.feature_dim)
        out = self.concat(out)
        out = out.reshape(-1, self.series_num, self.input_window, self.n_head * self.feature_dim)
        out = self.w_concat(out)
        return out
        # [batch_size, series_num, input_window, feature_dim]

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.permute(0, 2, 1, 3).contiguous().view(batch_size, length, d_model)
        return tensor

    def regularization(self):
        return self.attention.regularization() + self.Wv.regularization()

    def relprop(self, rel):
        rel = self.w_concat.relprop(rel)
        rel = rel.reshape(-1, self.series_num * self.input_window, self.n_head * self.feature_dim)
        rel = self.split(rel)
        rel = rel.reshape(-1, self.n_head, self.series_num, self.input_window, self.feature_dim)
        rel_q, rel_k, rel_v = self.attention.relprop(rel)
        rel_q, rel_k = self.concat(rel_q), self.concat(rel_k)
        rel_q, rel_k, rel_v = self.Wq.relprop(rel_q), self.Wk.relprop(rel_k), self.Wv.relprop(rel_v)
        return rel_q, rel_k, rel_v

class PositionwiseFeedForward(BaseModel):
    """
    This class implements the Positionwise Feed Forward Layer described in the paper.
    It is composed of two linear neural networks separated by a leaky ReLU activation function.

    Args:
        dim (int): Input dimension.
        hidden (int): Hidden dimension in the middle of the feed forward layer. d_FFN in paper.
        drop_prob (float): Dropout probability (Not used in practice).

    Attributes:
        linear1 (nn.Linear): First linear layer transforming input to hidden dimension.
        linear2 (nn.Linear): Second linear layer transforming hidden dimension back to input dimension.
        activation (nn.LeakyReLU): Leaky ReLU activation function.
        dropout (nn.Dropout): Dropout layer with specified probability (Not used in practice).
    """

    def __init__(self, dim, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = Linear(dim, hidden, bias=True)
        # He Initialization
        self.linear1.weight.data.normal_(0, math.sqrt(2.0/(dim+hidden)))
        self.linear2 = Linear(hidden, dim, bias=True)
        # He Initialization
        self.linear2.weight.data.normal_(0, math.sqrt(2.0/(hidden+dim)))
        self.activation = LeakyReLU()
        self.dropout = Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def relprop(self, rel):
        rel = self.linear2.relprop(rel)
        rel = self.dropout.relprop(rel)
        rel = self.activation.relprop(rel)
        rel = self.linear1.relprop(rel)
        return rel

class EncoderLayer(BaseModel):

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, ffn_hidden, drop_prob, tau, device):
        super().__init__()
        self.qk = Clone()
        self.attention = MultiHeadAttention(series_num, input_window, feature_dim, d_model, n_head, tau, device)
        self.norm1 = LayerNorm([input_window, feature_dim])
        self.dropout1 = Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(dim=feature_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm([input_window, feature_dim])
        self.dropout2 = Dropout(drop_prob)

    def forward(self, x_embedding, x):
        # 1. compute self attention
        # x_embedding: [batch_size, series_num, d_model]
        # x: [batch_size, series_num, input_window, feature_dim]
        q, k = self.qk(x_embedding, 2)
        x = self.attention(q=q, k=k, v=x)
        # x: [batch_size, series_num, input_window, feature_dim]
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x)
        
        # 3. positionwise feed forward network
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x)
        return x
        # x: [batch_size, series_num, input_window, feature_dim]

    def regularization(self):
        return self.attention.regularization()
    
    def relprop(self, rel):
        rel = self.norm2.relprop(rel)
        rel = self.dropout2.relprop(rel)
        rel = self.ffn.relprop(rel)
        rel = self.norm1.relprop(rel)
        rel = self.dropout1.relprop(rel)
        rel_q, rel_k, rel_v = self.attention.relprop(rel)
        rel_emb = self.qk.relprop((rel_q, rel_k))
        return rel_emb, rel

class Encoder(BaseModel):
    """
    This class implements an Encoder Layer of the Causality-Aware Transformer.

    Args:
        series_num (int): Number of time series in the input.
        input_window (int): Length of the input time series window.
        feature_dim (int): Dimension of each feature in the time series.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        ffn_hidden (int): Hidden dimension in the feed forward layer. d_FFN in paper.
        drop_prob (float): Dropout probability (Not used in practice).
        tau (float): Temperature hyperparameter for attention softmax.
        device (str): Device for computation ('cpu' or 'cuda').

    Attributes:
        qk (Clone): Instance of the Clone class.
        attention (MultiHeadAttention): Instance of the MultiHeadAttention class.
        norm1 (LayerNorm): Layer normalization after the first attention block.
        dropout1 (Dropout): Dropout layer after the first attention block (Not used in practice).
        ffn (PositionwiseFeedForward): Instance of the PositionwiseFeedForward class.
        norm2 (LayerNorm): Layer normalization after the feed forward block.
        dropout2 (Dropout): Dropout layer after the feed forward block (Not used in practice).
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, n_layers, ffn_hidden, drop_prob, tau, device):
        super().__init__()
        self.emb = Embedding(series_num=series_num,
                             input_window=input_window,
                             feature_dim=feature_dim,
                             d_model=d_model,
                             drop_prob=drop_prob,
                             device=device)

        self.layers = nn.ModuleList([EncoderLayer(series_num=series_num,
                                                  input_window=input_window,
                                                  feature_dim=feature_dim,
                                                  d_model=d_model,
                                                  n_head=n_head,
                                                  ffn_hidden=ffn_hidden,
                                                  drop_prob=drop_prob,
                                                  tau=tau,
                                                  device=device)
                                     for _ in range(n_layers)])

    def forward(self, x):
        # x: [batch_size, series_num, input_window, feature_dim]
        embedding = self.emb(x)
        for layer in self.layers:
            x = layer(embedding, x)
        return x
        # x: [batch_size, series_num, input_window, feature_dim]

    def regularization(self):
        loss = 0
        for layer in self.layers:
            loss += layer.regularization()
        return loss/len(self.layers)

    def relprop(self, rel):
        for layer in self.layers:
            emb_rel, rel = layer.relprop(rel)
        return rel

class PredictModel(BaseModel):
    """
    This class implements the PredictModel, a causality-aware transformer-based deep learning model
    for making predictions on time series data.

    Args:
        config (dict): Configuration dictionary containing data loader and architecture arguments.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        n_layers (int): Number of encoder layers.
        ffn_hidden (int): Hidden dimension in the feed forward layer. d_FFN in paper.
        drop_prob (float): Dropout probability (Not used in practice).
        tau (float): Temperature hyperparameter for attention softmax.

    Attributes:
        data_feature (dict): Data loader arguments.
        model_config (dict): Architecture arguments.
        input_window (int): Length of the input time series window.
        output_window (int): Length of the output time series window.
        series_num (int): Number of time series in the input.
        feature_dim (int): Dimension of each feature in the time series.
        output_dim (int): Dimension of the model's output.
        d_model (int): Dimension of the embedding vector. D_QK in paper.
        n_head (int): Number of attention heads. h in paper.
        n_layers (int): Number of encoder layers.
        ffn_hidden (int): Hidden dimension in the feed forward layer. d_FFN in paper.
        drop_prob (float): Dropout probability (Not used in practice).
        tau (float): Temperature hyperparameter for attention softmax.
        device (torch.device): Device for computation.
        encoder (Encoder): Instance of the Encoder class.
        fc (nn.Linear): Linear layer for prediction output.
    """

    def __init__(self, config, d_model, n_head, n_layers, ffn_hidden, drop_prob, tau):
        super().__init__()
        self.data_feature = config['data_loader']['args']
        self.model_config = config['arch']['args']

        self.input_window = self.data_feature.get('time_step')
        self.output_window = self.data_feature.get('output_window')
        self.series_num = self.data_feature.get('series_num')
        self.feature_dim = self.data_feature.get('feature_dim')
        self.output_dim = self.data_feature.get('output_dim')

        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden
        self.drop_prob = drop_prob
        self.tau = tau

        self.device, device_ids = prepare_device(config['n_gpu'])

        self.encoder = Encoder(series_num=self.series_num,
                               input_window=self.input_window,
                               feature_dim=self.feature_dim,
                               d_model=self.d_model, 
                               n_head=self.n_head,
                               n_layers=self.n_layers,
                               ffn_hidden=self.ffn_hidden,
                               drop_prob=self.drop_prob,
                               tau=self.tau,
                               device=self.device)
        
        self.fc = Linear(in_features=self.feature_dim, out_features=self.output_dim, bias=True)
        # He Initialization
        self.fc.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.output_dim)))
    def forward(self, x):
        # x = [batch_size, input_window, series_num, feature_dim]
        x = x.permute(0, 2, 1, 3)  # [batch_size, series_num, input_window, feature_dim]
        out = self.encoder(x) # [batch_size, series_num, input_window, feature_dim]
        out = self.fc(out) # [batch_size, series_num, input_window, output_dim]
        out = out.permute(0, 2, 1, 3)
        out = out[:,-self.output_window:,...]
        return out
    
    def regularization(self):
        return self.encoder.regularization()
    
    def relprop(self, rel):
        pad = torch.zeros((rel.shape[0],self.input_window-self.output_window,rel.shape[2],rel.shape[3])).to(self.device)
        rel = torch.cat((pad,rel),1)
        rel = rel.permute(0, 2, 1, 3)
        rel = self.fc.relprop(rel)
        rel = self.encoder.relprop(rel)
        rel = rel.permute(0, 2, 1, 3)
        return rel
