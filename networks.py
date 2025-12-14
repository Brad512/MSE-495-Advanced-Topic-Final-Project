import torch
import torch.nn as nn
from numpy import *
import torch.nn.init as init
from transformer_classes import PositionalEncoding, EncoderLayer

##Joint Backbone Network with two subnetworks for mean and variance
class SharedNetwork(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_classes, rank):
        super(SharedNetwork, self).__init__()
        ##add embeddings here
        self.rank = rank
        self.d_model = d_model

        self.embedding = nn.Embedding(src_vocab_size,self.d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc_mean = nn.Sequential(
            nn.Linear(d_model, 16),
            # nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Linear(16, 8),
            # nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Linear(8, num_classes)
            )

        self.fc_variance = nn.Sequential(
            nn.Linear(d_model, 16),
            # nn.BatchNorm1d(16),
            nn.ELU(), 
            nn.Linear(16, 8),
            # nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Linear(8, num_classes)
            )

        self.dropout = nn.Dropout(dropout)
        ### cnn layers        
        self.cnn1 = nn.Sequential( nn.Conv1d(self.d_model,128,2, stride=1), 
                                   # nn.BatchNorm1d(128),
                                   nn.ELU(),
                                   nn.Conv1d(128,32,2, stride=1),
                                   # nn.BatchNorm1d(32),
                                   nn.ELU(),
                                   nn.Conv1d(32,32,2, stride=1),
                                   # nn.BatchNorm1d(32),
                                   nn.ELU(),
                                   nn.Conv1d(32,32,2, stride=1),
                                   )

    def forward(self, classes, src_mask, seq_len, need_grad):
        classes = classes.permute((1,0)) ##Input dim = [N,L], output_dim = [L,N]
        src = self.embedding(classes)
        src = self.dropout(self.positional_encoding(src, self.rank))
        enc_output = src.permute(1,0,2)
        for _, enc_layer in enumerate(self.encoder_layers):
            enc_output, attn_prob, attn_grad = enc_layer(enc_output, src_mask, need_grad)
            '''attn_prob & attm: [N, heads,L,L]'''
            attn_prob = torch.mean(attn_prob, dim=1)
        enc_output = enc_output.permute(0,2,1) #In: [N,L,f], out: [N,f,L]
        cnn_out = self.cnn1(enc_output)
        avg_cnn_out = torch.mean(cnn_out, dim=2)
        mean_output = self.fc_mean(avg_cnn_out)
        variance_output = torch.exp(self.fc_variance(avg_cnn_out)) + 1e-6 #Exponential transformation with small epsilon for numerical stability

        return mean_output, variance_output, attn_prob, attn_grad


##Fully separate sub-networks for mean and variance prediction
class SplitNetwork(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_classes, rank):
        super(SplitNetwork, self).__init__()
        # Shared embedding layer
        self.rank = rank
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, self.d_model, padding_idx=0)
        
        # Mean prediction network
        self.positional_encoding_mean = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder_layers_mean = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.cnn_mean = nn.Sequential( nn.Conv1d(self.d_model, 128, 2, stride=1),
                                       nn.ReLU(),
                                       nn.Conv1d(128, 32, 2, stride=1),
                                       nn.ReLU(),
                                       nn.Conv1d(32, 32, 2, stride=1),
                                       nn.ReLU(),
                                       nn.Conv1d(32, 32, 2, stride=1),
                                       )
        self.fc_mean = nn.Sequential( nn.Linear(d_model, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, 8), 
                                      nn.ReLU(),
                                      nn.Linear(8, num_classes)
                                      )

        # Variance prediction network
        self.positional_encoding_variance = PositionalEncoding(self.d_model, max_seq_length)
        self.encoder_layers_variance = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.cnn_variance = nn.Sequential( nn.Conv1d(self.d_model, 128, 2, stride=1),
                                      nn.ReLU(),
                                      nn.Conv1d(128, 32, 2, stride=1),
                                      nn.ReLU(),
                                      nn.Conv1d(32, 32, 2, stride=1),
                                      nn.ReLU(),
                                      nn.Conv1d(32, 32, 2, stride=1),
                                      )
        self.fc_variance = nn.Sequential( nn.Linear(d_model, 16),
                                          nn.ReLU(),
                                          nn.Linear(16, 8),
                                          nn.ReLU(),
                                          nn.Linear(8, num_classes)
                                          )
        self.dropout = nn.Dropout(dropout)

    def forward(self, classes, src_mask, seq_len, need_grad):
        classes = classes.permute((1,0)) ##Input dim = [N,L], output_dim = [L,N]
        src = self.embedding(classes)
        # Mean prediction path
        src_mean = self.dropout(self.positional_encoding_mean(src, self.rank))
        enc_output_mean = src_mean.permute(1,0,2)
        for enc_layer in self.encoder_layers_mean:
            enc_output_mean, attn_prob_mean, attn_grad_mean = enc_layer(enc_output_mean, src_mask, need_grad)
            '''attn_prob & attm: [N, heads,L,L]'''
            attn_prob_mean = torch.mean(attn_prob_mean, dim=1)
        enc_output_mean = enc_output_mean.permute(0,2,1) #In: [N,L,f], out: [N,f,L]
        cnn_out_mean = self.cnn_mean(enc_output_mean)
        out_mean = torch.mean(cnn_out_mean, dim=2)
        mean_output = self.fc_mean(out_mean)
        # Variance prediction path
        src_var = self.dropout(self.positional_encoding_variance(src, self.rank))
        enc_output_var = src_var.permute(1,0,2)
        for enc_layer in self.encoder_layers_variance:
            enc_output_var, attn_prob_var, attn_grad_var = enc_layer(enc_output_var, src_mask, need_grad)
            attn_prob_var = torch.mean(attn_prob_var, dim=1)
        enc_output_var = enc_output_var.permute(0,2,1) #In: [N,L,f], out: [N,f,L]
        cnn_out_var = self.cnn_variance(enc_output_var)
        avg_cnn_out_var = torch.mean(cnn_out_var, dim=2)
        variance_output = torch.exp(self.fc_variance(avg_cnn_out_var)) + 1e-6 #Exponential transformation with small epsilon for numerical stability
        
        return mean_output, variance_output, attn_prob_mean, attn_grad_mean