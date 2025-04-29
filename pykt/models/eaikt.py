import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
import numpy as np
from enum import IntEnum
from .qdkt import QueEmbedder
from .que_base_model import QueBaseModel, QueEmb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class EaiKTNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, n_blocks, dropout, d_ff=256,
                 kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="",
                 flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768):
        super().__init__()
        self.model_name = "eaikt"
        self.num_c = num_c
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_q = num_q
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.emb_size = emb_size

        if self.num_q > 0:
            self.difficult_param = nn.Embedding(self.num_q + 1, 1)
            self.q_embed_diff = nn.Embedding(self.num_q + 1,
                                             emb_size)  # question emb
            self.qa_embed_diff = nn.Embedding(2 * self.num_q + 1, emb_size)  # interaction emb
        self.que_emb = QueEmbedder(num_q, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
        self.qa_embed = nn.Embedding(2, self.emb_size)
        self.model = Architecture(num_q=num_q, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                  d_model=self.emb_size, d_feature=self.emb_size / num_attn_heads, d_ff=d_ff,
                                  kq_same=self.kq_same, model_type=self.model_type)
        self.out = nn.Sequential(
            nn.Linear(self.emb_size + self.emb_size,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_q + 1 and self.num_q > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q, c, r):
        q_embed_data = self.que_emb(q)  # BS, seqlen,  d_model# c_ct

        qa_embed_data = self.qa_embed(r) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q, c, r):
        q_embed_data, qa_embed_data = self.base_emb(q, c, r)
        if self.num_q > 0:
            q_embed_diff_data = self.q_embed_diff(q)
            pid_embed_data = self.difficult_param(q).sigmoid()
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data * 2
            qa_embed_diff_data = self.qa_embed_diff(r)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (qa_embed_diff_data + q_embed_diff_data)

        c_reg_loss = 0.
        d_output = self.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        return preds, c_reg_loss

class EaiKT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size, n_blocks=1, dropout=0.1, emb_type='qid', kq_same=1, final_fc_dim=512,
                 num_attn_heads=8, separate_qa=False, l2=1e-5, d_ff=256, emb_path="", flag_load_emb=False,
                 flag_emb_freezed=False, pretrain_dim=768, device='cpu', seed=0, **kwargs):
        model_name = "eaikt"
        super().__init__(model_name=model_name, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim,
                         device=device, seed=seed)
        self.model = EaiKTNet(num_q=num_q, num_c=num_c, emb_size=emb_size, n_blocks=n_blocks, dropout=dropout, d_ff=d_ff,
                             kq_same=kq_same, final_fc_dim=final_fc_dim, num_attn_heads=num_attn_heads,
                             separate_qa=separate_qa,
                             l2=l2, emb_type=emb_type, emb_path=emb_path, flag_load_emb=flag_load_emb,
                             flag_emb_freezed=flag_emb_freezed, pretrain_dim=pretrain_dim)
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")

    def train_one_step(self, data, process=True, weighted_loss=0):
        y, reg_loss, data_new = self.predict_one_step(data, return_details=True, process=process)
        loss = self.get_loss(y, data_new['rshft'], data_new['sm'], weighted_loss=weighted_loss)  # get loss
        loss = loss + reg_loss
        return y, loss

    def predict_one_step(self, data, return_details=False, process=True):
        data_new = self.batch_to_device(data, process=process)
        y, reg_loss = self.model(data_new['cq'].long(), data_new['cc'].long(), data_new['cr'].long())

        def sigmoid_inverse(x, epsilon=1e-8):
            return torch.log(x / (1 - x + epsilon) + epsilon)
        y = y[:, 1:]
        y = sigmoid_inverse(y)
        y = torch.sigmoid(y)
        if return_details:
            return y, reg_loss, data_new
        else:
            return y

class Architecture(nn.Module):
    def __init__(self, num_q, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data
        y = qa_pos_embed
        x = q_pos_embed

        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y,
                          apply_pos=True)
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask,
                zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)
    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output

def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(q.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(q.device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(
            x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(q.device)
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    total_effect = torch.clamp(torch.clamp(
        (dist_scores * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
