import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from dgl import function as fn

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x,2),dim=1,keepdim=False) / 2.)

class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        '''
        Information Aggregation 
        '''
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
                 

        ## three types of aggregators
        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # GCN Aggregator, Equ(6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # GraphSage Aggregator, Equ(7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # Bi-Interation Aggregator, Equ(8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      
        else:
            raise NotImplementedError
        
    def forward(self, mode, g, entity_embed):
        '''
        mode: aggregator tyep, gcn/graphsage/bi-interaction
        g: dgl garph
        '''

        g = g.local_var()
        g.ndata['node'] = entity_embed

        # Equation (3) & (10)
        # Use custom function to ensure deterministic behavior when predicting
        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))                        

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))      

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))                      
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))                      
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out
        


class KGAT(nn.Module):
    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  
        self.mess_dropout = eval(args.mess_dropout)  
        self.n_layers = len(eval(args.conv_dim_list)) 

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        # Embedding
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim)
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))  
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

    def edge_softmax_fix(self, graph, score):
        def reduce_sum(nodes):
            accum = torch.sum(nodes.mailbox['temp'], 1)
            return {'out_sum': accum}
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(fn.copy_e('out', 'temp'), reduce_sum)
        graph.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def att_score(self, edges):
        # Equation (4)
        r_mul_t = torch.matmul(self.entity_user_embed(edges.src['id']), self.W_r)                       
        r_mul_h = torch.matmul(self.entity_user_embed(edges.dst['id']), self.W_r)                     
        r_embed = self.relation_embed(edges.data['type'])                                              
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)  
        return {'att': att}

    def compute_attention(self, g):
        g = g.local_var() 
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)  
            self.W_r = self.W_R[i] 
            g.apply_edges(self.att_score, edge_idxs) 

        # Equation (5)
        g.edata['att'] = self.edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    def calc_kg_loss(self, h, r, pos_t, neg_t):

        r_embed = self.relation_embed(r)                 
        W_r = self.W_R[r]                                

        h_embed = self.entity_user_embed(h)              
        pos_t_embed = self.entity_user_embed(pos_t)      
        neg_t_embed = self.entity_user_embed(neg_t)      

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)            
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)    
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)   

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def cf_embedding(self, mode, g):
        g = g.local_var()
        ego_embed = self.entity_user_embed(g.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(mode, g, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11) 
        all_embed = torch.cat(all_embed, dim=1)         
        return all_embed

    def cf_score(self, mode, g, user_ids, item_ids):

        all_embed = self.cf_embedding(mode, g)     
        user_embed = all_embed[user_ids]             
        item_embed = all_embed[item_ids]            

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))   
        return cf_score

    def calc_cf_loss(self, mode, g, user_ids, item_pos_ids, item_neg_ids):
        all_embed = self.cf_embedding(mode, g)                      
        user_embed = all_embed[user_ids]                            
        item_pos_embed = all_embed[item_pos_ids]                   
        item_neg_embed = all_embed[item_neg_ids]                    

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   

        # Equation (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss



    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)


class KGAT_ablation(KGAT):
    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None,
                 ablation_kge=True, ablation_att=True):

        super(KGAT_ablation, self).__init__(args, n_users, n_entities, n_relations, user_pre_embed, item_pre_embed)

        assert ablation_kge or ablation_att

        self.ablation_kge = ablation_kge
        self.ablation_att = ablation_att

    def calc_kg_loss(self, h, r, pos_t, neg_t):

        r_embed = self.relation_embed(r)
        W_r = self.W_R[r]

        h_embed = self.entity_user_embed(h)
        pos_t_embed = self.entity_user_embed(pos_t)
        neg_t_embed = self.entity_user_embed(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)

        # apply ablation on TransR embedding component
        if not self.ablation_kge:
            kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
            kg_loss = torch.mean(kg_loss)
            loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        else:
            loss = self.kg_l2loss_lambda * l2_loss

        return loss

    def att_score(self, edges):
        r_mul_t = torch.matmul(self.entity_user_embed(edges.src['id']), self.W_r)
        r_mul_h = torch.matmul(self.entity_user_embed(edges.dst['id']), self.W_r)
        r_embed = self.relation_embed(edges.data['type'])
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)

        # apply ablation on attention
        return {'att': att} if not self.ablation_att else {'att': torch.zeros_like(att)}