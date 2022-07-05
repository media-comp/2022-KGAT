import os
import random
from time import time

from tqdm import tqdm, trange

import pandas as pd
import numpy as np
from DataLoader import *
from parser import *
from KGAT import *
from utility.metrics import *
from utility.parser import *

import torch
import torch.nn as nn
import torch.optim as optim
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # CPU or GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data
    data = DataLoader(args)

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)


    # model
    if not args.ablation_kge and not args.ablation_att:
        model = KGAT(args, data.n_users, data.n_entities, data.n_relations, None, None)
    else:
        model = KGAT_ablation(args, data.n_users, data.n_entities, data.n_relations, args.ablation_kge, args.ablation_att)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # move graph data to GPU
    if use_cuda:
        data.train_graph = data.train_graph.to(device)

    train_graph = data.train_graph

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list = []
    recall_list = []
    ndcg_list = []
    epoch = 0

    # train model
    for epoch in range(1, args.n_epoch + 1):
        epoch_start_time = time()
        model.train()

        # update attention scores
        with torch.no_grad():
            att = model('calc_att', train_graph)
        train_graph.edata['att'] = att
        print('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - epoch_start_time))

        # train cf
        cf_start_time = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            batch_start_time = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict)
            if use_cuda:
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model('calc_cf_loss', train_graph, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)

            cf_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - batch_start_time, cf_batch_loss.item(), cf_total_loss / iter))
        print('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - cf_start_time, cf_total_loss / n_cf_batch))

        # train kg
        kg_start_time = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            batch_start_time = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict)
            if use_cuda:
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            kg_batch_loss = model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

            kg_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - batch_start_time, kg_batch_loss.item(), kg_total_loss / iter))
        print('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - kg_start_time, kg_total_loss / n_kg_batch))

        print('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - epoch_start_time))


        # evaluate cf
        if (epoch % args.evaluate_every) == 0:
            cf_evaluate_start_time = time()
            _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
            print('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - cf_evaluate_start_time, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            if len(recall_list) - recall_list.index(max(recall_list))  - 1 >= args.stopping_steps:
                break
            elif recall_list.index(max(recall_list)) == len(recall_list) - 1:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                model_state_file = os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch))
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print('Save model on epoch {:04d}!'.format(epoch))

    metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K)]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)



def evaluate(model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
    model.eval()

    with torch.no_grad():
        att = model.compute_attention(train_graph)
    train_graph.edata['att'] = att

    n_users = len(test_user_dict.keys())
    # item_ids_batch = item_ids
    item_ids_batch = item_ids.cpu().numpy()

    cf_scores = []
    precision = []
    recall = []
    ndcg = []

    with torch.no_grad():
        for user_ids_batch in tqdm(user_ids_batches, desc='Evaluating Iteration'):
            cf_scores_batch = model('predict', train_graph, user_ids_batch, item_ids) 
            cf_scores_batch = cf_scores_batch.cpu()
            user_ids_batch = user_ids_batch.cpu().numpy()
            precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict, test_user_dict, user_ids_batch, item_ids_batch, K)

            cf_scores.append(cf_scores_batch.numpy())
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

    cf_scores = cf_scores[0]
    precision_k = sum(np.concatenate(precision)) / n_users
    recall_k = sum(np.concatenate(recall)) / n_users
    ndcg_k = sum(np.concatenate(ndcg)) / n_users
    return cf_scores, precision_k, recall_k, ndcg_k
    
def predict(args):
    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoader(args)

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    checkpoint = torch.load(args.pretrain_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if use_cuda:
        data.train_graph = data.train_graph.to(device)

    train_graph = data.train_graph
    # predict
    cf_scores, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))


if __name__ == '__main__':
    args = parser()
    train(args)