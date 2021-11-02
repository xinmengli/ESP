import argparse
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from gnn_sp import GNN_graphpred
from mlp_mt import MLP_MT
from utils import compute_data_split, batch_data_list, convert_candidate_to_data_list, compute_lda_feature
num_atom_type = 27


def train(epoch):
    model.train()

    for epoch_i in tqdm(range(epoch)):
        train_loss = []
        train_cosine = []
        for train_data_batch in train_loader:
            train_data_batch = train_data_batch.to(device)

            optimizer.zero_grad()

            out, loss_mt_sum = model(train_data_batch.x, train_data_batch.edge_index, train_data_batch.edge_attr,
                                     train_data_batch.batch, train_data_batch.instrument,
                                     train_data_batch.fp, train_data_batch.shift, train_data_batch.lda_feature,
                                     train_data_batch.ontology_feature)
            loss_cosine = -F.cosine_similarity(out, train_data_batch.y).mean()
            loss = loss_cosine + loss_mt_sum

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_cosine.append(-loss_cosine.item())

        val_cosine = eval(model, val_loader)
        print("epoch %d, train loss %.2f, train cosine %.2f, eval cosine similarity %.2f" % (
            epoch_i + 1, np.mean(train_loss), np.mean(train_cosine), val_cosine)
              )

        torch.save(model.state_dict(), './results/best_model_' + args.model_file_suffix + '.pt')


@torch.no_grad()
def eval(model, loader):
    is_train = model.training

    model.eval()
    eval_cosine = []
    for eval_data_batch in loader:
        eval_data_batch = eval_data_batch.to(device)

        out, _ = model(eval_data_batch.x, eval_data_batch.edge_index, eval_data_batch.edge_attr, eval_data_batch.batch,
                    eval_data_batch.instrument, eval_data_batch.fp, eval_data_batch.shift)
        loss = -F.cosine_similarity(out, eval_data_batch.y).mean()
        eval_cosine.append(-loss.item())

    if is_train:
        model.train()

    return np.mean(eval_cosine)


@torch.no_grad()
def compute_rank(model, stop_i=-1):
    model.load_state_dict(torch.load('./results/best_model_' + args.model_file_suffix + '.pt', map_location='cpu'))

    is_train = model.training

    model.eval()
    test_data_list = [data[i] for i in range(len(data)) if test_mask[i]]

    with open('./data/torch_tecand_1000bin_te_cand100.pkl', 'rb') as fi:
        test_candidate_dict = pickle.load(fi)

    rank = []
    loss_list = []
    for i in tqdm(range(len(test_data_list))):
        if i > 0 and i % 100 == 0:
            print("Average rank %.3f +- %.3f" % (np.mean(rank), np.std(rank)))
        if stop_i != -1 and i >= stop_i:
            print("Average rank %.3f +- %.3f" % (np.mean(rank), np.std(rank)))
            return rank
        test_data = test_data_list[i]
        test_data[2][1] = test_data[2][1][:, :num_atom_type]
        test_inchi_key = test_data[0]
        test_candidate_val = test_candidate_dict[test_inchi_key]

        test_candidate = convert_candidate_to_data_list(test_data, copy.deepcopy(test_candidate_val))

        rank_data_batch = batch_data_list([test_data] + test_candidate)
        rank_data_batch = rank_data_batch.to(device)

        out, _ = model(rank_data_batch.x, rank_data_batch.edge_index, rank_data_batch.edge_attr, rank_data_batch.batch,
                    rank_data_batch.instrument, rank_data_batch.fp, rank_data_batch.shift)
        loss = -F.cosine_similarity(out, rank_data_batch.y)
        loss = loss.cpu().numpy()
        test_rank = (loss[0] > loss[2:]).sum() + 1  # skip itself

        rank.append(test_rank)
        loss_list.append(loss[0])

    with open('./ranks.csv', 'w') as f:
        for i in range(len(rank)):
            f.write(str(rank[i])+',')

    if is_train:
        model.train()

    return rank, loss_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # cluster parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--model_file_suffix', type=str, default='')

    # training parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--l2norm', type=float, default=0.0)
    parser.add_argument('--drop_ratio', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    # model parameters
    parser.add_argument('--hidden_dims', type=int, default=1024)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--JK', type=str, default="last")
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--model', type=str, default='gnn')
    parser.add_argument('--disable_mt_lda', action='store_true')
    parser.add_argument('--correlation_mat_rank', type=int, default=0)

    parser.add_argument('--mt_lda_weight', type=float, default=0.01)
    parser.add_argument('--correlation_mix_residual_weight', type=float, default=0.7)

    parser.add_argument('--disable_two_step_pred', action='store_true')
    parser.add_argument('--disable_reverse', action='store_true')
    parser.add_argument('--disable_fingerprint', action='store_true')
    parser.add_argument('--disable_mt_fingerprint', action='store_true')
    parser.add_argument('--disable_mt_ontology', action='store_true')
    parser.add_argument('--full_dataset', action='store_true')

    args = parser.parse_args()
    print(str(args))
    with open('./data/torch_trvate_1000bin.pkl', 'rb') as fi:
        data = pickle.load(fi)

    if args.full_dataset:
        with open('./data/trvate_idx.pkl', 'rb') as fi:
            split = pickle.load(fi)
        split = (split[0], split[1], split[2])

    else:
        # M+H data set (default)
        with open('./data/MHfilter_trvate_idx.pkl', 'rb') as fi:
            split = pickle.load(fi)
        with open("./data/filter_te_idx.pkl", 'rb') as f:
            filter_te_idx = pickle.load(f)
        split = (split[0], split[1], filter_te_idx)

    train_mask, val_mask, test_mask = compute_data_split(len(data), random=False, split=split)

    train_ms = [data[i][4] for i in range(len(data)) if train_mask[i]]
    train_ms = np.vstack(train_ms)
    lda_topic = compute_lda_feature(train_ms)

    train_data = [data[i] for i in range(len(data)) if train_mask[i]]
    val_data = [data[i] for i in range(len(data)) if val_mask[i]]

    # append lda feature
    if not args.disable_mt_lda:
        for i in range(len(train_data)):
            train_data[i].append(lda_topic[i])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=batch_data_list)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=batch_data_list)

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

    n_ms = data[0][4].shape[0]
    if args.model == 'gnn':
        model = GNN_graphpred(num_layer=args.num_hidden_layers,
                              emb_dim=args.hidden_dims,
                              num_tasks=n_ms, JK=args.JK, drop_ratio=args.drop_ratio, graph_pooling=args.graph_pooling,
                              gnn_type="gin",
                              disable_two_step_pred=args.disable_two_step_pred,
                              disable_reverse=args.disable_reverse,
                              disable_fingerprint=args.disable_fingerprint,
                              disable_mt_fingerprint=args.disable_mt_fingerprint,
                              disable_mt_lda=args.disable_mt_lda,
                              disable_mt_ontology=args.disable_mt_ontology,
                              correlation_mat_rank=args.correlation_mat_rank,
                              mt_lda_weight=args.mt_lda_weight,
                              correlation_mix_residual_weight=args.correlation_mix_residual_weight
                              ).to(device)
    elif args.model == 'mlp':
        # model = MLP(emb_dim=args.hidden_dims, output_dim=n_ms, drop_ratio=args.drop_ratio).to(device)
        model = MLP_MT(emb_dim=args.hidden_dims, output_dim=n_ms, drop_ratio=args.drop_ratio,
                       disable_mt_lda=args.disable_mt_lda,
                       correlation_mat_rank=args.correlation_mat_rank,
                       mt_lda_weight=args.mt_lda_weight,
                       correlation_mix_residual_weight=args.correlation_mix_residual_weight).to(device)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # train(epoch=args.epochs)

    test_data_rank, test_data_loss = compute_rank(model)
    test_data_rank = np.array(test_data_rank)
    test_data_loss = np.array(test_data_loss)
    np.savez('./results/prediction_' + args.model_file_suffix, test_data_rank=test_data_rank, test_data_loss=test_data_loss)

    print("Average rank %.3f +- %.3f" % (test_data_rank.mean(), test_data_rank.std()))
    for i in range(1, 21):
        print("Rank at %d %.3f" % (i, (test_data_rank <= i).sum() / float(test_data_rank.shape[0])))

