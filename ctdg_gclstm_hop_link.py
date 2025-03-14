import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM 
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from ctdg_gclstm_hop_utils import get_neighbor_sampler_ct

import wandb
import timeit



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, K=1):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(in_channels=node_feat_dim, 
                                out_channels=hidden_dim, 
                                K=K,) #K is the Chebyshev filter size
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight, h, c):
        r"""
        forward function for the model, 
        this is used for each snapshot
        h: node hidden state matrix from previous time
        c: cell state matrix from previous time
        """
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(2*in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, x_i_neigbors=None, x_j_neigbors=None):
        
        if x_i_neigbors is not None and x_i_neigbors.numel() > 0:
            x_i = torch.cat([x_i, x_i_neigbors], dim=1)
        if x_j_neigbors is not None and x_j_neigbors.numel() > 0:
            x_j = torch.cat([x_j, x_j_neigbors], dim=1)

        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def pad_sequences(node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                    nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
    """
    pad the sequences for nodes in node_ids
    :param node_ids: ndarray, shape (batch_size, )
    :param node_interact_times: ndarray, shape (batch_size, )
    :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
    :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
    :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
    :param patch_size: int, patch size
    :param max_input_sequence_length: int, maximal number of neighbors for each node
    :return:
    """
    assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
    max_seq_length = 0
    # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
    for idx in range(len(nodes_neighbor_ids_list)):
        assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
        if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
            # cut the sequence by taking the most recent max_input_sequence_length interactions
            nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
            nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
            nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
        if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
            max_seq_length = len(nodes_neighbor_ids_list[idx])

    # include the target node itself
    max_seq_length += 1
    if max_seq_length % patch_size != 0:
        max_seq_length += (patch_size - max_seq_length % patch_size)
    assert max_seq_length % patch_size == 0

    # pad the sequences
    # three ndarrays with shape (batch_size, max_seq_length)
    padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.longlong)
    padded_nodes_edge_ids = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.longlong)
    padded_nodes_neighbor_times = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.float32)

    for idx in range(len(node_ids)):
        padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
        padded_nodes_edge_ids[idx] = 0
        padded_nodes_neighbor_times[idx] = node_interact_times[idx]

        if len(nodes_neighbor_ids_list[idx]) > 0:
            padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
            padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
            padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

    # three ndarrays with shape (batch_size, max_seq_length)
    return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times


def test_tgb(
        neighbor_sampler,
        hop,
        h,
             h_0,
             c_0, 
             test_loader, 
             test_snapshots, 
             ts_list,
             node_feat,
             model, 
             link_pred,
             neg_sampler,
             evaluator,
             metric, 
             split_mode='val'):
    
    model.eval()
    link_pred.eval()

    perf_list = []
    ts_idx = min(list(ts_list.keys()))
    max_ts_idx = max(list(ts_list.keys()))

    for batch in test_loader:
        pos_src, pos_dst, pos_t, pos_msg = (
        batch.src,
        batch.dst,
        batch.t,
        batch.msg,
        )
        neg_batch_list = neg_sampler.query_batch(np.array(pos_src.cpu()), np.array(pos_dst.cpu()), np.array(pos_t.cpu()), split_mode=split_mode)
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.full((1 + len(neg_batch),), pos_src[idx], device=args.device)
            query_dst = torch.tensor(
                        np.concatenate(
                            ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                            axis=0,
                        ),
                        device=args.device,
                    )
            query_t = torch.full((1 + len(neg_batch),), pos_t[idx], device=args.device)
            

            if hop == -1 or hop == 0:
                with torch.no_grad():
                    y_pred = link_pred(h[query_src], h[query_dst])
                y_pred = y_pred.squeeze(dim=-1).detach()
            
            # ADD
            elif hop == 1:
                src_neighbors, src_neighbors_edge, src_neighbors_times = neighbor_sampler.get_all_first_hop_neighbors(node_ids=query_src.cpu(), node_interact_times=query_t.cpu())
                dst_neighbors, dst_neighbors_edge, dst_neighbors_times = neighbor_sampler.get_all_first_hop_neighbors(node_ids=query_dst.cpu(), node_interact_times=query_t.cpu())

                src_pad_neighbors, src_pad_neighbors_edge, src_pad_neighbors_times = \
                    pad_sequences(node_ids=query_src.cpu(), node_interact_times=query_t.cpu(), nodes_neighbor_ids_list=src_neighbors,
                                    nodes_edge_ids_list=src_neighbors_edge, nodes_neighbor_times_list=src_neighbors_times,
                                    patch_size=1, max_input_sequence_length=32)
                dst_pad_neighbors, dst_pad_neighbors_edge, dst_pad_neighbors_times = \
                    pad_sequences(node_ids=query_dst.cpu(), node_interact_times=query_t.cpu(), nodes_neighbor_ids_list=dst_neighbors,
                                    nodes_edge_ids_list=dst_neighbors_edge, nodes_neighbor_times_list=dst_neighbors_times,
                                    patch_size=1, max_input_sequence_length=32)

                with torch.no_grad():
                    y_pred = link_pred(h[query_src], h[query_dst],h[src_pad_neighbors].mean(dim=1), h[dst_pad_neighbors].mean(dim=1))
                y_pred = y_pred.squeeze(dim=-1).detach()  

            input_dict = {
            "y_pred_pos": np.array([y_pred[0].cpu()]),
            "y_pred_neg": np.array(y_pred[1:].cpu()),
            "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
        
        #* update the model now if the prediction batch has moved to next snapshot
        while (pos_t[-1] > ts_list[ts_idx] and ts_idx < max_ts_idx):
            with torch.no_grad():
                cur_index = test_snapshots[ts_idx]
                cur_index = cur_index.long().to(args.device)
                edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
                h = h.detach()
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            ts_idx += 1

    #* update to the final snapshot
    with torch.no_grad():
        cur_index = test_snapshots[max_ts_idx]
        cur_index = cur_index.long().to(args.device)
        edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
        h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
        h = h.detach()
        h_0 = h_0.detach()
        c_0 = c_0.detach()

    test_metrics = float(np.mean(np.array(perf_list)))

    return test_metrics, h, h_0, c_0



    








if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader

    set_random(args.seed)

    batch_size = args.batch_size

    #ctdg dataset
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets")
    full_data = dataset.get_TemporalData()
    full_data = full_data.to(args.device) # TemporalData(src=[4873540], dst=[4873540], t=[4873540], msg=[4873540, 1], y=[4873540])
    #get masks
    train_mask = dataset.train_mask # torch.Size([4873540])
    val_mask = dataset.val_mask # torch.Size([4873540])
    test_mask = dataset.test_mask # torch.Size([4873540])
    train_edges = full_data[train_mask] # TemporalData(src=[3413837], dst=[3413837], t=[3413837], msg=[3413837, 1], y=[3413837])
    val_edges = full_data[val_mask] # TemporalData(src=[730784], dst=[730784], t=[730784], msg=[730784, 1], y=[730784])
    test_edges = full_data[test_mask] # TemporalData(src=[728919], dst=[728919], t=[728919], msg=[728919, 1], y=[728919])

    #* set up TGB queries, this is only for val and test
    metric = dataset.eval_metric # mrr
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    min_dst_idx, max_dst_idx = int(full_data.dst.min()), int(full_data.dst.max())


    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="utg",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "gclstm",
            "dataset": args.dataset,
            "time granularity": args.time_scale,
            }
        )
    #! set up node features
    node_feat = dataset.node_feat
    if (node_feat is not None):
        node_feat = node_feat.to(args.device)
        node_feat_dim = node_feat.size(1)
    else:
        node_feat_dim = 256
        node_feat = torch.randn((full_data.num_nodes,node_feat_dim)).to(args.device)

    edge_feat_dim = 1
    hidden_dim = 256

    #* load the discretized version
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    num_epochs = args.max_epoch
    lr = args.lr

    # ADD: neighbor_sampler 
    train_neighbor_sampler = get_neighbor_sampler_ct(data=full_data[train_mask], sample_neighbor_strategy='recent', time_scale=args.time_scale)
    full_neighbor_sampler = get_neighbor_sampler_ct(data=full_data, sample_neighbor_strategy='recent', time_scale=args.time_scale)


    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print (f"Run {seed}")
        
        #* initialization of the model to prep for training
        model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device)
        node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)
        link_pred = LinkPredictor(hidden_dim, hidden_dim, 1,
                                2, 0.2).to(args.device)


        optimizer = torch.optim.Adam(
            set(model.parameters()) | set(link_pred.parameters()), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val = 0
        best_test = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print (f"--------------------{epoch}----------------------")
            train_start_time = timeit.default_timer()
            optimizer.zero_grad()
            total_loss = 0
            model.train()
            link_pred.train()
            snapshot_list = train_data['edge_index']
            # ADD: get timestep
            timestep_list = train_data['ts_map']
            h_0, c_0, h = None, None, None
            total_loss = 0
            for snapshot_idx in range(train_data['time_length']):

                optimizer.zero_grad()
                if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                    cur_index = snapshot_list[snapshot_idx]
                    cur_index = cur_index.long().to(args.device)
                    # TODO, also need to support edge attributes correctly in TGX
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)

                neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_index.shape[1],),
                        dtype=torch.long,
                        device=args.device,
                    )

                if args.hop == -1 or args.hop == 0:
                    pos_out = link_pred(h[pos_index[0]], h[pos_index[1]])
                    neg_out = link_pred(h[pos_index[0]], h[neg_dst])
                
                # ADD
                elif args.hop == 1:
                    # get timestep that used to find neighbors
                    node_interact_times = np.ones(pos_index[0].shape[0], dtype=int) * timestep_list[snapshot_idx]

                    src_neighbors, src_neighbors_edge, src_neighbors_times = train_neighbor_sampler.get_all_first_hop_neighbors(node_ids=pos_index[0], node_interact_times=node_interact_times)
                    dst_neighbors, dst_neighbors_edge, dst_neighbors_times = train_neighbor_sampler.get_all_first_hop_neighbors(node_ids=pos_index[1], node_interact_times=node_interact_times)

                    src_pad_neighbors, src_pad_neighbors_edge, src_pad_neighbors_times = \
                        pad_sequences(node_ids=pos_index[0], node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_neighbors,
                                        nodes_edge_ids_list=src_neighbors_edge, nodes_neighbor_times_list=src_neighbors_times,
                                        patch_size=1, max_input_sequence_length=32)
                    dst_pad_neighbors, dst_pad_neighbors_edge, dst_pad_neighbors_times = \
                        pad_sequences(node_ids=pos_index[1], node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_neighbors,
                                        nodes_edge_ids_list=dst_neighbors_edge, nodes_neighbor_times_list=dst_neighbors_times,
                                        patch_size=1, max_input_sequence_length=32)

                    
                    pos_out = link_pred(h[pos_index[0]], h[pos_index[1]],h[src_pad_neighbors].mean(dim=1), h[dst_pad_neighbors].mean(dim=1))
                    neg_out = link_pred(h[pos_index[0]], h[neg_dst], h[src_pad_neighbors].mean(dim=1), h[dst_pad_neighbors].mean(dim=1))


                loss = criterion(pos_out, torch.ones_like(pos_out))
                loss += criterion(neg_out, torch.zeros_like(neg_out))

                loss.backward()
                optimizer.step()

                total_loss += float(loss) / pos_index.shape[1]


                h_0 = h_0.detach()
                c_0 = c_0.detach()

            train_time = timeit.default_timer() - train_start_time
            print (f'Epoch {epoch}/{num_epochs}, Loss: {total_loss}')
            print ("Train time: ", train_time)
            
            #? Evaluation starts here
            val_snapshots = data['val_data']['edge_index']
            ts_list = data['val_data']['ts_map']
            val_loader = TemporalDataLoader(val_edges, batch_size=batch_size)
            evaluator = Evaluator(name=args.dataset)
            neg_sampler = dataset.negative_sampler
            dataset.load_val_ns()

            start_epoch_val = timeit.default_timer()
            val_metrics, h, h_0, c_0 = test_tgb(full_neighbor_sampler, args.hop, h, h_0, c_0, val_loader, val_snapshots, ts_list,
                node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='val')
            val_time = timeit.default_timer() - start_epoch_val
            print(f"Val {metric}: {val_metrics}")
            print ("Val time: ", val_time)
            if (args.wandb):
                wandb.log({"train_loss":(total_loss),
                        "val_" + metric: val_metrics,
                        "train time": train_time,
                        "val time": val_time,
                        })
                
            #! report test results when validation improves
            if (val_metrics > best_val):
                dataset.load_test_ns()
                test_snapshots = data['test_data']['edge_index']
                ts_list = data['test_data']['ts_map']
                test_loader = TemporalDataLoader(test_edges, batch_size=batch_size)
                neg_sampler = dataset.negative_sampler
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                test_metrics, h, h_0, c_0 = test_tgb(full_neighbor_sampler, args.hop, h, h_0, c_0, test_loader, test_snapshots, ts_list,
                node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='test')
                test_time = timeit.default_timer() - test_start_time
                best_val = val_metrics
                best_test = test_metrics

                print ("test metric is ", test_metrics)
                print ("test elapsed time is ", test_time)
                print ("--------------------------------")
                if ((epoch - best_epoch) >= args.patience and epoch > 1):
                    best_epoch = epoch
                    break
                best_epoch = epoch
        print ("run finishes")
        print ("best epoch is, ", best_epoch)
        print ("best val performance is, ", best_val)
        print ("best test performance is, ", best_test)
        print ("------------------------------------------")
