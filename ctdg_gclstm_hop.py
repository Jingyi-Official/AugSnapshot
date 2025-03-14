import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM 
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from ctdg_gclstm_hop_utils import get_neighbor_sampler

import wandb
import timeit


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, neighbor_sampler, K=1):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(in_channels=node_feat_dim, 
                                out_channels=hidden_dim, 
                                K=K,) #K is the Chebyshev filter size
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_sampler = neighbor_sampler

    def compute_joint_neighbor_features(self, src_nodes, dst_nodes, node_interact_times):
        """
        Compute joint neighborhood features for positional encoding.
        """
        src_feats, dst_feats = neighbor_sampler.compute_src_dst_node_temporal_embeddings(
            src_nodes.cpu().numpy(), dst_nodes.cpu().numpy(), node_interact_times.cpu().numpy()
        )
        src_feats = torch.tensor(src_feats, dtype=torch.float32, device=src_nodes.device)
        dst_feats = torch.tensor(dst_feats, dtype=torch.float32, device=dst_nodes.device)
        
        joint_feat = torch.cat([src_feats, dst_feats], dim=-1)  # Concatenate features
        return joint_feat

    def forward(self, x, edge_index, edge_weight, h, c, snapshot_ts):
        
        
        # Compute joint neighbor features as positional encodings
        src_node_ids = edge_index[0]
        dst_node_ids = edge_index[1]
        node_interact_times = snapshot_ts
        src_nodes_neighbor_ids_list, _, _ = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)
        dst_nodes_neighbor_ids_list, _, _ = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
        
        if not all(isinstance(item, np.ndarray) and item.size == 0 for item in src_nodes_neighbor_ids_list):
            src_neighbor_raw_features = x[src_nodes_neighbor_ids_list[0], :]
            dst_neighbor_raw_features = x[dst_nodes_neighbor_ids_list[0], :]

            # merge with node embeddings
            x[src_node_ids] += src_neighbor_raw_features
            x[dst_node_ids] += dst_neighbor_raw_features
        
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def test_tgb(h,
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
            with torch.no_grad():
                y_pred = link_pred(h[query_src], h[query_dst])
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


    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print (f"Run {seed}")
        
        #* initialization of the model to prep for training
        neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy='recent', time_scaling_factor=1e-06)
        model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, neighbor_sampler=neighbor_sampler, K=1).to(args.device)
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
            print ("------------------------------------------")
            train_start_time = timeit.default_timer()
            optimizer.zero_grad()
            total_loss = 0
            model.train()
            link_pred.train()
            snapshot_list = train_data['edge_index']
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
                    h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0, np.ones(edge_attr.shape[0], dtype=int) * timestep_list[snapshot_idx])
                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0, np.ones(edge_attr.shape[0], dtype=int) * timestep_list[snapshot_idx])

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)

                neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_index.shape[1],),
                        dtype=torch.long,
                        device=args.device,
                    )

                pos_out = link_pred(h[pos_index[0]], h[pos_index[1]])
                neg_out = link_pred(h[pos_index[0]], h[neg_dst])

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
            val_metrics, h, h_0, c_0 = test_tgb(h, h_0, c_0, val_loader, val_snapshots, ts_list,
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
                test_metrics, h, h_0, c_0 = test_tgb(h, h_0, c_0, test_loader, test_snapshots, ts_list,
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
