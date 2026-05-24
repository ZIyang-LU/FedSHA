import argparse
import os
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, List, Dict, Tuple
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid, Amazon
import torch_geometric
import torch_sparse
from torch import nn


METHOD_ALIASES = {
    'fedgcn-ala': 'fedsha',
    'fedgcn-ala-1hop': 'fedsha-1hop',
    'fedgcn-ala-0hop': 'fedsha-w/o-se',
    'fedgcn-1hop': 'fedgcn',
}

SUPPORTED_METHODS = {
    'fedavg-gcn',
    'fedgcn',
    'fedprox',
    'fedala-gcn',
    'fedsha',
    'fedsha-1hop',
    'fedsha-2hop',
    'fedsha-w/o-se',
    'fedsha-w/o-pa',
}

PERSONALIZED_METHODS = {
    'fedala-gcn',
    'fedsha',
    'fedsha-1hop',
    'fedsha-2hop',
    'fedsha-w/o-se',
}


def normalize_method(method: str) -> str:
    method = method.lower()
    method = METHOD_ALIASES.get(method, method)
    if method not in SUPPORTED_METHODS:
        supported = ', '.join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported method: {method}. Supported methods: {supported}")
    return method


# --- 1. GCN 模型 (原始 - 适用于 Cora/Citeseer) ---
class GCN(torch.nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float = 0.5, NumLayers: int = 2):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        # 第一层
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
        # 中间层 (如果有)
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        # 输出层
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class AmazonGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, NumLayers=3):
        super(AmazonGCN, self).__init__()

        # 1) Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        # 2) Hidden GCN layers (nhid -> nhid)
        for _ in range(NumLayers - 1):
            # ✅ 最小改动点 1：cached=False 更适配联邦/子图/动态邻接
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=False))
            self.lns.append(nn.LayerNorm(nhid))

        # ✅ 最小改动点 2：分类头用 Linear，避免最后一层再做图传播放大切图偏差
        self.classifier = nn.Linear(nhid, nclass)

        self.dropout = dropout
        self.num_layers = NumLayers

    def forward(self, x, adj_t):
        # Step 1: encode
        x = self.feature_encoder(x)

        # Step 2: propagation (residual + LN)
        for i, conv in enumerate(self.convs):
            x_in = x
            x = conv(x, adj_t)
            x = self.lns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in

        # Step 3: classify (no more graph propagation here)
        x = self.classifier(x)
        return torch.log_softmax(x, dim=-1)

# --- 2. ALA 模块 (适配 GCN) ---
class AdaptiveLocalAggregation:
    def __init__(self,
                 cid: int,
                 loss: torch.nn.Module,
                 features: torch.Tensor,
                 adj: Any,
                 labels: torch.Tensor,
                 idx_train: torch.Tensor,
                 rand_percent: int,
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = 'cpu',
                 threshold: float = 0.1,
                 num_pre_loss: int = 10) -> None:
        self.cid = cid
        self.loss = loss
        self.features = features
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None
        self.start_phase = True

    def adaptive_local_aggregation(self,
                                   global_model: torch.nn.Module,
                                   local_model: torch.nn.Module) -> None:

        # --- [修改点 1]：更智能的参数筛选，排除 BatchNorm ---
        params_g_dict = dict(global_model.named_parameters())
        params_local_dict = dict(local_model.named_parameters())

        all_names = list(params_local_dict.keys())

        # 简单策略：只针对包含 "conv" 或 "lin" 或 "predictions" 的层的 weight/bias
        # 排除 "bn", "norm"
        target_names = []
        for name in all_names:
            if 'bn' in name or 'norm' in name:
                continue
            target_names.append(name)

        # 强制修正：对于 Amazon Computers，强烈建议只微调最后 2 个参数
        num_params_to_adapt = self.layer_idx * 2  # 假设每层有 W 和 b

        adapt_names = target_names[-num_params_to_adapt:] if self.layer_idx > 0 else []

        # 1. 保护/重置低层参数 (Low-layers) 为全局参数
        with torch.no_grad():
            for name, param in params_local_dict.items():
                if name not in adapt_names:
                    param.data = params_g_dict[name].data.clone()

        # 如果没有需要 adapt 的，直接返回
        if not adapt_names:
            return

        # 2. 准备 High-layers (需要学习权重的参数)
        params_p = [params_local_dict[name] for name in adapt_names]
        params_gp = [params_g_dict[name] for name in adapt_names]

        # 临时模型
        model_t = copy.deepcopy(local_model)
        params_t_dict = dict(model_t.named_parameters())
        params_tp = [params_t_dict[name] for name in adapt_names]

        # 冻结所有非目标参数
        for name, param in model_t.named_parameters():
            if name not in adapt_names:
                param.requires_grad = False

        # 初始化权重 weights
        if self.weights is None or len(self.weights) != len(params_p):
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # 初始化临时参数 p_t = p_local + (p_global - p_local) * w
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param.data + (param_g.data - param.data) * weight.data

        # --- 权重学习循环 ---
        losses = []
        cnt = 0
        num_sample = int(len(self.idx_train) * (self.rand_percent / 100))
        num_sample = max(num_sample, 1)

        while True:
            rand_perm = torch.randperm(len(self.idx_train))
            rand_indices = self.idx_train[rand_perm[:num_sample]]

            model_t.zero_grad()
            output = model_t(self.features, self.adj)
            loss_value = self.loss(output[rand_indices], self.labels[rand_indices])
            loss_value.backward()

            with torch.no_grad():
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    diff = param_g.data - param.data
                    grad_weight = param_t.grad * diff
                    # 梯度更新
                    weight.data = torch.clamp(weight.data - self.eta * grad_weight, 0, 1)

                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param.data + (param_g.data - param.data) * weight.data

            losses.append(loss_value.item())
            cnt += 1

            if not self.start_phase:
                if cnt >= 5: break  # 稍微多跑几步
            else:
                # 收敛检测
                if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                    break
                if cnt > 100: break

        self.start_phase = False

        # 写回
        with torch.no_grad():
            for param, param_t in zip(params_p, params_tp):
                param.data = param_t.data.clone()


# --- 3. 工具函数 (数据划分与加载) ---
def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def label_dirichlet_partition(labels: np.array, N: int, K: int, n_parties: int, beta: float) -> list:
    min_size = 0
    min_require_size = 10
    split_data_indexes = []

    while min_size < min_require_size:
        idx_batch: list[list[int]] = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])
    return split_data_indexes


def build_structural_enhanced_subgraphs(edge_index: torch.Tensor, split_data_indexes: list, num_clients: int,
                                        se_hops: int, idx_train: torch.Tensor, idx_test: torch.Tensor) -> tuple:
    communicate_indexes = []
    in_com_train_data_indexes = []
    edge_indexes_clients = []

    for i in range(num_clients):
        communicate_index = split_data_indexes[i].clone()
        if se_hops == 0:
            # 0-hop means only the nodes themselves, no neighbors
            communicate_index, current_edge_index, _, __ = torch_geometric.utils.k_hop_subgraph(communicate_index, 0,
                                                                                                edge_index,
                                                                                                relabel_nodes=True)
        else:
            for hop in range(se_hops):
                if hop != se_hops - 1:
                    communicate_index = \
                        torch_geometric.utils.k_hop_subgraph(communicate_index, 1, edge_index, relabel_nodes=False)[0]
                else:
                    communicate_index, current_edge_index, _, __ = torch_geometric.utils.k_hop_subgraph(
                        communicate_index, 1, edge_index, relabel_nodes=True)

        communicate_index = communicate_index.to("cpu")
        current_edge_index = current_edge_index.to("cpu")
        communicate_indexes.append(communicate_index)

        current_edge_index = torch_sparse.SparseTensor(
            row=current_edge_index[0], col=current_edge_index[1],
            sparse_sizes=(len(communicate_index), len(communicate_index))
        )
        edge_indexes_clients.append(current_edge_index)

        inter_train = intersect1d(split_data_indexes[i], idx_train)
        in_com_train_data_indexes.append(torch.searchsorted(communicate_index, inter_train).clone())

    in_com_test_data_indexes = []
    for i in range(num_clients):
        inter_test = intersect1d(split_data_indexes[i], idx_test)
        in_com_test_data_indexes.append(torch.searchsorted(communicate_indexes[i], inter_test).clone())

    return communicate_indexes, in_com_train_data_indexes, in_com_test_data_indexes, edge_indexes_clients


def load_data(dataset_name: str) -> Tuple[
    torch.Tensor, torch.Tensor, SparseTensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = os.path.join(os.getcwd(), 'data', dataset_name)
    ds_name_lower = dataset_name.lower()
    if ds_name_lower in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name)
        data = dataset[0]
    elif ds_name_lower in ['computers', 'amazon-computers', 'amazon_computers', 'amazon']:
        dataset = Amazon(root=path, name='Computers')
        data = dataset[0]
    elif ds_name_lower in ['photo', 'amazon-photo', 'amazon_photo']:
        dataset = Amazon(root=path, name='Photo')
        data = dataset[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")





    features = data.x
    labels = data.y
    edge_index = data.edge_index
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()

    if hasattr(data, 'train_mask') and data.train_mask is not None:
        idx_train = data.train_mask.nonzero(as_tuple=False).flatten()
        idx_val = data.val_mask.nonzero(as_tuple=False).flatten() if data.val_mask is not None else torch.tensor([],
                                                                                                                 dtype=torch.long)
        idx_test = data.test_mask.nonzero(as_tuple=False).flatten() if data.test_mask is not None else torch.tensor([],
                                                                                                                    dtype=torch.long)

    else:
        num_nodes = data.num_nodes
        num_classes = int(labels.max().item() + 1)

        idx_train_list, idx_val_list, idx_test_list = [], [], []

        labels_np = labels.cpu().numpy()
        for c in range(num_classes):
            idx_c = np.where(labels_np == c)[0]
            np.random.shuffle(idx_c)

            n_c = len(idx_c)
            n_train = int(0.2 * n_c)
            n_val = int(0.2 * n_c)
            # test 用剩余的 20%（含取整误差）
            idx_train_list += idx_c[:n_train].tolist()
            idx_val_list   += idx_c[n_train:n_train + n_val].tolist()
            idx_test_list  += idx_c[n_train + n_val:].tolist()

        idx_train = torch.tensor(np.sort(idx_train_list), dtype=torch.long)
        idx_val   = torch.tensor(np.sort(idx_val_list), dtype=torch.long)
        idx_test  = torch.tensor(np.sort(idx_test_list), dtype=torch.long)

    print(f"Loaded {dataset_name} with {data.num_nodes} nodes, feature dim {features.shape[1]}.")
    return features, edge_index, adj_t, labels, idx_train, idx_val, idx_test
    # else:
    #     # 随机划分
    #     num_nodes = data.num_nodes
    #     perm = np.random.permutation(num_nodes)
    #
    #     n_train = int(0.2 * num_nodes)
    #     n_val = int(0.2 * num_nodes)
    #
    #     idx_train = torch.tensor(np.sort(perm[:n_train]), dtype=torch.long)
    #     idx_val   = torch.tensor(np.sort(perm[n_train:n_train + n_val]), dtype=torch.long)
    #     idx_test  = torch.tensor(np.sort(perm[n_train + n_val:]), dtype=torch.long)
    #     print(f"Loaded {dataset_name} with {data.num_nodes} nodes, feature dim {features.shape[1]}.")
    #     return features, edge_index, adj_t, labels, idx_train, idx_val, idx_test

# --- 4. Trainer 类 (集成 ALA) ---
class FedSHATrainer:
    def __init__(self, client_id, edge_index_client, labels_client, features_client,
                 in_com_train_data_indexes, in_com_test_data_indexes,
                 args_hidden, class_num, device, args, dataset_name):

        self.client_id = client_id
        self.device = device
        self.args = args
        # 保存最近一次下发的全局模型参数（state_dict）
        self.global_state_dict = None
        # self.edge_index_client = edge_index_client
        self.edge_index_client = edge_index_client.to(device)
        self.labels_client = labels_client.to(device)
        self.features_client = features_client.to(device)
        self.in_com_train_data_indexes = in_com_train_data_indexes.to(device)
        self.in_com_test_data_indexes = in_com_test_data_indexes.to(device)

        # 根据数据集选择模型
        ds_lower = dataset_name.lower()
        if ds_lower in ['computers', 'amazon-computers', 'amazon']:
            self.model = AmazonGCN(
                nfeat=features_client.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers
            ).to(device)
        else:
            self.model = GCN(
                features_client.shape[1],
                args_hidden,
                class_num,
                NumLayers=args.num_layers
            ).to(device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
        self.criterion = torch.nn.NLLLoss()

        self.train_losses, self.train_accuracies = [], []
        self.test_losses, self.test_accuracies = [], []

        # --- [MODIFICATION] 更新 ALA 初始化判断逻辑 ---
        # Methods with personalized aggregation initialize ALA.
        if args.method in PERSONALIZED_METHODS:
            self.ala = AdaptiveLocalAggregation(
                cid=client_id,
                loss=self.criterion,
                features=self.features_client,
                adj=self.edge_index_client,
                labels=self.labels_client,
                idx_train=self.in_com_train_data_indexes,
                rand_percent=args.rand_percent,
                layer_idx=args.layer_idx,
                eta=args.eta,
                device=self.device
            )

    def get_local_model_state(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def set_global_model_state(self, state_dict: Dict[str, torch.Tensor]):

        # 保存一份全局参数用于 FedProx 中的正则项
        self.global_state_dict = {k: v.clone().to(self.device) for k, v in state_dict.items()}

        # --- [MODIFICATION] 更新 ALA 应用判断逻辑 ---
        if self.args.method in PERSONALIZED_METHODS:
            global_model_temp = copy.deepcopy(self.model)
            global_model_temp.load_state_dict(state_dict)
            global_model_temp.to(self.device)
            self.ala.adaptive_local_aggregation(global_model_temp, self.model)
            del global_model_temp
        else:
            self.model.load_state_dict(state_dict)

    def local_train(self):
        self.model.train()
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            output = self.model(self.features_client, self.edge_index_client)
            loss = self.criterion(output[self.in_com_train_data_indexes],
                                  self.labels_client[self.in_com_train_data_indexes])
            # --- FedProx: 添加 proximal 项 ---
            if self.args.method == 'fedprox' and self.global_state_dict is not None:
                prox_mu = getattr(self.args, 'prox_mu', 0.0)
                if prox_mu > 0.0:
                    prox_reg = 0.0
                    for name, param in self.model.named_parameters():
                        # 只对浮点参数（权重/偏置）计算prox
                        if name in self.global_state_dict and param.requires_grad and param.data.is_floating_point():
                            diff = param - self.global_state_dict[name].to(self.device)
                            prox_reg = prox_reg + 0.5 * prox_mu * torch.sum(diff * diff)
                    loss = loss + prox_reg

            loss.backward()
            # [新增] 梯度裁剪：防止梯度爆炸导致的突然掉点
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

        loss, acc = self._evaluate(self.in_com_train_data_indexes)
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)

        t_loss, t_acc = self._evaluate(self.in_com_test_data_indexes)
        self.test_losses.append(t_loss)
        self.test_accuracies.append(t_acc)

        return len(self.in_com_train_data_indexes)

    def _evaluate(self, indices: torch.Tensor) -> Tuple[float, float]:
        if len(indices) == 0: return 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.features_client, self.edge_index_client)
            loss = self.criterion(output[indices], self.labels_client[indices]).item()
            pred = output[indices].max(1)[1]
            correct = pred.eq(self.labels_client[indices]).sum().item()
            accuracy = correct / len(indices)
        return loss, accuracy


# --- 5. Server 类 (已修复模型初始化问题) ---
class FederatedServer:
    def __init__(self, nfeat, nhid, nclass, device, trainers: List['FedSHATrainer'], args):
        self.device = device
        self.trainers = trainers
        self.args = args

        # --- FIX: 根据数据集选择正确的全局模型架构 ---
        ds_lower = args.dataset.lower()
        if ds_lower in ['computers', 'amazon-computers', 'amazon']:
            self.global_model = AmazonGCN(
                nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=0.5,
                NumLayers=args.num_layers
            ).to(device)
        else:
            self.global_model = GCN(nfeat, nhid, nclass, NumLayers=args.num_layers).to(device)
        # ---------------------------------------------

        # 初始化：分发全局模型给所有客户端
        global_state = self.global_model.state_dict()
        for trainer in self.trainers:
            trainer.model.load_state_dict(global_state)

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]]):
        if not client_updates: return
        total_weight = sum(w for _, w in client_updates)
        if total_weight == 0: return

        # FedAvg 聚合
        avg_state_dict = copy.deepcopy(client_updates[0][0])

        # 遍历所有参数
        for param in avg_state_dict:
            # 检查参数是否为浮点型 (weights, bias, running_mean 等)
            if avg_state_dict[param].is_floating_point():
                avg_state_dict[param] = torch.zeros_like(avg_state_dict[param])
                for client_state, weight in client_updates:
                    avg_state_dict[param] += client_state[param] * (weight / total_weight)
            else:
                # 对于整型参数 (如 num_batches_tracked)，不进行加权平均
                # 直接保留第一个客户端的值 (或者也可以取最大值，这里取第一个简单有效)
                avg_state_dict[param] = client_updates[0][0][param]

        self.global_model.load_state_dict(avg_state_dict)

        # 分发给客户端 (此时触发 ALA)
        for trainer in self.trainers:
            trainer.set_global_model_state(avg_state_dict)

    def train(self):
        client_updates = []
        for trainer in self.trainers:
            weight = trainer.local_train()
            state_dict = trainer.get_local_model_state()
            client_updates.append((state_dict, weight))

        self.aggregate(client_updates)

        # 仅统计测试集非空的客户端


# --- 6. 主程序 ---
if __name__ == "__main__":
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="computers", type=str,
                        choices=['cora', 'citeseer', 'computers', 'photo', 'amazon', 'pubmed'])
    # --- [MODIFICATION] 添加新的模型选项 ---
    parser.add_argument("-m", "--method", default="fedsha", type=str)
    parser.add_argument("-f", "--fedtype", default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument("-c", "--global_rounds", default=300, type=int)
    parser.add_argument("-i", "--local_step", default=5, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.1, type=float)
    parser.add_argument("-n", "--num_clients", "--n_trainer", dest="num_clients", default=10, type=int)
    parser.add_argument("-nl", "--num_layers", default=3, type=int)
    parser.add_argument("-nhop", "--se_hops", "--num_hops", dest="se_hops", default=2, type=int)
    parser.add_argument("-g", "--gpu", action="store_true")
    parser.add_argument("-iid_b", "-b", "--dirichlet_beta", "--iid_beta", dest="dirichlet_beta",
                        default=0.1, type=float)
    parser.add_argument("-r", "--repeat_time", default=1, type=int)

    # ALA Specific
    parser.add_argument('-et', "--eta", type=float, default=1, help="ALA weight learning rate")
    parser.add_argument('-s', "--rand_percent", type=int, default=80, help="ALA sample percent")
    parser.add_argument('-p', "--layer_idx", type=int, default=2, help="ALA layer index (from end)")
    # prox
    parser.add_argument("--prox_mu", type=float, default=0.1, help="FedProx proximal term coefficient (mu)")
    parser.add_argument("-seed", "--seed", default=2025, type=int)

    args = parser.parse_args()

    if args.fedtype is not None:
        args.method = args.fedtype
    args.method = normalize_method(args.method)

    if args.method in ['fedavg-gcn', 'fedprox', 'fedala-gcn', 'fedsha-w/o-se']:
        args.se_hops = 0
    elif args.method == 'fedsha-1hop':
        args.se_hops = 1
    elif args.method == 'fedsha-2hop':
        args.se_hops = 2

    # --- [MODIFICATION] 强制模型特定的 Hop 设置 ---

    # Amazon 建议参数
    if args.dataset.lower() in ['computers', 'amazon-computers', 'amazon']:
        args_hidden = 128
        args.learning_rate = 0.01
        args.num_layers = 3
        args.local_step = 5
        print(f"--- [Auto-Config] Detected Computers: Forcing LR={args.learning_rate}, Layers={args.num_layers} ---")
    else:
        args_hidden = 128
    print("Arguments:", args)
    setup_seed(args.seed)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features, edge_index_2xn, adj_t, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    class_num = labels.max().item() + 1

    labels = labels.to(device)
    features = features.to(device)
    adj_t = adj_t.to(device)

    final_accs = []

    for repeat in range(args.repeat_time):
        print(f"\n--- Repeat {repeat + 1}/{args.repeat_time} ---")

        split_indices_np = label_dirichlet_partition(
            labels.cpu().numpy(), len(labels), class_num, args.num_clients, beta=args.dirichlet_beta
        )
        split_indices = [torch.tensor(np.sort(idx)) for idx in split_indices_np]

        comm_idxs, train_idxs, test_idxs, edge_idxs_clients = build_structural_enhanced_subgraphs(
            edge_index_2xn, split_indices, args.num_clients, args.se_hops, idx_train, idx_test
        )

        trainers = []
        for i in range(args.num_clients):
            trainers.append(FedSHATrainer(
                i, edge_idxs_clients[i], labels[comm_idxs[i]], features[comm_idxs[i]],
                train_idxs[i], test_idxs[i], args_hidden, class_num, device, args, args.dataset
            ))

        server = FederatedServer(features.shape[1], args_hidden, class_num, device, trainers, args)

        # [新增] 用于记录每一轮的 Client-Avg 精度
        round_client_avg_history = []
        round_weighted_acc_history = []

        for _ in range(args.global_rounds):
            server.train()
            acc_and_n = [
                (t.test_accuracies[-1], len(t.in_com_test_data_indexes))
                for t in trainers
                if t.test_accuracies and len(t.in_com_test_data_indexes) > 0
            ]

            if acc_and_n:
                test_accs = [a for a, _ in acc_and_n]
                weights = [n for _, n in acc_and_n]
                weighted_acc = np.average(test_accs, weights=weights)
            else:
                weighted_acc = 0.0

            round_weighted_acc_history.append(weighted_acc)

            # --- [新增] 评估全局模型 ---

            # --- [修改核心]：计算当前轮次的 Client-Avg 并记录 ---
            # 获取所有客户端当前轮的测试精度
            current_round_accs = [t.test_accuracies[-1] for t in trainers if t.test_accuracies]

            if len(current_round_accs) > 0:
                # 计算算术平均 (Client-Avg)
                current_client_avg = np.mean(current_round_accs)
                round_client_avg_history.append(current_client_avg)
            else:
                round_client_avg_history.append(0.0)
            # ---------------------------------------------------

        # --- [修改核心]：计算最后 10 轮的平均值 (作为本次 Repeat 的最终结果) ---
        # 防止轮数少于10轮的情况
        last_n = 10

        # Client-Avg（你原来就有）
        last_client_avg = np.mean(round_client_avg_history[-last_n:])

        # 新增：Weighted
        last_weighted_avg = np.mean(round_weighted_acc_history[-last_n:])

        print(
            f"""
        Final Results (Repeat {repeat + 1}, Avg of last {min(last_n, len(round_client_avg_history))} rounds):
          Client-Avg Test Acc    : {last_client_avg:.4f}
          Weighted Test Acc      : {last_weighted_avg:.4f}
        """
        )
        # One-line summaries (easy to parse by scripts)
        last_n_used = min(last_n, len(round_client_avg_history))
        print(f"Final Test Acc (Repeat {repeat + 1}) [Avg of last {last_n_used} rounds Client-Avg]: {last_client_avg:.4f}")
        print(f"Final Test Acc (Repeat {repeat + 1}) [Avg of last {last_n_used} rounds Weighted-Avg]: {last_weighted_avg:.4f}")

        final_accs.append(last_client_avg)  # 如果你仍然以 Client-Avg 作为主指标
    if args.repeat_time > 1:
        print(
            f"\nAverage Test Acc over {args.repeat_time} runs: {np.mean(final_accs):.4f} (Std: {np.std(final_accs):.4f})")
