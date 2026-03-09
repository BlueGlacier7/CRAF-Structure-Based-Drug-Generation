import numpy as np
import os
import random
import lmdb
import pickle
import zlib
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph
from tqdm import tqdm
import re
from scripts.utils.transform import *
from scripts.utils.data import ComplexData, torchify_dict
from scripts.utils import LoadDataset

lmdb_path = './dataset/CrossDocked2020_training_set.lmdb'
output_dir = './graph_results'
os.makedirs(output_dir, exist_ok=True)
batch_size = 1000

## Data Augmentation
def drop_nodes(data, ratio):
    node_num = data.pos.size(0)
    drop_num = int(node_num * ratio)
    if drop_num == 0 or node_num - drop_num < 2:
        return data
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_keep = np.setdiff1d(np.arange(node_num), idx_drop)
    old2new = -np.ones(node_num, dtype=int)
    old2new[idx_keep] = np.arange(len(idx_keep))
    mask = np.isin(data.edge_index[0].cpu().numpy(), idx_keep) & np.isin(data.edge_index[1].cpu().numpy(), idx_keep)
    edge_index = data.edge_index[:, mask]
    edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None
    edge_index = torch.tensor(old2new[edge_index.cpu().numpy()], dtype=torch.long)
    new_data = Data()
    for key in data.keys:
        item = data[key]
        if key in ['x', 'pos']:
            new_data[key] = item[idx_keep]
        elif key == 'edge_index':
            new_data[key] = edge_index
        elif key == 'edge_attr' and edge_attr is not None:
            new_data[key] = edge_attr
        elif key in ['idx_ligand_ctx_in_cpx', 'idx_protein_in_cpx']:
            mask_idx = np.isin(item.cpu().numpy(), idx_keep)
            new_idx = old2new[item.cpu().numpy()[mask_idx]]
            new_data[key] = torch.tensor(new_idx, dtype=torch.long)
        else:
            if key in data:
                new_data[key] = item
    return new_data

def permute_edges(data, ratio):
    edge_num = data.edge_index.size(1)
    drop_num = int(edge_num * ratio)
    if drop_num == 0 or edge_num - drop_num < 1:
        return data
    idx_drop = np.random.choice(edge_num, drop_num, replace=False)
    idx_keep = np.setdiff1d(np.arange(edge_num), idx_drop)
    edge_index = data.edge_index[:, idx_keep]
    edge_attr = data.edge_attr[idx_keep] if data.edge_attr is not None else None
    new_data = Data()
    for key in data.keys:
        item = data[key]
        if key == 'edge_index':
            new_data[key] = edge_index
        elif key == 'edge_attr' and edge_attr is not None:
            new_data[key] = edge_attr
        else:
            new_data[key] = item
    return new_data

def subgraph(data, ratio):
    node_num = data.pos.size(0)
    sub_num = int(node_num * (1 - ratio))
    if sub_num < 2:
        return data
    subset = torch.randperm(node_num)[:sub_num]
    edge_index, edge_attr = pyg_subgraph(
        subset, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=node_num
    )
    new_data = Data()
    for key in data.keys:
        item = data[key]
        if key in ['x', 'pos']:
            new_data[key] = item[subset]
        elif key == 'edge_index':
            new_data[key] = edge_index
        elif key == 'edge_attr' and edge_attr is not None:
            new_data[key] = edge_attr
        elif key in ['idx_ligand_ctx_in_cpx', 'idx_protein_in_cpx']:
            mask_idx = np.isin(item.cpu().numpy(), subset.cpu().numpy())
            old2new = -np.ones(node_num, dtype=int)
            old2new[subset.cpu().numpy()] = np.arange(len(subset))
            new_idx = old2new[item.cpu().numpy()[mask_idx]]
            new_data[key] = torch.tensor(new_idx, dtype=torch.long)
        else:
            new_data[key] = item
    return new_data

augementation_methods = [drop_nodes, permute_edges, subgraph]

def filter_core_fields(data):
    keys_to_keep = [
        'x', 'pos', 'edge_index', 'edge_attr',
        'idx_ligand_ctx_in_cpx', 'idx_protein_in_cpx',
        'protein_atom_feature', 'ligand_atom_feature_full'
    ]
    for key in list(data.keys):
        if key not in keys_to_keep:
            del data[key]
    return data

def compress_dtype(data):
    # x, edge_attr, protein_atom_feature, ligand_atom_feature_full: int64 -> int16
    for key in ['x', 'edge_attr', 'protein_atom_feature', 'ligand_atom_feature_full']:
        if hasattr(data, key) and getattr(data, key) is not None:
            setattr(data, key, getattr(data, key).to(torch.int16))
    # edge_index, idx_ligand_ctx_in_cpx, idx_protein_in_cpx: int64 -> int32
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        data.edge_index = data.edge_index.to(torch.int32)
    for key in ['idx_ligand_ctx_in_cpx', 'idx_protein_in_cpx']:
        if hasattr(data, key) and getattr(data, key) is not None:
            setattr(data, key, getattr(data, key).to(torch.int32))
    return data

def pyg_graph_from_complexdata(data, k=16, edge_type_pro_lig=0):
    ligand_feat = data['ligand_atom_feature_full']  # [N, 15]
    protein_feat = data['protein_atom_feature']     # [M, 27]
    ligand_feat = torch.cat([ligand_feat, torch.zeros((ligand_feat.shape[0], protein_feat.shape[1] - ligand_feat.shape[1]), dtype=ligand_feat.dtype)], dim=1)
    x = torch.cat([ligand_feat, protein_feat], dim=0)  # [N_ligand+N_protein, D]
    pos = torch.cat([data['ligand_pos'], data['protein_pos']], dim=0)  # [N_ligand+N_protein, 3]
    n_lig = data['ligand_pos'].shape[0]
    n_pro = data['protein_pos'].shape[0]
    idx_ligand_ctx_in_cpx = torch.arange(n_lig, dtype=torch.long)
    idx_protein_in_cpx = torch.arange(n_pro, dtype=torch.long) + n_lig
    ligand_edge_index = data['ligand_bond_index'].clone()
    ligand_edge_attr = data['ligand_bond_type'].clone()
    protein_edge_index = data['protein_bond_index'].clone() + n_lig
    protein_edge_attr = data['protein_bond_type'].clone()
    edge_index_chem = torch.cat([ligand_edge_index, protein_edge_index], dim=1)
    edge_attr_chem = torch.cat([ligand_edge_attr, protein_edge_attr], dim=0)

    ## edge between protein and ligand
    edge_index_knn = knn_graph(pos, k=k, flow='target_to_source')
    chem_edges_set = set([(i.item(), j.item()) for i, j in edge_index_chem.t()])
    knn_edges = [(i.item(), j.item()) for i, j in edge_index_knn.t()]
    new_edges = []
    for idx, (i, j) in enumerate(knn_edges):
        if (i, j) not in chem_edges_set and (j, i) not in chem_edges_set:
            if (i < n_lig and j >= n_lig) or (i >= n_lig and j < n_lig):
                new_edges.append((i, j))
    if new_edges:
        edge_index_pro_lig = torch.tensor(new_edges, dtype=torch.long).t()
        edge_attr_pro_lig = torch.full((edge_index_pro_lig.shape[1],), edge_type_pro_lig, dtype=torch.long)
        edge_index = torch.cat([edge_index_chem, edge_index_pro_lig], dim=1)
        edge_attr = torch.cat([edge_attr_chem, edge_attr_pro_lig], dim=0)
    else:
        edge_index = edge_index_chem
        edge_attr = edge_attr_chem
    num_bond_types = 4
    edge_attr = F.one_hot(edge_attr, num_classes=num_bond_types).float()

    g = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        idx_ligand_ctx_in_cpx=idx_ligand_ctx_in_cpx,
        idx_protein_in_cpx=idx_protein_in_cpx,
        protein_atom_feature=data['protein_atom_feature'],
        ligand_atom_feature_full=data['ligand_atom_feature_full'],
    )
    return g

def batch_write_compressed(env, graphs, start_idx=0):
    with env.begin(write=True) as txn:
        for i, g in enumerate(graphs):
            key = str(start_idx + i).encode()
            value = zlib.compress(pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(key, value)

def parse_filename(filename):
    match = re.search(r'([^/]+)/[^/]+_rec_([^_]+)_', filename)
    if match:
        pro_type = match.group(1)
        ligand_pdbid = match.group(2)
        return pro_type, ligand_pdbid
    return None, None

def extract_all_ids_from_lmdb(lmdb_path, parse_filename_func):
    all_ids = []
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data = pickle.loads(value)
            protein_filename = getattr(data, 'protein_filename', None)
            ligand_filename = getattr(data, 'ligand_filename', None)
            pro_type, ligand_pdbid = parse_filename_func(protein_filename)
            all_ids.append({
                "key": key.decode('utf-8') if isinstance(key, bytes) else str(key),
                "protein_filename": protein_filename,
                "ligand_filename": ligand_filename,
                "pro_type": pro_type,
                "ligand_pdbid": ligand_pdbid
            })
    env.close()
    return all_ids


protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
])

dataset = LoadDataset(lmdb_path, transform=transform)
print(f"[main] Dataset loaded, total samples: {len(dataset)}")

lmdb_raw_path = os.path.join(output_dir, 'raw_graphs.lmdb')
lmdb_aug1_path = os.path.join(output_dir, 'aug_graphs_1.lmdb')
lmdb_aug2_path = os.path.join(output_dir, 'aug_graphs_2.lmdb')

raw_graphs, aug_graphs_1, aug_graphs_2 = [], [], []
batch_idx = 0

env_raw = lmdb.open(lmdb_raw_path, map_size=100*(1024**3), subdir=False, lock=False)
env_aug1 = lmdb.open(lmdb_aug1_path, map_size=100*(1024**3), subdir=False, lock=False)
env_aug2 = lmdb.open(lmdb_aug2_path, map_size=100*(1024**3), subdir=False, lock=False)

for idx, data in enumerate(tqdm(dataset)):
    pyg_graph = pyg_graph_from_complexdata(data)
    pyg_graph = filter_core_fields(pyg_graph)
    methods = random.sample(augementation_methods, 2)
    aug1 = filter_core_fields(methods[0](pyg_graph, 0.2))
    aug2 = filter_core_fields(methods[1](pyg_graph, 0.2))
    #pyg_graph = compress_dtype(pyg_graph)
    #aug1 = compress_dtype(aug1)
    #aug2 = compress_dtype(aug2)
    raw_graphs.append(pyg_graph)
    aug_graphs_1.append(aug1)
    aug_graphs_2.append(aug2)
    if (idx + 1) % batch_size == 0:
        batch_write_compressed(env_raw, raw_graphs, batch_idx * batch_size)
        batch_write_compressed(env_aug1, aug_graphs_1, batch_idx * batch_size)
        batch_write_compressed(env_aug2, aug_graphs_2, batch_idx * batch_size)
        raw_graphs, aug_graphs_1, aug_graphs_2 = [], [], []
        batch_idx += 1

if raw_graphs:
    batch_write_compressed(env_raw, raw_graphs, batch_idx * batch_size)
    batch_write_compressed(env_aug1, aug_graphs_1, batch_idx * batch_size)
    batch_write_compressed(env_aug2, aug_graphs_2, batch_idx * batch_size)

all_ids = extract_all_ids_from_lmdb(lmdb_path, parse_filename)
np.save(os.path.join(output_dir, 'all_ids.npy'), all_ids)
print(f"Saved all_ids with filenames to {os.path.join(output_dir, 'all_ids.npy')}")

env_raw.close()
env_aug1.close()
env_aug2.close()
