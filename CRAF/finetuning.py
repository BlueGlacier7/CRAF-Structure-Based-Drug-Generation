import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from scripts.model import CRAF_main, reset_parameters, freeze_parameters
from scripts.model.craf import config as default_config
from scripts.utils import Experiment, LoadDataset
from scripts.utils.transform import *
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")


protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
focal_masker = FocalMaker(r=4, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
atom_composer = AtomComposer(
    knn=16, num_workers=16, graph_type='knn', radius=10, use_protein_bond=True
)
combine = Combine(traj_fn, focal_masker, atom_composer)
transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    combine,
    collate_fn
])

dataset = LoadDataset('./dataset/CrossDocked2020_training_set.lmdb', transform=transform)
print('Num data:', len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=1000, shuffle=True, random_seed=0)
print("train_set:",len(train_set))
print("valid_set:",len(valid_set))

device = 'cuda:0'
cl_pretrain_ckpt = './path/to/contrastive_pretraining/ckpt'
ligand_pretrain_ckpt = './path/to/ligand_pretraining/ckpt'

cl_ckpt_exists = cl_pretrain_ckpt is not None and os.path.exists(cl_pretrain_ckpt)
ligand_ckpt_exists = ligand_pretrain_ckpt is not None and os.path.exists(ligand_pretrain_ckpt)

if ligand_ckpt_exists and cl_ckpt_exists:
    ligand_ckpt = torch.load(ligand_pretrain_ckpt, map_location=device)
    config = ligand_ckpt['config']
    model = CRAF_main(config).to(device)
    model.load_state_dict(ligand_ckpt['model'])
    cl_ckpt = torch.load(cl_pretrain_ckpt, map_location=device)
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in cl_ckpt['model'].items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(encoder_state_dict)
    print("Loaded ligand pretrain + contrastive encoder.")
else:
    raise RuntimeError("Unexpected pretrain ckpt logic.")

print(model.get_parameter_number())
keys = ['edge_flow.flow_layers.5', 'atom_flow.flow_layers.5', 
        'pos_predictor.mu_net', 'pos_predictor.logsigma_net', 'pos_predictor.pi_net',
        'focal_net.net.1']
model = reset_parameters(model, keys)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2.e-4, weight_decay=0, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-5)

exp = Experiment(
    model, train_set, optimizer, valid_set=valid_set, scheduler=scheduler,
    device=device, use_amp=False
)
exp.fit_step(
    400000, valid_per_step=1000, train_batch_size=4, valid_batch_size=4, print_log=True,
    with_tb=True, logdir='./finetuning_log', schedule_key='loss', num_workers=4, 
    pin_memory=False, follow_batch=[], exclude_keys=[], collate_fn=None, 
    max_edge_num_in_batch=1000000
)
