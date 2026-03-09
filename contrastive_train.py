import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import numpy as np
import pandas as pd
import os
import datetime
from torch_geometric.nn import GlobalAttention
from scripts.model.craf import CRAF_main, config as craf_config
from scripts.model.net_utils import embed_compose
from scripts.utils.cl_train_loader import get_data_loaders, get_args
from scripts.utils.cl_logger import Logger


class ScalarVectorAttentionReadout(nn.Module):
    def __init__(self, scalar_dim, vector_dim, out_dim=128):
        super().__init__()
        self.input_dim = scalar_dim + vector_dim * 3
        self.gate_nn = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.01),  
            nn.Linear(128, 1)
        )
        self.attn_pool = GlobalAttention(self.gate_nn)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, out_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, h_cpx, batch):
        device = next(self.parameters()).device
        # h_cpx[0]: [N, scalar_dim], h_cpx[1]: [N, vector_dim, 3]
        scalar = h_cpx[0].to(device)  # [N, scalar_dim]
        vector = h_cpx[1].reshape(h_cpx[1].shape[0], -1).to(device)  # [N, vector_dim*3]
        node_emb = torch.cat([scalar, vector], dim=-1)  # [N, input_dim]
        batch = batch.to(device)
        graph_emb = self.attn_pool(node_emb, batch)
        return self.mlp(graph_emb)                      # [B, out_dim]

class ContrastiveTraining(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.pf = CRAF_main(config).to(device)
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.device = device
        self.readout = ScalarVectorAttentionReadout(
            scalar_dim=self.emb_dim[0],
            vector_dim=self.emb_dim[1],
            out_dim=128  
        ).to(device)

    def forward(self, data, return_node_emb=False):
        if hasattr(data, 'idx_ligand_ctx_in_cpx'):
            data.idx_ligand_ctx_in_cpx = data.idx_ligand_ctx_in_cpx.long()
        if hasattr(data, 'idx_protein_in_cpx'):
            data.idx_protein_in_cpx = data.idx_protein_in_cpx.long()
        if hasattr(data, 'edge_index'):
            data.edge_index = data.edge_index.long()
        h_cpx = embed_compose(
            data.x.float(), data.pos, 
            data.idx_ligand_ctx_in_cpx, data.idx_protein_in_cpx,
            self.pf.ligand_atom_emb, self.pf.protein_atom_emb, self.emb_dim
        )
        h_cpx = self.pf.encoder(
            node_attr=h_cpx,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_feature=data.edge_attr,
            annealing=1.0
        )
        if return_node_emb:
            return h_cpx  # h_cpx[0]: [N, 64], h_cpx[1]: [N, 16, 3]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(h_cpx[0].shape[0], dtype=torch.long, device=h_cpx[0].device)
        graph_emb = self.readout(h_cpx, batch)

        return graph_emb
    
class NTXentLoss(nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.use_cosine_similarity = use_cosine_similarity
        if use_cosine_similarity:
            self.similarity_function = self._cosine_similarity
        else:
            self.similarity_function = self._dot_similarity

    def _dot_similarity(self, x, y):
        return torch.matmul(x, y.T)

    def _cosine_similarity(self, x, y):
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        return torch.matmul(x, y.T)

    def _get_fake_neg_mask(self, pro_types, ligand_pdbids):
        batch_size = len(pro_types)
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and pro_types[i] == pro_types[j] and ligand_pdbids[i] == ligand_pdbids[j]:
                    mask[i, j] = True
                    mask[i, j + batch_size] = True
                    mask[i + batch_size, j] = True
                    mask[i + batch_size, j + batch_size] = True
        return mask

    def forward(self, z1, z2, pro_types, ligand_pdbids):
        """
        z1, z2: [batch, dim]
        """
        batch_size = z1.size(0)
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)  # [2*batch, dim]
        similarity_matrix = self.similarity_function(representations, representations)  # [2*batch, 2*batch]

        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        fake_neg_mask = self._get_fake_neg_mask(pro_types, ligand_pdbids)
        similarity_matrix = similarity_matrix.masked_fill(fake_neg_mask, -float('inf'))

        positives = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)  # [batch]
        negatives = torch.exp(similarity_matrix / self.temperature).sum(dim=-1)  # [2*batch]

        loss = -torch.log(positives / (negatives[:batch_size] + 1e-8)).mean()
        return loss
    
def to_device(data, device):  
    if isinstance(data, (list, tuple)):
        return [d.to(device) for d in data]
    else:
        return data.to(device)
    
def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    all_ids = np.load(args.all_ids, allow_pickle=True)
    train_loader, val_loader = get_data_loaders(
        args.aug1, args.aug2, batch_size=args.batch_size, val_ratio=args.val_ratio
    )
    sample = train_loader.dataset[0][1]
    craf_config.num_bond_types = 4
    craf_config.num_atom_type = 9
    craf_config.protein_atom_feature_dim = sample.protein_atom_feature.shape[1]
    craf_config.ligand_atom_feature_dim = sample.ligand_atom_feature_full.shape[1]
    craf_config.msg_annealing = True
    model = ContrastiveTraining(craf_config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-6)
    ntxent_criterion = NTXentLoss(device=device, temperature=args.temperature, use_cosine_similarity=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(args.save_path, now)
    os.makedirs(output_dir, exist_ok=True)

    logger = Logger(output_dir)
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0
        grad_norms = []
        pos_sims = []
        neg_sims = []
        num_batches = 0

        for batch in train_loader:
            idx, aug1, aug2 = batch
            aug1 = to_device(aug1, device)
            aug2 = to_device(aug2, device)
            emb1 = model(aug1)
            emb2 = model(aug2)
            if emb1.dim() == 1: emb1 = emb1.unsqueeze(0)
            if emb2.dim() == 1: emb2 = emb2.unsqueeze(0)
            pro_types = [all_ids[i]['pro_type'] for i in idx]
            ligand_pdbids = [all_ids[i]['ligand_pdbid'] for i in idx]
            loss = ntxent_criterion(emb1, emb2, pro_types, ligand_pdbids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            with torch.no_grad():
                emb1_norm = nn.functional.normalize(emb1, dim=1)
                emb2_norm = nn.functional.normalize(emb2, dim=1)
                pos_sim = (emb1_norm * emb2_norm).sum(dim=-1)
                pos_sim_mean = pos_sim.mean().item()
                pos_sims.append(pos_sim_mean)
                representations = torch.cat([emb1_norm, emb2_norm], dim=0)
                sim_matrix = torch.matmul(representations, representations.T)
                batch_size = emb1.size(0)
                mask = torch.eye(batch_size * 2, dtype=torch.bool, device=emb1.device)
                sim_matrix = sim_matrix.masked_fill(mask, float('nan'))
                fake_neg_mask = ntxent_criterion._get_fake_neg_mask(pro_types, ligand_pdbids)
                sim_matrix = sim_matrix.masked_fill(fake_neg_mask, float('nan'))
                neg_sim_mean = torch.nanmean(sim_matrix).item()
                neg_sims.append(neg_sim_mean)

        mean_train_loss = train_loss_sum / num_batches
        mean_grad_norm = np.mean(grad_norms)
        mean_pos_sim = np.mean(pos_sims)
        mean_neg_sim = np.mean(neg_sims)
        lr = optimizer.param_groups[0]['lr']

        logger.log('grad_norm', mean_grad_norm, epoch)
        logger.log('pos_sim', mean_pos_sim, epoch)
        logger.log('neg_sim', mean_neg_sim, epoch)
        logger.log('train_loss', mean_train_loss, epoch)
        logger.log('lr', lr, epoch)

        model.eval()
        val_loss_sum = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                idx, aug1, aug2 = batch
                aug1 = to_device(aug1, device)
                aug2 = to_device(aug2, device)
                emb1 = model(aug1)
                emb2 = model(aug2)
                if emb1.dim() == 1: emb1 = emb1.unsqueeze(0)
                if emb2.dim() == 1: emb2 = emb2.unsqueeze(0)
                pro_types = [all_ids[i]['pro_type'] for i in idx]
                ligand_pdbids = [all_ids[i]['ligand_pdbid'] for i in idx]
                loss = ntxent_criterion(emb1, emb2, pro_types, ligand_pdbids)
                val_loss_sum += loss.item()
                val_batches += 1
        mean_val_loss = val_loss_sum / val_batches
        logger.log('val_loss', mean_val_loss, epoch)

        print(f"Epoch {epoch}, train_loss: {mean_train_loss:.4f}, val_loss: {mean_val_loss:.4f}, "
              f"pos_sim: {mean_pos_sim:.4f}, neg_sim: {mean_neg_sim:.4f}, lr: {lr:.6f}")

        history.append({
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
            'grad_norm': mean_grad_norm,
            'pos_sim': mean_pos_sim,
            'neg_sim': mean_neg_sim,
            'lr': lr
        })

        scheduler.step(mean_val_loss)

        if epoch % 10 == 0:
            torch.save({
                'config': craf_config,
                'model': model.pf.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': epoch,
            }, os.path.join(output_dir, f'epoch_{epoch}.pt'))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_path = os.path.join(output_dir, f'epoch_{epoch}.pt')
            torch.save({
                'config': craf_config,
                'model': model.pf.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': epoch,
            }, best_model_path)
            print(f"Best model updated at epoch {epoch}, val loss: {mean_val_loss:.4f}")
        torch.cuda.empty_cache()

    torch.save({
        'config': craf_config,
        'model': model.pf.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'iteration': epoch,
    }, os.path.join(output_dir, f'epoch_{args.epochs}.pt'))
    print(f"Model saved to {os.path.join(output_dir, f'epoch_{args.epochs}.pt')}")
    logger.close()

    data_df = pd.DataFrame(history)
    data_csv_path = os.path.join(output_dir, 'data_history.csv')
    data_df.to_csv(data_csv_path, index=False)
    print(f"Training history saved to {data_csv_path}")

if __name__ == '__main__':
    main()
