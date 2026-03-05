# scripts/train.py

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from x_transformers import TransformerWrapper, Decoder, Encoder

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd

from utils_model import get_model
from tqdm import tqdm

import datetime
import wandb

import pdb

VERBOSE = 0
N_VALS_TO_LOG = 10

# Import your dataset utilities
from utils_data import get_loaders 

@hydra.main(version_base=None, config_path=os.path.abspath("config"), config_name="default")
def main(cfg: DictConfig):

    if VERBOSE:
      print(OmegaConf.to_yaml(cfg))

    # Set seeds
    torch.manual_seed(cfg.model_seed)
    torch.cuda.manual_seed(cfg.model_seed)
    torch.cuda.manual_seed_all(cfg.model_seed)  # if using multi-GPU
    random.seed(cfg.model_seed)
    np.random.seed(cfg.data.seed)

    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


    ##########################################################################
    # Get the loaders & model
    ##########################################################################
    loader_train, loader_val = get_loaders(cfg.data)
    cfg.data.n_samples_train = len(loader_train.dataset)
    cfg.data.n_samples_val = len(loader_val.dataset)

    if cfg.model.vocab_size == -1:
      num_tokens = loader_train.dataset.n_states + 1 # +1 for the ignore
      cfg.model.vocab_size = num_tokens
    model = get_model(cfg.model)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # create a wandb run
    wandb.init(project=cfg.wandb.project,
               config=OmegaConf.to_container(cfg, resolve=True),
               name=f"{cfg.wandb.name}_{timestamp}",
               entity=cfg.wandb.entity)
    
    
    os.makedirs(f"checkpoints/{cfg.wandb.name}_{timestamp}", exist_ok=True)

    ##########################################################################
    # Define training loop
    ##########################################################################
    def train(model, loader, loaders_val, lr=1e-4, weight_decay=0, epochs=1, lr_schedule='cosine',
             n_steps_to_eval=100, n_worst_samples=0, scheduler_eta_min=0.0):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(loader) * epochs
        if lr_schedule == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=scheduler_eta_min
            )
        else:
            scheduler = None
        model.train()

        step_count = 0
        train_losses = []
        val_losses, val_accs = [], []

        best_val_loss = float('inf')

        for epoch_i in range(epochs):
            for batch in tqdm(loader, desc=f"Epoch {epoch_i+1}/{epochs}"):
                x, y, ignore_percentage = batch[0].cuda().long(), batch[1].cuda().long(), batch[2].cuda().float()
                logits = model(x)
                if cfg.training.weight_by_ignores:
                    loss = nn.CrossEntropyLoss(reduction='none')(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = (loss.reshape(logits.size(0), -1) * ignore_percentage.reshape(-1, 1)).mean()
                else:
                    loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_losses.append(loss.item())
                step_count += 1
                curr_lr = optimizer.param_groups[0]['lr']

                # Evaluate occasionally
                if step_count % n_steps_to_eval == 0 or step_count == 1:
                    train_batch_acc = (torch.argmax(logits, dim=-1) == y).float().mean().item()
                    avg_train_loss = np.mean(train_losses[-10:])
                    
                    # Evaluate on val set
                    model.eval()
                    val_loss, val_acc, worst_samples = eval_model(model, loader_val, n_worst_samples=n_worst_samples)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    model.train()
                    # Log to wandb
                    wandb.log({
                        'train_loss': avg_train_loss,
                        'train_batch_acc': train_batch_acc,
                        'val_loss': val_losses[-1],
                        'val_acc': val_accs[-1],
                        'curr_lr': curr_lr,
                    }, step=step_count)

                    # log model weight norms
                    weight_norms = {}
                    for name, param in model.named_parameters():
                        l1_norm = torch.norm(param, p=1).item()
                        l2_norm = torch.norm(param, p=2).item()
                        lfro_norm = torch.norm(param, p='fro').item()
                        linf_norm = torch.norm(param, p=float('inf')).item()
                        weight_norms[name + '_l1'] = l1_norm
                        weight_norms[name + '_l2'] = l2_norm
                        weight_norms[name + '_fro'] = lfro_norm
                        weight_norms[name + '_inf'] = linf_norm
                    wandb.log(weight_norms, step=step_count)

                    # Print to console
                    print(f"Step {step_count}: train_loss={avg_train_loss:.4f} / train_acc: {train_batch_acc:.4f}")
                    print(f"val_loss: {val_loss:.4f} / val_acc: {val_acc:.4f}")
                    
                    # save the model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), f"checkpoints/{cfg.wandb.name}_{timestamp}/best.pth")
                    torch.save(model.state_dict(), f"checkpoints/{cfg.wandb.name}_{timestamp}/last.pth")

                    # Check for early stopping
                    if len(val_losses) > 10 and np.mean(val_losses[-10:]) < 1e-8:
                        print("Early stopping at loss < 1e-8")
                        return train_losses, val_losses, val_accs
  
                if step_count % cfg.training.n_steps_to_save == 0 or step_count == 1:
                    torch.save(model.state_dict(), f"checkpoints/{cfg.wandb.name}_{timestamp}/step{step_count}.pth")

            if worst_samples is not None and step_count > 1: # i.e.don't update the train set on the first step
                # update the train set at the end of each epoch
                loader.dataset.X = torch.cat([loader.dataset.X, worst_samples['X']])
                loader.dataset.y = torch.cat([loader.dataset.y, worst_samples['y']])

        # save the model
        torch.save(model.state_dict(), f"checkpoints/{cfg.wandb.name}_{timestamp}/last.pth")

        return train_losses, val_losses, val_accs


    def eval_model(model, loader, n_samples=-1, n_worst_samples=0):
        model.eval()
        val_losses, val_accs = [], []
        samples_count = 0
        all_samples_X = []
        all_samples_y = []
        all_seq_accs = []
        all_seq_worst_pred = []
        for batch in loader:
            xv, yv, _ = batch[0].cuda().long(), batch[1].cuda().long(), batch[2]
            logits_v = model(xv)

            # eval loss & acc
            loss_v = nn.CrossEntropyLoss(reduction='none')(logits_v.reshape(-1, logits_v.size(-1)), yv.reshape(-1))
            val_losses.append(loss_v.mean().item())
            acc_v_by_seq = (torch.argmax(logits_v, dim=-1) == yv).float().mean(-1)
            val_accs.append(acc_v_by_seq.mean().item())

            if n_worst_samples > 0:
                all_samples_X.append(xv)
                all_samples_y.append(yv)
                all_seq_accs.append(acc_v_by_seq)
                loss_v = loss_v.reshape(xv.size(0), -1)
                max_loss_per_seq = torch.max(loss_v, dim=-1)[0]
                all_seq_worst_pred.append(max_loss_per_seq)

            if n_samples > 0 and samples_count >= n_samples:
                break
        
        if n_worst_samples > 0:
            # select the worse samples according to per-seq accuracy or per-seq max loss
            all_samples_X = torch.cat(all_samples_X)
            all_samples_y = torch.cat(all_samples_y)
            all_seq_accs = torch.cat(all_seq_accs)
            all_seq_worst_pred = torch.cat(all_seq_worst_pred)

            n_worst_samples = min(n_worst_samples, all_samples_X.size(0))
            worst_idx_by_acc = torch.argsort(all_seq_accs)[:n_worst_samples//2]
            worst_idx_by_loss = torch.argsort(all_seq_worst_pred)[-n_worst_samples//2:]
            # remove duplicates
            worst_idx = torch.cat([worst_idx_by_acc, worst_idx_by_loss])
            worst_idx = torch.unique(worst_idx)
            worst_samples_X = all_samples_X[worst_idx]
            worst_samples_y = all_samples_y[worst_idx]
            worst_samples = {'X': worst_samples_X, 'y': worst_samples_y}
        else:
            worst_samples = None
            
        return np.mean(val_losses), np.mean(val_accs), worst_samples

    ##########################################################################
    # Train the model
    ##########################################################################
    print("=== Training joint mixture model ===")
    train_losses, val_losses, val_accs = train(
        model, 
        loader_train,
        loader_val,
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs, 
        n_steps_to_eval=cfg.training.n_steps_to_eval,
        lr_schedule=cfg.training.lr_schedule,
        scheduler_eta_min=cfg.training.get('scheduler_eta_min', 0.0),
    )

    print("=== Training finished ===")



if __name__ == "__main__":
    main()