import os
import pickle
import numpy as np
from tqdm import tqdm

import jax
import optax
from jax import numpy as jnp
from jax import random as jr

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd

import datetime
import wandb

import math

from src.transformer import Transformer
from src.opt import Adam, SGD
from src.utils_data import (
    get_loaders,
    FlipFlopAutomaton,
    SymmetricAutomaton,
    AlternatingAutomaton,
)

VERBOSE = 0
N_VALS_TO_LOG = 10


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="default")
def main(cfg: DictConfig):

    if VERBOSE:
        print(OmegaConf.to_yaml(cfg))

    # Set seeds
    key = jr.PRNGKey(cfg.model_seed)
    np.random.seed(cfg.data.seed)

    ##########################################################################
    # Get the loaders & model
    ##########################################################################
    train_loader_fn, val_loader_fn = get_loaders(cfg.data)

    # Recreate datasets temporarily to determine n_samples and vocab size.
    if cfg.data.task == "flipflop":
        temp_train_set = FlipFlopAutomaton(
            cfg.data.n_states,
            cfg.data.length,
            cfg.data.random_length,
            cfg.data.seed,
            cfg.data.n_ignores_train,
            cfg.data.p_ignores_train,
            cfg.data.n_samples_train,
            cfg.data.fdata_train,
        )
        temp_val_set = FlipFlopAutomaton(
            cfg.data.n_states,
            cfg.data.length,
            cfg.data.random_length,
            cfg.data.seed,
            cfg.data.n_ignores_val,
            cfg.data.p_ignores_val,
            cfg.data.n_samples_val,
            cfg.data.fdata_val,
        )
    elif cfg.data.task == "symmetric":
        temp_train_set = SymmetricAutomaton(
            cfg.data.n_states,
            cfg.data.length,
            cfg.data.random_length,
            cfg.data.seed,
            cfg.data.label_type,
            cfg.data.n_actions,
            cfg.data.n_samples_train,
            cfg.data.fdata_train,
        )
        temp_val_set = SymmetricAutomaton(
            cfg.data.n_states,
            cfg.data.length,
            cfg.data.random_length,
            cfg.data.seed,
            cfg.data.label_type,
            cfg.data.n_actions,
            cfg.data.n_samples_val,
            cfg.data.fdata_val,
        )
    elif cfg.data.task == "alternating":
        temp_train_set = AlternatingAutomaton(
            cfg.data.n_states,
            cfg.data.length,
            cfg.data.random_length,
            cfg.data.seed,
            cfg.data.label_type,
            cfg.data.n_samples_train,
            cfg.data.fdata_train,
        )
        temp_val_set = AlternatingAutomaton(
            cfg.data.n_states,
            cfg.data.length,
            cfg.data.random_length,
            cfg.data.seed,
            cfg.data.label_type,
            cfg.data.n_samples_val,
            cfg.data.fdata_val,
        )
    else:
        raise ValueError(f"Task {cfg.data.task} not supported")
    
    cfg.data.n_samples_train = len(temp_train_set)
    cfg.data.n_samples_val = len(temp_val_set)

    # Determine vocab size
    if cfg.model.V == -1:
        if cfg.data.task == "flipflop":
            num_tokens = temp_train_set.n_states + 1  # +1 for the ignore token
        elif cfg.data.task in ("symmetric", "alternating"):
            # Keep parity with torch training script: vocab_size = n!
            num_tokens = int(math.factorial(int(temp_train_set.n_states)))
        else:
            raise ValueError(f"Task {cfg.data.task} not supported")
        cfg.model.V = int(num_tokens)
    
    # Initialize model
    model = Transformer(
        D=cfg.model.D,
        L=cfg.model.L,
        M=cfg.model.M,
        H=cfg.model.H,
        K=cfg.model.K,
        V=cfg.model.V,
        fsdp=cfg.model.get('fsdp', False),
        scan_unroll=cfg.model.get('scan_unroll', True),
        dtype=cfg.model.get('dtype', 'bfloat16'),
        norm_dtype=cfg.model.get('norm_dtype', None),
        norm_eps=cfg.model.get('norm_eps', 1e-6),
        rope_dtype=cfg.model.get('rope_dtype', None),
        rope_freq=cfg.model.get('rope_freq', 10000),
        flash_attention=cfg.model.get('flash_attention', True),
        grad_checkpoint=cfg.model.get('grad_checkpoint', False),
    )
    
    key, init_key = jr.split(key)
    params = model.init(init_key)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # create a wandb run
    wandb.init(project=cfg.wandb.project,
               config=OmegaConf.to_container(cfg, resolve=True),
               name=f"{cfg.wandb.name}_{timestamp}",
               entity=cfg.wandb.entity)
    
    os.makedirs(f"checkpoints/{cfg.wandb.name}_{timestamp}", exist_ok=True)

    def save_params(params, path):
        """Save JAX parameters to pickle file."""
        # Convert JAX arrays to numpy for serialization
        params_numpy = jax.tree.map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, params)
        with open(path, 'wb') as f:
            pickle.dump(params_numpy, f)

    def load_params(path):
        """Load JAX parameters from pickle file."""
        with open(path, 'rb') as f:
            params_numpy = pickle.load(f)
        # Convert numpy arrays back to JAX arrays
        params = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, params_numpy)
        return params

    # Load checkpoint if specified
    if cfg.model.load_ckpt and cfg.model.ckpt_path and os.path.exists(cfg.model.ckpt_path):
        print(f"Loading checkpoint from {cfg.model.ckpt_path}")
        params = load_params(cfg.model.ckpt_path)

    ##########################################################################
    # Define training loop
    ##########################################################################
    def train(model, params, train_loader_fn, val_loader_fn, lr=1e-4, weight_decay=0, 
              epochs=1, lr_schedule='cosine', n_steps_to_eval=100, n_worst_samples=0, 
              scheduler_eta_min=0.0, weight_by_ignores=False):
        
        # Setup optimizer
        if cfg.training.get('optimizer', 'adam') == 'adam':
            optimizer = Adam(lr=lr, b1=0.9, b2=0.95, eps=1e-8)
        else:
            optimizer = SGD(lr=lr, momentum=0.9)
        
        base_optimizer = optimizer.build(model.lrs)
        
        # Add weight decay if specified
        if weight_decay > 0:
            base_optimizer = optax.chain(
                optax.add_decayed_weights(weight_decay),
                base_optimizer
            )
        
        # Setup learning rate schedule
        total_steps = cfg.data.n_samples_train // cfg.data.batch_size * epochs
        if lr_schedule == 'cosine':
            # Schedule as multiplier: 1.0 -> eta_min/lr so effective_lr = lr * schedule(step)
            schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps,
                alpha=scheduler_eta_min / lr if lr > 0 else 0.0
            )
            # Chain the schedule with the optimizer (schedule should come first)
            opt_update = optax.chain(
                optax.scale_by_schedule(schedule),
                base_optimizer
            )
        else:
            opt_update = base_optimizer
        
        opt_state = opt_update.init(params)

        # Define loss function
        def loss_fn(params, x, y, ignore_percentage=None):
            logits = model.apply(params, x)
            # logits: [batch_size, seq_len, vocab_size]
            # y: [batch_size, seq_len]
            logits_flat = logits.reshape(-1, logits.shape[-1])
            y_flat = y.reshape(-1)
            
            if weight_by_ignores and ignore_percentage is not None:
                # Weight loss by ignore percentage
                loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits_flat, y_flat)
                loss_per_seq = loss_per_token.reshape(logits.shape[0], -1)
                weights = ignore_percentage.reshape(-1, 1)
                loss = (loss_per_seq * weights).mean()
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, y_flat).mean()
            return loss

        # Single jitted step: loss + grad + optimizer update (one XLA call per step)
        def step_fn(params, opt_state, x, y, ignore_pct):
            ignore_pct = ignore_pct if weight_by_ignores else None
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y, ignore_pct)
            updates, new_opt_state = opt_update.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        step_fn = jax.jit(step_fn)
        # Jitted forward for eval and train_batch_acc (avoids recompilation from raw model.apply)
        apply_fn = jax.jit(model.apply)

        def eval_batch_loss_acc(params, xv, yv):
            """Jitted: forward + loss + acc for one batch (single device round-trip)."""
            logits_v = apply_fn(params, xv)
            logits_flat = logits_v.reshape(-1, logits_v.shape[-1])
            yv_flat = yv.reshape(-1)
            loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits_flat, yv_flat)
            loss_per_seq = loss_per_token.reshape(xv.shape[0], -1)
            batch_loss = loss_per_seq.mean()
            acc_v_by_seq = (jnp.argmax(logits_v, axis=-1) == yv).astype(jnp.float32).mean(axis=-1)
            batch_acc = acc_v_by_seq.mean()
            return batch_loss, batch_acc, logits_v, loss_per_seq, acc_v_by_seq

        eval_batch_jit = jax.jit(eval_batch_loss_acc)

        def eval_model(apply_fn, params, val_loader_fn, n_samples=-1, n_worst_samples=0):
            val_losses, val_accs = [], []
            samples_count = 0
            all_samples_X = []
            all_samples_y = []
            all_seq_accs = []
            all_seq_worst_pred = []
            val_loader = val_loader_fn()
            for batch in val_loader:
                xv = jnp.asarray(batch[0], dtype=jnp.int32)
                yv = jnp.asarray(batch[1], dtype=jnp.int32)
                batch_loss, batch_acc, logits_v, loss_per_seq, acc_v_by_seq = eval_batch_jit(params, xv, yv)
                val_losses.append(float(batch_loss))
                val_accs.append(float(batch_acc))
                if n_worst_samples > 0:
                    all_samples_X.append(xv)
                    all_samples_y.append(yv)
                    all_seq_accs.append(acc_v_by_seq)
                    all_seq_worst_pred.append(jnp.max(loss_per_seq, axis=-1))
                if n_samples > 0 and samples_count >= n_samples:
                    break
                samples_count += xv.shape[0]
            if n_worst_samples > 0:
                all_samples_X = jnp.concatenate(all_samples_X, axis=0)
                all_samples_y = jnp.concatenate(all_samples_y, axis=0)
                all_seq_accs = jnp.concatenate(all_seq_accs, axis=0)
                all_seq_worst_pred = jnp.concatenate(all_seq_worst_pred, axis=0)
                n_sel = min(n_worst_samples, all_samples_X.shape[0])
                worst_idx_by_acc = jnp.argsort(all_seq_accs)[:n_sel // 2]
                worst_idx_by_loss = jnp.argsort(all_seq_worst_pred)[-n_sel // 2:]
                worst_idx = jnp.unique(jnp.concatenate([worst_idx_by_acc, worst_idx_by_loss]))
                worst_samples = {'X': all_samples_X[worst_idx], 'y': all_samples_y[worst_idx]}
            else:
                worst_samples = None
            return np.mean(val_losses), np.mean(val_accs), worst_samples

        step_count = 0
        train_losses = []
        val_losses, val_accs = [], []

        best_val_loss = float('inf')
        worst_samples = None

        # Training key for shuffling
        train_key = jr.PRNGKey(42)

        for epoch_i in range(epochs):
            train_loader = train_loader_fn(key=train_key)
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch_i+1}/{epochs}"):
                x = jnp.asarray(batch[0], dtype=jnp.int32)
                y = jnp.asarray(batch[1], dtype=jnp.int32)
                if len(batch) == 3:
                    ignore_percentage = jnp.asarray(batch[2], dtype=jnp.float32)
                else:
                    ignore_percentage = None

                # Single jitted step (forward + backward + optimizer update)
                params, opt_state, loss = step_fn(params, opt_state, x, y, ignore_percentage)

                train_losses.append(float(loss))
                step_count += 1

                # Get current learning rate
                if lr_schedule == 'cosine':
                    curr_lr = float(schedule(step_count))
                else:
                    curr_lr = lr

                # Evaluate occasionally
                if step_count % n_steps_to_eval == 0 or step_count == 1:
                    # Compute training accuracy on current batch (jitted forward)
                    logits = apply_fn(params, x)
                    train_batch_acc = float((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32).mean())
                    avg_train_loss = np.mean(train_losses[-10:])
                    
                    # Evaluate on val set(s) (uses jitted apply inside)
                    if isinstance(val_loader_fn, (list, tuple)):
                        per_val_losses = []
                        per_val_accs = []
                        log_dict = {
                            'train_loss': avg_train_loss,
                            'train_batch_acc': train_batch_acc,
                            'curr_lr': curr_lr,
                        }

                        for val_loader in val_loader_fn:
                            val_loss_i, val_acc_i, worst_samples = eval_model(
                                apply_fn, params, val_loader, n_worst_samples=n_worst_samples
                            )
                            per_val_losses.append(val_loss_i)
                            per_val_accs.append(val_acc_i)

                            # Log separate metrics per validation loader
                            display_name = getattr(val_loader, "display", "val")
                            log_dict[f"val_loss_{display_name}"] = val_loss_i
                            log_dict[f"val_acc_{display_name}"] = val_acc_i

                        # Aggregate for tracking best/early stopping
                        val_loss = float(np.mean(per_val_losses))
                        val_acc = float(np.mean(per_val_accs))
                    else:
                        val_loss, val_acc, worst_samples = eval_model(
                            apply_fn, params, val_loader_fn, n_worst_samples=n_worst_samples
                        )
                        log_dict = {
                            'train_loss': avg_train_loss,
                            'train_batch_acc': train_batch_acc,
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'curr_lr': curr_lr,
                        }

                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    
                    # Log model weight norms (single device sync via device_get)
                    params_cpu = jax.device_get(params)
                    weight_norms = {}
                    def _path_elem_to_str(p):
                        """Format one path element for logging: no brackets, show indices/names clearly."""
                        if isinstance(p, (int, np.integer)):
                            return str(int(p))
                        if isinstance(p, str):
                            return p
                        if isinstance(p, (list, tuple)):
                            return '_'.join(_path_elem_to_str(x) for x in p)
                        s = str(p)
                        # Strip bracket notation e.g. "['head']" -> "head", "(0,)" -> "0"
                        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                            inner = s[1:-1].strip()
                            if not inner:
                                return s
                            if ',' in inner:
                                return '_'.join(part.strip().strip("'\"") for part in inner.split(','))
                            return inner.strip("'\"")
                        return s
                    def _log_norms(path, param):
                        name = '/'.join(_path_elem_to_str(p) for p in path)
                        p = np.asarray(param)
                        weight_norms[name + '_l1'] = float(np.abs(p).sum())
                        weight_norms[name + '_l2'] = float(np.linalg.norm(p))
                        weight_norms[name + '_fro'] = float(np.linalg.norm(p.ravel()))
                        weight_norms[name + '_inf'] = float(np.abs(p).max())
                    jax.tree.map_with_path(_log_norms, params_cpu)
                    log_dict.update(weight_norms)
                    
                    wandb.log(log_dict, step=step_count)

                    # Print to console
                    print(f"Step {step_count}: train_loss={avg_train_loss:.4f} / train_acc: {train_batch_acc:.4f}")
                    print(f"val_loss: {val_loss:.4f} / val_acc: {val_acc:.4f}")
                    
                    # save the model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_params(params, f"checkpoints/{cfg.wandb.name}_{timestamp}/best.pkl")
                    save_params(params, f"checkpoints/{cfg.wandb.name}_{timestamp}/last.pkl")

                    # Check for early stopping
                    if len(val_losses) > 10 and np.mean(val_losses[-10:]) < 1e-8:
                        print("Early stopping at loss < 1e-8")
                        return train_losses, val_losses, val_accs
  
                if step_count % cfg.training.n_steps_to_save == 0 or step_count == 1:
                    save_params(params, f"checkpoints/{cfg.wandb.name}_{timestamp}/step{step_count}.pkl")

            # Update training dataset with worst samples if needed
            if worst_samples is not None and step_count > 1:
                # Note: This would require modifying the dataset, which is more complex in JAX
                # For now, we'll skip this feature or implement it differently
                pass

        # save the model
        save_params(params, f"checkpoints/{cfg.wandb.name}_{timestamp}/last.pkl")

        return train_losses, val_losses, val_accs

    ##########################################################################
    # Train the model
    ##########################################################################
    print("=== Training joint mixture model ===")
    train_losses, val_losses, val_accs = train(
        model, 
        params,
        train_loader_fn,
        val_loader_fn,
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs, 
        n_steps_to_eval=cfg.training.n_steps_to_eval,
        lr_schedule=cfg.training.lr_schedule,
        scheduler_eta_min=cfg.training.get('scheduler_eta_min', 0.0),
        weight_by_ignores=cfg.training.get('weight_by_ignores', False),
    )

    print("=== Training finished ===")


if __name__ == "__main__":
    main()
