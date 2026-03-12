import os
import pickle
from tqdm import tqdm

import jax
from jax import numpy as jnp
from jax import random as jr


class FlipFlopAutomaton:
    # NOTE: code adapted from https://huggingface.co/datasets/synthseq/automata/blob/main/automata.py
    def __init__(self, n_states, length, random_length, seed, n_ignores, p_ignores, n_samples, fdata):
        self.name = 'flipflop'

        self.n_states = n_states
        self.length = length
        self.random_length = random_length
        self.seed = seed
        self.n_ignores = n_ignores
        self.p_ignores = p_ignores
        self.n_samples = n_samples

        # data: generated, or loaded from file
        if fdata != '' and os.path.exists(fdata):
            # load a pickle file
            with open(fdata, 'rb') as f:
                samples = pickle.load(f)
                # Convert numpy arrays to JAX arrays
                import numpy as np
                if isinstance(samples['X'], np.ndarray):
                    self.X = jnp.array(samples['X'])
                else:
                    self.X = samples['X']
                if isinstance(samples['y'], np.ndarray):
                    self.y = jnp.array(samples['y'])
                else:
                    self.y = samples['y']
            self.n_samples = len(self.X)
        else:
            self.X = None
            self.key = jr.PRNGKey(self.seed)
            if self.n_ignores != '':
                self.n_ignores = [int(each) for each in self.n_ignores.split(';')]
                # print("n_ignores", self.n_ignores)
            else:
                self.n_ignores = None
                self.p_ignores = [float(each) for each in self.p_ignores.split(';')]
                # print("p_ignores", self.p_ignores)

            self.T = self.length
            self.random_length = self.random_length # whether to vary the sequence length

            self.n_states = n_states 
            self.n_actions = self.n_states + 1
            self.transition = jnp.array([list(range(self.n_actions))] + [[i+1]*self.n_actions for i in range(self.n_states)]).T
            
            # Generate all samples up front (parallelized)
            # Generate all random keys upfront
            keys = jr.split(self.key, self.n_samples)
            
            # Define a function to generate one sample
            # For vmap to work, all outputs must have the same shape, so we generate
            # all samples with max length T, then handle variable lengths via masking
            def generate_sample(key):
                # Split key for different random operations
                key1, key2, key3, key4, key5 = jr.split(key, 5)
                
                # Sample actual length (for variable length case)
                if self.random_length:
                    raise NotImplementedError("Random length not implemented")
                
                # Always generate with max length T for vmap compatibility
                # T = self.T
                
                # get ignore positions
                if self.n_ignores is not None:
                    curr_n_ignore = jr.choice(key2, jnp.array(self.n_ignores))
                    # Generate a random permutation of all indices
                    permuted_indices = jr.permutation(key3, jnp.arange(self.T))
                    # Create mask: first curr_n_ignore elements should be ignored
                    mask = jnp.arange(self.T) < curr_n_ignore
                    # Set ignore_pos directly: positions that appear early in permutation are ignored
                    ignore_pos = jnp.zeros(self.T, dtype=bool)
                    ignore_pos = ignore_pos.at[permuted_indices].set(mask)
                else:
                    curr_p_ignore = jr.choice(key2, jnp.array(self.p_ignores))
                    ignore_pos_full = jr.uniform(key3, shape=(self.T,)) < curr_p_ignore
                    # Only apply ignore probability up to actual length
                    ignore_pos = jnp.where(jnp.arange(self.T) < self.T, ignore_pos_full, False)
                
                ignore_pos = ignore_pos.at[0].set(False) # ensure that the first position is always a write
                
                # get writes
                writes = jr.randint(key4, shape=(self.T,), minval=1, maxval=self.n_states+1)
                x = writes * (1 - ignore_pos.astype(int))
                
                y = self.f(x)
                
                return x, y
            
            # Vectorize the sample generation function
            generate_samples_vmap = jax.vmap(generate_sample)
            
            # Generate all samples in parallel
            print(f"Generating {self.n_samples} samples in parallel...")
            self.X, self.y = generate_samples_vmap(keys)


        self.__info__ = f"Flipflop (yay!) with n={self.n_states} states:\n" \
            +f"- Inputs: tokens are either 0 (read) or 1:{self.n_states} (write).\n" \
            + "- Labels: the state id.\n" \
            + "- Config:\n" \
            + "  - n (int): number of write states; i.e. the states are 1,2,...,n, plus a default start state 0.\n" \
            # + self.__info__

    def f(self, x):
        state, states = 0, []
        for action_id in x:
            state = self.transition[state, action_id]
            states += state,
        return jnp.array(states)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        ignore_percentage = (x == 0).astype(float).mean().item()
        return x, y, ignore_percentage

    def sample_length(self, key):
        if self.random_length:
            return int(jr.choice(key, jnp.arange(1, self.T+1)))
        return self.T

    def __len__(self):
        return len(self.X)


def get_flipflop_labels(seq):
    seq_len = len(seq)
    state = seq[0]
    labels = [state]
    for i in range(1, seq_len):
        if seq[i] != 0:
            state = seq[i]
        labels.append(state)
    if isinstance(seq, jnp.ndarray):
        labels = jnp.array(labels)
    else:
        import numpy as np
        labels = np.array(labels)
    return labels


def get_loaders(cfg_data):
    """
    Returns data generators compatible with JAX training loops.
    Instead of PyTorch DataLoaders, returns iterable datasets and batch functions.
    """
    if cfg_data.task == 'flipflop':
        train_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                      cfg_data.n_ignores_train, cfg_data.p_ignores_train, cfg_data.n_samples_train,
                                      cfg_data.fdata_train)
        val_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                    cfg_data.n_ignores_val, cfg_data.p_ignores_val, cfg_data.n_samples_val,
                                    cfg_data.fdata_val)
    else:
        raise ValueError(f"Task {cfg_data.task} not supported")
    
    def batch_generator(dataset, batch_size, shuffle=True, key=None):
        """Generate batches from dataset. Uses direct array indexing (no per-sample .item() syncs)."""
        n_samples = len(dataset)
        indices = jnp.arange(n_samples)
        if shuffle:
            if key is None:
                key = jr.PRNGKey(42)
            key, subkey = jr.split(key)
            indices = jr.permutation(subkey, indices)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_x = dataset.X[batch_indices]
            batch_y = dataset.y[batch_indices]
            # Vectorized ignore percentage per sample (no device round-trips)
            batch_ignore = (batch_x == 0).astype(jnp.float32).mean(axis=-1)
            yield batch_x, batch_y, batch_ignore
    
    def train_loader(key=None):
        return batch_generator(train_set, cfg_data.batch_size, shuffle=True, key=key)
    
    def val_loader(key=None):
        return batch_generator(val_set, cfg_data.eval_batch_size, shuffle=False, key=key)
    
    return train_loader, val_loader


def get_adv_example(model, xs, ys, n_states, n_iters=10, n_random_pos=5,
                    check_acc=0, acc_threshold=0.6):
    """
    Generate adversarial examples using JAX.
    Note: model should be a JAX-compatible function that takes xs and returns logits.
    Args:
        model: JAX-compatible model function
        xs: input sequences [batch_size, seq_len]
        ys: target labels [batch_size, seq_len]
        n_states: number of states in the automaton
        n_iters: number of iterations
        n_random_pos: number of random positions to try per iteration
        check_acc: whether to check accuracy during generation
        acc_threshold: accuracy threshold to stop early
    """
    accs = []
    for i in range(n_iters):
        logits = model(xs)
        if i == 0 and check_acc:
            acc_orig = (jnp.argmax(logits, axis=-1) == ys).astype(float).mean().item()
            accs.append(acc_orig)
        
        # identify the most confident position
        # entropies shape: [batch_size, seq_len]
        probs = jax.nn.softmax(logits, axis=-1)
        entropies = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        # min_entropy_pos: per-sample position with minimum entropy [batch_size]
        min_entropy_pos = jnp.argmin(entropies, axis=-1)
        min_entropy_pos = jnp.where(min_entropy_pos == 0, 1, min_entropy_pos)
        min_entropy = entropies[jnp.arange(xs.shape[0]), min_entropy_pos]
        
        # perturb the input sequence at diff random positions
        key = jr.PRNGKey(i)
        for _ in range(n_random_pos):
            # choose a random position before the current one (using first sample's min pos)
            key, subkey = jr.split(key)
            # Use the minimum entropy position for the first sample in batch
            max_pos = int(min_entropy_pos[0])
            if max_pos <= 0:
                max_pos = 1
            rand_pos = jr.randint(subkey, shape=(), minval=0, maxval=max_pos)
            rand_pos = int(rand_pos)
            
            # compare the n_states number of other inputs for this sampled position
            best_candidate = xs
            best_candidate_entropy = min_entropy
            
            # prepare the batch of perturbed inputs
            for si in range(n_states+1):
                # Create perturbed version where rand_pos is set to si
                other_xs = xs.at[:, rand_pos].set(si)
                # forward pass on the batch at once
                logits = model(other_xs)
                # check the entropy of the perturbed input
                probs_perturbed = jax.nn.softmax(logits, axis=-1)
                entropies_perturbed = -jnp.sum(probs_perturbed * jnp.log(probs_perturbed + 1e-10), axis=-1)
                curr_entropy = entropies_perturbed[jnp.arange(xs.shape[0]), min_entropy_pos]
                # update the best candidate & entropy if the entropy is higher
                update_sample_mask = curr_entropy > best_candidate_entropy
                best_candidate = jnp.where(update_sample_mask[:, None], other_xs, best_candidate)
                best_candidate_entropy = jnp.where(update_sample_mask, curr_entropy, best_candidate_entropy)
            
            xs = best_candidate

        if check_acc:
            # check the accuracy on the perturbed input
            logits = model(xs)
            acc_perturbed = (jnp.argmax(logits, axis=-1) == ys).astype(float).mean().item()
            accs.append(acc_perturbed)
            if acc_perturbed < acc_threshold:
                break
    
    return xs, ys, accs
