import os
import pickle
from tqdm import tqdm

import jax
from jax import numpy as jnp
from jax import random as jr
import numpy as np
import itertools


class FlipFlopAutomaton:
    # NOTE: code adapted from https://huggingface.co/datasets/synthseq/automata/blob/main/automata.py
    def __init__(self, n_states, length, random_length, seed, n_ignores, p_ignores, n_samples, fdata,
                 display=None):
        self.name = 'flipflop'

        self.n_states = n_states
        self.length = length
        self.random_length = random_length
        self.seed = seed
        self.n_ignores = n_ignores
        self.p_ignores = p_ignores
        self.n_samples = n_samples

        if display is None:
            display = f"n_ignores={n_ignores}, p_ignores={p_ignores}, n_samples={n_samples}"
        self.display = display

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
            raise NotImplementedError("Random length not implemented")
            # return int(jr.choice(key, jnp.arange(1, self.T+1)))
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


class PermutationAutomaton:
    """
    Parent class for permutation-group automata.
    Subclasses: SymmetricAutomaton, AlternatingAutomaton
    """

    def __init__(
        self,
        n_states,
        length,
        random_length,
        seed,
        n_samples,
        label_type,
        fdata="",
    ):
        self.n_states = int(n_states)  # number of objects being permuted
        self.length = int(length)
        self.random_length = int(random_length)
        self.seed = int(seed)
        self.n_samples = int(n_samples)
        self.label_type = str(label_type)  # 'state' or 'first_chair'
        self.fdata = fdata

        self.np_rng = np.random.default_rng(self.seed)

        # Subclasses must define:
        # - self.actions: dict[int -> np.ndarray] (permutation matrices)
        # - self.n_actions: int
        # - self.state_encode: callable(state_vec)->str
        # - self.state_label_map: dict[str->int]
        self.X = None
        self.y = None

        self.__info__ = (
            "  - label_type (str): choosing from the following options:\n"
            "    - 'state' (default): the state id.\n"
            "    - 'first_chair': the element in the first position of the permutation.\n"
            "          e.g. if the current permutation is [2,1,4,3], then 'first_chair' is 2.\n"
        )

    def get_state_label(self, state_vec):
        enc = self.state_encode(state_vec)
        return self.state_label_map[enc]

    def f(self, x):
        curr_state = np.arange(self.n_states)
        labels = []
        for action_id in x:
            curr_state = self.actions[int(action_id)].dot(curr_state)
            if self.label_type == "state":
                labels.append(self.get_state_label(curr_state))
            elif self.label_type == "first_chair":
                labels.append(int(curr_state[0]))
            else:
                raise ValueError(f"Unknown label_type: {self.label_type}")
        return np.asarray(labels, dtype=np.int32)

    def sample_length(self):
        if self.random_length:
            # variable-length would create ragged arrays; keep parity with torch code
            return int(self.np_rng.choice(range(1, self.length + 1)))
        return self.length

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_actions), replace=True, size=T).astype(np.int32)
        y = self.f(x)
        return x, y

    def _init_samples(self):
        if self.fdata != "" and os.path.exists(self.fdata):
            with open(self.fdata, "rb") as f:
                samples = pickle.load(f)
            self.X = jnp.asarray(samples["X"], dtype=jnp.int32)
            self.y = jnp.asarray(samples["y"], dtype=jnp.int32)
            self.n_samples = int(len(self.X))
            return

        X = []
        y = []
        for _ in tqdm(range(self.n_samples), desc="Generating samples"):
            xi, yi = self.sample()
            X.append(xi)
            y.append(yi)

        # These tasks are typically fixed-length; if random_length is enabled,
        # numpy will create dtype=object and this will fail (intended).
        self.X = jnp.asarray(np.asarray(X), dtype=jnp.int32)
        self.y = jnp.asarray(np.asarray(y), dtype=jnp.int32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return int(len(self.X)) if self.X is not None else int(self.n_samples)


class SymmetricAutomaton(PermutationAutomaton):
    def __init__(
        self,
        n_states,
        length,
        random_length,
        seed,
        label_type,
        n_actions,
        n_samples,
        fdata="",
    ):
        super().__init__(
            n_states=n_states,
            length=length,
            random_length=random_length,
            seed=seed,
            n_samples=n_samples,
            label_type=label_type,
            fdata=fdata,
        )

        self.name = f"S{self.n_states}"

        # State encoding / labeling: enumerate all permutations of 0..n-1
        self.state_encode = lambda state: "".join(str(int(each)) for each in state)
        self.state_label_map = {}
        for si, state in enumerate(itertools.permutations(range(self.n_states))):
            self.state_label_map[self.state_encode(state)] = int(si)

        # Actions (default 3): id, shift-by-1, swap-first-two, plus optional extras
        self.n_actions = int(n_actions)
        self.actions = {0: np.eye(self.n_states, dtype=np.int32)}
        shift_idx = list(range(1, self.n_states)) + [0]
        self.actions[1] = np.eye(self.n_states, dtype=np.int32)[shift_idx]
        swap_idx = [1, 0] + list(range(2, self.n_states))
        self.actions[2] = np.eye(self.n_states, dtype=np.int32)[swap_idx]

        if self.n_actions > 3:
            # Add additional permutations in itertools.permutations order, skipping duplicates
            all_perms = list(itertools.permutations(range(self.n_states)))[1:]
            cnt = 2
            for perm in all_perms:
                action = np.eye(self.n_states, dtype=np.int32)[list(perm)]
                if np.array_equal(action, self.actions[0]) or np.array_equal(action, self.actions[1]):
                    continue
                self.actions[cnt] = action
                cnt += 1
                if cnt == self.n_actions:
                    break

        self.__info__ = (
            f"Symmetric group on n={self.n_states} objects:\n"
            f"- Inputs: tokens are either 0 (no-op), or 1:{self.n_actions} (corresponding to {self.n_actions} permutations).\n"
            "- Labels: depending on 'label_type'.\n"
            "- Config:\n"
            "  - n_states (int): number of objects, i.e. there are n! states.\n"
            "  - n_actions (int): number of permutations to include in the generator set;\n"
            "        the ordering is given by itertools.permutations, and the first 'n_actions' permutations will be included.\n"
            + self.__info__
        )

        self._init_samples()


def _is_even_permutation(perm):
    """
    Return True iff perm (tuple/list of 0..n-1) is even.
    Uses inversion parity (O(n^2)), fine for small n.
    """
    inv = 0
    p = list(perm)
    n = len(p)
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, n):
            inv += 1 if pi > p[j] else 0
    return (inv % 2) == 0


class AlternatingAutomaton(PermutationAutomaton):
    def __init__(
        self,
        n_states,
        length,
        random_length,
        seed,
        label_type,
        n_samples,
        fdata="",
    ):
        super().__init__(
            n_states=n_states,
            length=length,
            random_length=random_length,
            seed=seed,
            n_samples=n_samples,
            label_type=label_type,
            fdata=fdata,
        )

        self.name = f"A{self.n_states}"

        # State labeling: only even permutations
        self.state_encode = lambda state: "".join(str(int(each)) for each in state)
        self.state_label_map = {}
        cnt = 0
        for state in itertools.permutations(range(self.n_states)):
            if not _is_even_permutation(state):
                continue
            self.state_label_map[self.state_encode(state)] = int(cnt)
            cnt += 1

        # Actions: all 3-cycles of the form (12x) for x in [2..n-1]
        self.actions = {0: np.eye(self.n_states, dtype=np.int32)}
        for idx in range(2, self.n_states):
            shift_idx = list(range(self.n_states))
            shift_idx[0], shift_idx[1], shift_idx[idx] = (
                shift_idx[1],
                shift_idx[idx],
                shift_idx[0],
            )
            self.actions[idx - 1] = np.eye(self.n_states, dtype=np.int32)[shift_idx]
        self.n_actions = int(len(self.actions))

        self.__info__ = (
            f"Alternating group on n={self.n_states} objects:\n"
            f"- Inputs: tokens from 0 to n-3, corresponding to all 3-cycles of the form (12x).\n"
            "- Labels: depending on 'label_type'.\n"
            "- Config:\n"
            "  - n_states (int): number of objects, i.e. there are n!/2 states.\n"
            + self.__info__
        )

        self._init_samples()


def get_loaders(cfg_data):
    """
    Returns data generators compatible with JAX training loops.
    Instead of PyTorch DataLoaders, returns iterable datasets and batch functions.
    """
    if cfg_data.task == 'flipflop':
        train_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                      cfg_data.n_ignores_train, cfg_data.p_ignores_train, cfg_data.n_samples_train,
                                      cfg_data.fdata_train)
        if cfg_data.eval_separately:
            if cfg_data.n_ignores_val != '':
                # eval separately for each ignore number
                n_ignore_val_list = [str(each) for each in cfg_data.n_ignores_val.split(';')]
                val_set_list = []
                for n_ignore_val in n_ignore_val_list:
                    display = f"n_ignores={n_ignore_val}"
                    val_set = FlipFlopAutomaton(
                        cfg_data.n_states,
                        cfg_data.length,
                        cfg_data.random_length,
                        cfg_data.seed,
                        n_ignores=n_ignore_val,
                        p_ignores=cfg_data.p_ignores_val,
                        n_samples=cfg_data.n_samples_val,
                        fdata=cfg_data.fdata_val,
                        display=display
                    )
                    val_set_list.append(val_set)
            else:
                # eval separately for each ignore percentage
                p_ignores_val_list = [float(each) for each in cfg_data.p_ignores_val.split(';')]
                val_set_list = []
                for p_ignores_val in p_ignores_val_list:
                    display = f"p_ignores={p_ignores_val}"
                    val_set = FlipFlopAutomaton(
                        cfg_data.n_states,
                        cfg_data.length,
                        cfg_data.random_length,
                        cfg_data.seed,
                        cfg_data.n_ignores_val,
                        p_ignores_val,
                        cfg_data.n_samples_val,
                        cfg_data.fdata_val,
                        display=display,
                    )
                    val_set_list.append(val_set)
        else:
            val_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                        cfg_data.n_ignores_val, cfg_data.p_ignores_val, cfg_data.n_samples_val,
                                        cfg_data.fdata_val)
    elif cfg_data.task == "symmetric":
        train_set = SymmetricAutomaton(
            cfg_data.n_states,
            cfg_data.length,
            cfg_data.random_length,
            cfg_data.seed,
            cfg_data.label_type,
            cfg_data.n_actions,
            cfg_data.n_samples_train,
            cfg_data.fdata_train,
        )
        val_set = SymmetricAutomaton(
            cfg_data.n_states,
            cfg_data.length,
            cfg_data.random_length,
            cfg_data.seed,
            cfg_data.label_type,
            cfg_data.n_actions,
            cfg_data.n_samples_val,
            cfg_data.fdata_val,
        )
    elif cfg_data.task == "alternating":
        train_set = AlternatingAutomaton(
            cfg_data.n_states,
            cfg_data.length,
            cfg_data.random_length,
            cfg_data.seed,
            cfg_data.label_type,
            cfg_data.n_samples_train,
            cfg_data.fdata_train,
        )
        val_set = AlternatingAutomaton(
            cfg_data.n_states,
            cfg_data.length,
            cfg_data.random_length,
            cfg_data.seed,
            cfg_data.label_type,
            cfg_data.n_samples_val,
            cfg_data.fdata_val,
        )
    else:
        raise ValueError(f"Task {cfg_data.task} not supported")
    
    def batch_generator(dataset, batch_size, shuffle=True, key=None):
        """Generate batches from dataset."""
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
            if getattr(dataset, "name", None) == "flipflop":
                batch_ignore = (batch_x == 0).astype(jnp.float32).mean(axis=-1)
                yield batch_x, batch_y, batch_ignore
            else:
                yield batch_x, batch_y
    
    def train_loader(key=None):
        return batch_generator(train_set, cfg_data.batch_size, shuffle=True, key=key)

    # Build validation loaders
    if cfg_data.task == 'flipflop' and cfg_data.eval_separately:
        val_loaders = []

        for val_set in val_set_list:
            def make_val_loader(vs):
                def val_loader(key=None):
                    return batch_generator(vs, cfg_data.eval_batch_size, shuffle=False, key=key)
                # Attach display attribute so callers can log with it
                val_loader.display = vs.display
                return val_loader

            val_loaders.append(make_val_loader(val_set))

        return train_loader, val_loaders
    else:
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
