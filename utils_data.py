import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
from sympy.combinatorics.permutations import Permutation
import torch.nn.functional as F

import pdb



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
                self.X = samples['X']
                self.y = samples['y']
            self.n_samples = len(self.X)
        else:
            self.X = None
            self.np_rng = np.random.default_rng(self.seed)
            if self.n_ignores != '':
                self.n_ignores = [int(each) for each in self.n_ignores.split(';')]
                # print("n_ignores", self.n_ignores)
            else:
                self.n_ignores = None
                self.p_ignores = [float(each) for each in self.p_ignores.split(';')]
                # print("p_ignores", self.p_ignores)

            self.T = self.length
            self.random_length = self.random_length # whether to vary the sequence length
            
            # Generate all samples up front
            self.X = []
            self.y = []
            for _ in tqdm(range(self.n_samples), desc="Generating samples"):
                T = self.sample_length()

                # get ignore positions
                if self.n_ignores is not None:
                    curr_n_ignore = self.np_rng.choice(self.n_ignores)
                    ignore_indices = self.np_rng.choice(range(T), size=curr_n_ignore, replace=False)
                    ignore_pos = np.zeros(T, dtype=bool)
                    ignore_pos[ignore_indices] = True

                else:
                    curr_p_ignore = self.np_rng.choice(self.p_ignores)
                    ignore_pos = self.np_rng.uniform(size=T) < curr_p_ignore
                
                ignore_pos[0] = 0 # ensure that the first position is always a write
                
                # get writes
                writes = self.np_rng.choice(range(1, self.n_states+1), size=T)
                x = writes * (1-ignore_pos)
                
                y = self.f(x)
                self.X.append(x)
                self.y.append(y)
            
            self.X = np.array(self.X)
            self.y = np.array(self.y)

        self.n_states = n_states 
        self.n_actions = self.n_states + 1
        self.transition = np.array([list(range(self.n_actions))] + [[i+1]*self.n_actions for i in range(self.n_states)]).T

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
        return np.array(states)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        ignore_percentage = (x == 0).astype(float).mean().item()
        return x, y, ignore_percentage

    def sample_length(self):
        if self.random_length:
            return self.np_rng.choice(range(1, self.T+1))
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
    if isinstance(seq, torch.Tensor):
        labels = torch.tensor(labels)
    else:
        labels = np.array(labels)
    return labels




class PermutationAutomaton:
    """
    This is a parent class that must be inherited.
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
        fdata='',
    ):
        self.n_states = n_states  # the symmetric group Sn
        self.length = length
        self.random_length = random_length
        self.seed = seed
        self.label_type = label_type  # Options: 'state', 'first_chair'
        self.n_samples = n_samples
        self.fdata = fdata

        self.np_rng = np.random.default_rng(self.seed)

        # Allow subclasses to append their own details.
        self.__info__ = ""
        self.__info__ = (
            "  - label_type (str): choosing from the following options:\n"
            + "    - 'state' (default): the state id.\n"
            + "    - 'first_chair': the element in the first position of the permutation.\n"
            + "          e.g. if the current permutation is [2,1,4,3], then 'first_chair' is 2.\n"
            + self.__info__
        )

        # Subclasses must set up:
        # - self.actions (dict[int -> np.ndarray])
        # - self.n_actions (int)
        # - self.state_encode (callable)
        # - self.state_label_map (dict[str -> int])
        self.X = None
        self.y = None

    def get_state_label(self, state):
        enc = self.state_encode(state)
        return self.state_label_map[enc]

    def f(self, x):
        curr_state = np.arange(self.n_states)
        labels = []
        for action_id in x:
            curr_state = self.actions[action_id].dot(curr_state)

            if self.label_type == 'state':
                labels += self.get_state_label(curr_state),
            elif self.label_type == 'first_chair':
                labels += curr_state[0],
        return np.array(labels)

    def sample_length(self):
        if self.random_length:
            # NOTE: variable-length sequences will produce a ragged dataset (dtype=object)
            return int(self.np_rng.choice(range(1, self.length + 1)))
        return self.length

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_actions), replace=True, size=T)  # TODO: check if this is correct

        return x, self.f(x)

    def _init_samples(self):
        # data: generated, or loaded from file
        if self.fdata != "" and os.path.exists(self.fdata):
            with open(self.fdata, "rb") as f:
                samples = pickle.load(f)
                self.X = samples["X"]
                self.y = samples["y"]
            self.n_samples = len(self.X)
            return

        self.X = []
        self.y = []
        for _ in tqdm(range(self.n_samples), desc="Generating samples"):
            x, y = self.sample()
            self.X.append(x)
            self.y.append(y)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X) if self.X is not None else int(self.n_samples)


class SymmetricAutomaton(PermutationAutomaton):
    """
    TODO: add options for labels as functions of states
    - parity (whether a state is even): this may need packages (e.g. Permutation from sympy)
    - position / toggle: for S3 ~ D6, we can add labels for substructures as in Dihedral groups.
    """
    def __init__(
        self,
        n_states,
        length,
        random_length,
        seed,
        label_type,
        n_actions,
        n_samples,
        fdata='',
    ):
        super().__init__(
            n_states=n_states,
            length=length,
            random_length=random_length,
            seed=seed,
            n_samples=n_samples,
            fdata=fdata,
            label_type=label_type,
        )

        self.name = f'S{self.n_states}'

        """
        Get states
        """
        self.state_encode = lambda state: ''.join([str(int(each)) for each in state])
        self.state_label_map = {}
        for si, state in enumerate(itertools.permutations(range(self.n_states))):
            enc = self.state_encode(state)
            self.state_label_map[enc] = si

        """
        Get actions (3 defaults: id, shift-by-1, swap-first-two)
        """
        self.n_actions = n_actions
        self.actions = {0: np.eye(self.n_states)}
        # shift all elements to the right by 1
        shift_idx = list(range(1, self.n_states)) + [0]
        self.actions[1] = np.eye(self.n_states)[shift_idx]
        # swap the first 2 elements
        shift_idx = [1, 0] + list(range(2, self.n_states))  # TODO: check if this is correct
        self.actions[2] = np.eye(self.n_states)[shift_idx]

        if self.n_actions > 3:  # TODO: check if this is correct
            # add permutations in the order given by itertools.permutations
            self.all_permutations = list(itertools.permutations(range(self.n_states)))[1:]
            cnt = 2
            for each in self.all_permutations:
                action = np.eye(self.n_states)[list(each)]
                if np.linalg.norm(action - self.actions[0]) == 0:
                    continue
                elif np.linalg.norm(action - self.actions[1]) == 0:
                    continue
                self.actions[cnt] = action
                cnt += 1
                if cnt == self.n_actions: break

        self.__info__ = f"Symmetric group on n={self.n_states} objects:\n" \
            +f"- Inputs: tokens are either 0 (no-op), or 1:{self.n_actions} (corresponding to {self.n_actions} permutations).\n" \
            + "- Labels: depending on 'label_type'.\n" \
            + "- Config:\n" \
            + "  - n_states (int): number of objects, i.e. there are n! states.\n" \
            + "  - n_actions (int): number of permutations to include in the generator set;\n" \
            + "        the ordering is given by itertools.permutations, and the first 'n_actions' permutations will be included.\n" \
            + self.__info__ 

        self._init_samples()


class AlternatingAutomaton(PermutationAutomaton):
    """
    TODO: other choices of generators (currently using (12x))?
    """
    def __init__(
        self,
        n_states,
        length,
        random_length,
        seed,
        label_type,
        n_samples,
        fdata='',
    ):
        super().__init__(
            n_states=n_states,
            length=length,
            random_length=random_length,
            seed=seed,
            n_samples=n_samples,
            fdata=fdata,
            label_type=label_type,
        )

        self.name = f'A{self.n_states}'

        """
        Get states
        """
        self.state_label_map = {}
        self.state_encode = lambda state: ''.join([str(int(each)) for each in state])
        cnt = 0
        for si, state in enumerate(itertools.permutations(range(self.n_states))):
            if not Permutation(state).is_even:
                continue
            enc = self.state_encode(state)
            self.state_label_map[enc] = cnt
            cnt += 1

        """
        Get actions: all 3 cycles of the form (12x)
        """
        self.actions = {0: np.eye(self.n_states)}
        for idx in range(2, self.n_states):
            # (1, 2, idx) 
            shift_idx = list(range(self.n_states))
            shift_idx[0],shift_idx[1], shift_idx[idx] = shift_idx[1], shift_idx[idx], shift_idx[0]
            self.actions[idx-1] = np.eye(self.n_states)[shift_idx]
        self.n_actions = len(self.actions)

        self.__info__ = f"Alternating group on n={self.n_states} objects:\n" \
            +f"- Inputs: tokens from 0 to n-3, corresponding to all 3-cycles of the form (12x).\n" \
            + "- Labels: depending on 'label_type'.\n" \
            + "- Config:\n" \
            + "  - n_states (int): number of objects, i.e. there are n!/2 states.\n" \
            + self.__info__ 

        self._init_samples()


def get_loaders(cfg_data):
    if cfg_data.task == 'flipflop':
        train_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                      cfg_data.n_ignores_train, cfg_data.p_ignores_train, cfg_data.n_samples_train,
                                      cfg_data.fdata_train)
        val_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                    cfg_data.n_ignores_val, cfg_data.p_ignores_val, cfg_data.n_samples_val,
                                    cfg_data.fdata_val)
    elif cfg_data.task == 'symmetric':
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
    elif cfg_data.task == 'alternating':
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
            
    train_loader = DataLoader(train_set, batch_size=cfg_data.batch_size,
                    shuffle=True, num_workers=cfg_data.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg_data.eval_batch_size,
                    shuffle=False, num_workers=cfg_data.num_workers)
    return train_loader, val_loader




def get_adv_example(model, xs, ys, n_iters=10, n_random_pos=5,
                    check_acc=0, acc_threshold=0.6):
    accs = []
    for i in range(n_iters):
        with torch.no_grad():
            logits = model(xs)
        if i == 0 and check_acc:
            acc_orig = (torch.argmax(logits, dim=-1) == ys).float().mean().item()
            accs.append(acc_orig)
        
        # identify the most confident position
        probs = F.softmax(logits, dim=-1)
        entropies = -torch.sum(probs * torch.log(probs), dim=-1)
        min_entropy_pos = torch.argmin(entropies)
        min_entropy_pos[min_entropy_pos == 0] = 1
        min_entropy = entropies[torch.arange(xs.shape[0]), min_entropy_pos]
        
        # perturb the input sequence at diff random positions
        for _ in range(n_random_pos):
            # choose a random position before the current one
            rand_pos = torch.randint(0, min_entropy_pos, (1,))
            # compare the n_states number of other inputs for this sampled position
            best_candidate = xs.clone()
            best_candidate_entropy = min_entropy.clone()
            other_xs = xs.clone()
            curr_max_entropy = 0
            # prepare the batch of perturbed inputs
            for si in range(n_states+1):
                for i in range(xs.shape[0]):
                    if si == xs[i, rand_pos]:
                        continue
                    other_xs[i, rand_pos] = si
                # forward pass on the batch at once
                with torch.no_grad():
                    logits = model(other_xs)
                # check the entropy of the perturbed input
                probs_perturbed = F.softmax(logits, dim=-1)
                entropies_perturbed = -torch.sum(probs_perturbed * torch.log(probs_perturbed), dim=-1)
                curr_entropy = entropies_perturbed[torch.arange(xs.shape[0]), min_entropy_pos]
                # update the best candidate & entropy if the entropy is higher
                update_sample_mask = curr_entropy > best_candidate_entropy
                best_candidate[update_sample_mask] = other_xs[update_sample_mask]
                best_candidate_entropy[update_sample_mask] = curr_entropy[update_sample_mask]
            
            xs = best_candidate

        if check_acc:
            # check the accuracy on the perturbed input
            with torch.no_grad():
                logits = model(xs)
            acc_perturbed = (torch.argmax(logits, dim=-1) == ys).float().mean().item()
            accs.append(acc_perturbed)
            if acc_perturbed < 0.6:
                break
    
    return xs, ys, accs


