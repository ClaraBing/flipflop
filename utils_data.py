import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
        if self.X is None:
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
            
            return x, self.f(x)
        else:
            return self.X[idx], self.y[idx]

    def sample_length(self):
        if self.random_length:
            return self.np_rng.choice(range(1, self.T+1))
        return self.T

    def __len__(self):
        if self.X is None:
            return self.n_samples
        else:
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


def get_loaders(cfg_data):
    if cfg_data.task == 'flipflop':
        train_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                      cfg_data.n_ignores_train, cfg_data.p_ignores_train, cfg_data.n_samples_train,
                                      cfg_data.fdata_train)
        val_set = FlipFlopAutomaton(cfg_data.n_states, cfg_data.length, cfg_data.random_length, cfg_data.seed,
                                    cfg_data.n_ignores_val, cfg_data.p_ignores_val, cfg_data.n_samples_val,
                                    cfg_data.fdata_val)
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


