import torch
from torch import nn, Tensor, einsum, autocast
import torch.nn.functional as F
from x_transformers import Encoder, Decoder, TransformerWrapper
from einops import rearrange, repeat, reduce, pack, unpack
from random import randrange, random
from contextlib import nullcontext
import math
import os

from x_transformers.x_transformers import LayerIntermediates
from x_transformers.x_transformers import default, exists, first, cast_tuple
from x_transformers.x_transformers import log, masked_mean, pad_at_dim, calc_entropy, calc_z_loss

import pdb



class CustomTransformerWrapper(TransformerWrapper):
    def __init__(self, cfg, attn_layers=None, num_tokens=None):
        super().__init__(
            num_tokens = num_tokens,
            max_seq_len = cfg.model.max_seq_len,
            attn_layers = attn_layers,
            emb_dropout = cfg.model.emb_dropout,
        )
        self.use_conv1d_embed = cfg.model.use_conv1d_embed
        if self.use_conv1d_embed:
            kernel_size = cfg.model.conv1d_embed.kernel_size
            padding = kernel_size // 2
            self.conv1d_after_embed = []
            for _ in range(cfg.model.conv1d_embed.num_layers):
                layer = nn.Conv1d(cfg.model.attn_layers.dim, cfg.model.attn_layers.dim,
                                  kernel_size=kernel_size, padding=padding)
                self.conv1d_after_embed += [layer, nn.BatchNorm1d(cfg.model.attn_layers.dim), nn.ReLU()]
            self.conv1d_after_embed = nn.Sequential(*self.conv1d_after_embed)
        else:
            self.conv1d_after_embed = None


    ### Overwrite the forward method to add the 1D convolution after the embedding
    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        return_logit_entropies = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: LayerIntermediates | None = None,
        token_emb_kwargs = dict(),
        to_logits_kwargs = dict(),
        **kwargs,
    ):
        b, seq_len, device, num_mems, has_memory_tokens, emb_frac_gradient, orig_mask = x.shape[0], x.shape[1], x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient, mask

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss
        return_embeddings = return_embeddings | (not exists(self.to_logits))

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        # add additional embeddings

        assert not (exists(self.embeds) ^ (len(embed_ids) > 0)), '`embed_num_tokens` must be defined on `TransformerWrapper`'

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, seq_len), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, '... -> b ...', b = b)
            x, cls_packed_shape = pack([cls_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value = True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - seq_len), dim = -2, value = 0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]
        
        if self.conv1d_after_embed is not None:
            x = x.permute(0, 2, 1)
            x = self.conv1d_after_embed(x)
            x = x.permute(0, 2, 1)

        # attention layers

        if self.attn_layers is not None:
            if not self.recycling:
                assert not exists(recycle_steps) or recycle_steps == 1, 'you did not train with recycling'

                # regular

                attended, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache,
                                                          return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

            else:
                # recycling

                recycle_steps = default(recycle_steps, (randrange(self.train_max_recycle_steps) + 1) if self.training else None)
                assert exists(recycle_steps) and recycle_steps > 0, '`recycle_steps` must be provided on forward if recycling is turned on and not training'

                for i in range(recycle_steps):
                    first_step = i == 0
                    last_step = i == (recycle_steps - 1)

                    context = nullcontext if last_step else torch.no_grad

                    with context():
                        maybe_recycled = self.recycled_proj(attended.detach()) if not first_step else 0.

                        attended, intermediates = self.attn_layers(x + maybe_recycled, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

            x = attended

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            x = x[:, :mem_seq]

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        if exists(self.cls_token):
            x, _ = unpack(x, cls_packed_shape, 'b * d')
            x = x.squeeze(1)  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # handle expansion to mixture if needed (for mixture of softmax)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim = -1)
            x = self.to_mixture(x)

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        # maybe sig softmax

        if self.sigsoftmax_logits:
            logits = logits + logits.sigmoid().log()

        # handle maybe combine mixture

        if exists(combine_mixture):
            with autocast('cuda', enabled = False):
                prob = logits.softmax(dim = -1)
                mos = einsum('... k d, ... k -> ... d', prob, combine_mixture)
                logits = log(mos)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple((rearrange(t, '... 1 -> ...') if t.shape[-1] == 1 else t) for t in cast_tuple(logits))

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings:
            out = x
        else:
            out = logits

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in  intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = [torch.cat(pair, dim = -2) for pair in zip(mems, hiddens)] if exists(mems) else hiddens
            new_mems = [t[..., -self.max_mem_len:, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out

def get_model(cfg, num_tokens):

    attn_layers = Encoder(
            dim = cfg.model.attn_layers.dim,
            depth = cfg.model.attn_layers.depth,
            heads = cfg.model.attn_layers.heads,
            layer_dropout = cfg.model.attn_layers.layer_dropout
    )
    model = CustomTransformerWrapper(
        cfg = cfg,
        num_tokens = num_tokens,
        attn_layers = attn_layers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.model.load_ckpt and cfg.model.ckpt_path and os.path.exists(cfg.model.ckpt_path):
        print(f"Loading checkpoint from {cfg.model.ckpt_path}")
        state_dict = torch.load(cfg.model.ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
      
    model.to(device)

    return model