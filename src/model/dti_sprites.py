from copy import deepcopy
from itertools import chain

import torch
from torch.optim import Adam, RMSprop
import torch.nn as nn

from .transformer import PrototypeTransformationNetwork as Transformer, N_HIDDEN_UNITS, N_LAYERS
from .tools import (copy_with_noise, generate_data, create_gaussian_weights, get_clamp_func,
                    create_mlp)
from utils.logger import print_warning


NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2


def layered_composition(layers, masks, occ_grid):
    # LBCHW size of layers and masks and LLB size for occ_grid
    occ_masks = (1 - occ_grid[..., None, None, None].transpose(0, 1) * masks).prod(1)  # LBCHW
    return (occ_masks * masks * layers).sum(0)  # BCHW


class DTISprites(nn.Module):
    name = 'dti_sprites'
    learn_masks = True

    def __init__(self, dataset, n_sprites, n_objects=1, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        else:
            img_size = dataset.img_size
            n_ch = dataset.n_channels

        # Prototypes & masks
        proto_init = kwargs.get('proto_init', 'sample')
        size = kwargs.get('sprite_size', img_size)
        std = kwargs.get('gaussian_weights_std')
        self.add_empty_sprite = kwargs.get('add_empty_sprite', False)
        self.lambda_empty_sprite = kwargs.get('lambda_empty_sprite', 0)
        self.n_sprites = n_sprites + 1 if self.add_empty_sprite else n_sprites
        samples = torch.stack(generate_data(dataset, n_sprites, proto_init, std=std, size=size, value=0.9))
        self.prototype_params = nn.Parameter(samples)
        clamp_name = kwargs.get('use_clamp', 'soft')
        self.clamp_func = get_clamp_func(clamp_name)
        self.mask_params = nn.Parameter(self.init_masks(n_sprites, kwargs.get('mask_init', 'constant'), size, std))
        self.cur_epoch = 0
        self.n_linear_layers = kwargs.get('n_linear_layers', N_LAYERS)
        self.estimate_minimum = kwargs.get('estimate_minimum', False)
        self.greedy_algo_iter = kwargs.get('greedy_algo_iter', 1)
        freeze_sprite = kwargs.get('freeze_sprite', False)
        self.freeze_milestone = freeze_sprite if freeze_sprite else -1
        assert isinstance(self.freeze_milestone, (int,))

        # Sprite transformers
        L = n_objects
        self.n_objects = n_objects
        self.has_layer_tsf = kwargs.get('transformation_sequence_layer', 'identity') not in ['id', 'identity']
        if self.has_layer_tsf:
            layer_kwargs = deepcopy(kwargs)
            layer_kwargs['transformation_sequence'] = kwargs['transformation_sequence_layer']
            layer_kwargs['curriculum_learning'] = kwargs['curriculum_learning_layer']
            self.layer_transformer = Transformer(n_ch, img_size, L, **layer_kwargs)
            self.encoder = self.layer_transformer.encoder
            tsfs = [Transformer(n_ch, size, self.n_sprites, encoder=self.encoder, **kwargs) for k in range(L)]
            self.sprite_transformers = nn.ModuleList(tsfs)
        else:
            if L > 1:
                self.layer_transformer = Transformer(n_ch, img_size, L, transformation_sequence='identity')
            first_tsf = Transformer(n_ch, img_size, self.n_sprites, **kwargs)
            self.encoder = first_tsf.encoder
            tsfs = [Transformer(n_ch, img_size, self.n_sprites, encoder=self.encoder, **kwargs) for k in range(L-1)]
            self.sprite_transformers = nn.ModuleList([first_tsf] + tsfs)

        # Background Transformer
        M = kwargs.get('n_backgrounds', 0)
        self.n_backgrounds = M
        self.learn_backgrounds = M > 0
        if self.learn_backgrounds:
            bkg_init = kwargs.get('bkg_init', 'constant')
            self.bkg_params = nn.Parameter(torch.stack(generate_data(dataset, M, init_type=bkg_init, value=0.5)))
            bkg_kwargs = deepcopy(kwargs)
            bkg_kwargs['transformation_sequence'] = kwargs['transformation_sequence_bkg']
            bkg_kwargs['curriculum_learning'] = kwargs['curriculum_learning_bkg']
            bkg_kwargs['padding_mode'] = 'border'
            self.bkg_transformer = Transformer(n_ch, img_size, M, encoder=self.encoder, **bkg_kwargs)

        # Image composition and aux
        self.pred_occlusion = kwargs.get('pred_occlusion', False)
        if self.pred_occlusion:
            nb_out = int(L * (L - 1) / 2)
            norm = kwargs.get('norm_layer')
            self.occ_predictor = create_mlp(self.encoder.out_ch, nb_out, N_HIDDEN_UNITS, self.n_linear_layers, norm)
            self.occ_predictor[-1].weight.data.zero_()
            self.occ_predictor[-1].bias.data.zero_()
        else:
            self.register_buffer('occ_grid', torch.tril(torch.ones(L, L), diagonal=-1))

        self._criterion = nn.MSELoss(reduction='none')
        self.empty_cluster_threshold = kwargs.get('empty_cluster_threshold', EMPTY_CLUSTER_THRESHOLD / n_sprites)
        self._reassign_cluster = kwargs.get('reassign_cluster', True)
        self.inject_noise = kwargs.get('inject_noise', 0)

    @staticmethod
    def init_masks(K, mask_init, size, std=None):
        if mask_init == 'constant':
            masks = torch.ones(K, 1, *size)
        elif mask_init == 'gaussian':
            assert std is not None
            mask = create_gaussian_weights(size, 1, std)
            masks = mask.unsqueeze(0).expand(K, -1, -1, -1)
        elif mask_init == 'random':
            masks = torch.rand(K, *size)
        else:
            raise NotImplementedError(f'unkwon mask_init: {mask_init}')
        return masks

    @property
    def n_prototypes(self):
        return self.n_sprites

    @property
    def masks(self):
        masks = self.mask_params
        if self.add_empty_sprite:
            masks = torch.cat([masks, torch.zeros(1, *masks[0].shape, device=masks.device)])

        if self.inject_noise and self.training:
            return masks
        else:
            return self.clamp_func(masks)

    @property
    def prototypes(self):
        params = self.prototype_params
        if self.add_empty_sprite:
            params = torch.cat([params, torch.zeros(1, *params[0].shape, device=params.device)])

        return self.clamp_func(params)

    @property
    def backgrounds(self):
        return self.clamp_func(self.bkg_params)

    @property
    def is_layer_tsf_id(self):
        if hasattr(self, 'layer_transformer'):
            return self.layer_transformer.only_id_activated
        else:
            return False

    @property
    def are_sprite_frozen(self):
        return True if self.freeze_milestone > 0 and self.cur_epoch < self.freeze_milestone else False

    def cluster_parameters(self):
        params = [self.prototype_params, self.mask_params]
        if self.learn_backgrounds:
            params.append(self.bkg_params)
        return iter(params)

    def transformer_parameters(self):
        params = [t.parameters() for t in self.sprite_transformers]
        if hasattr(self, 'layer_transformer'):
            params.append(self.layer_transformer.parameters())
        if self.learn_backgrounds:
            params.append(self.bkg_transformer.parameters())
        if self.pred_occlusion:
            params.append(self.occ_predictor.parameters())
        return chain(*params)

    def forward(self, x):
        B, C, H, W = x.size()
        L, K, M = self.n_objects, self.n_sprites, self.n_backgrounds or 1
        tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob = self.predict(x)

        if class_prob is None:
            target = self.compose(tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob)  # B(K**L*M)CHW
            x = x.unsqueeze(1).expand(-1, K**L*M, -1, -1, -1)
            distances = self.criterion(x, target)
            loss = distances.min(1)[0].mean()

        else:
            target = self.compose(tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob)  # BCHW
            loss = self.criterion(x.unsqueeze(1), target.unsqueeze(1)).mean()
            distances = 1 - class_prob.permute(2, 0, 1).flatten(1)  # B(L*K)

        return loss, distances

    def predict(self, x):
        B, C, H, W = x.size()
        h, w = self.prototypes.shape[2:]
        L, K, M = self.n_objects, self.n_sprites, self.n_backgrounds or 1
        prototypes = self.prototypes.unsqueeze(1).expand(K, B, C, -1, -1)
        masks = self.masks.unsqueeze(1).expand(K, B, 1, -1, -1)
        sprites = torch.cat([prototypes, masks], dim=2)
        if self.inject_noise and self.training:
            # XXX we use a canva to inject noise after transformations to avoid gridding artefacts
            if self.add_empty_sprite:
                canvas = torch.cat([torch.ones(K - 1, B, 1, h, w), torch.zeros(1, B, 1, h, w)]).to(x.device)
            else:
                canvas = torch.ones(K, B, 1, h, w, device=x.device)
            sprites = torch.cat([sprites, canvas], dim=2)
        if self.are_sprite_frozen:
            sprites = sprites.detach()

        features = self.encoder(x)
        tsf_sprites = torch.stack([self.sprite_transformers[k](x, sprites, features)[1] for k in range(L)], dim=0)
        if self.has_layer_tsf:
            layer_features = features.unsqueeze(1).expand(-1, K, -1).reshape(B*K, -1)
            tsf_layers = self.layer_transformer(x, tsf_sprites.view(L, B*K, -1, h, w), layer_features)[1]
            tsf_layers = tsf_layers.view(B, K, L, -1, H, W).transpose(0, 2)  # LKBCHW
        else:
            tsf_layers = tsf_sprites.transpose(1, 2)  # LKBCHW

        if self.inject_noise and self.training:
            tsf_layers, tsf_masks, tsf_noise = torch.split(tsf_layers, [C, 1, 1], dim=3)
        else:
            tsf_layers, tsf_masks = torch.split(tsf_layers, [C, 1], dim=3)

        if self.learn_backgrounds:
            backgrounds = self.backgrounds.unsqueeze(1).expand(M, B, C, -1, -1)
            tsf_bkgs = self.bkg_transformer(x, backgrounds, features)[1].transpose(0, 1)  # MBCHW
        else:
            tsf_bkgs = None

        if self.inject_noise and self.training:
            noise = torch.rand(K, 1, H, W, device=x.device)[None, None, ...].expand(L, B, K, 1, H, W).transpose(1, 2)
            tsf_masks = tsf_masks + tsf_noise * (2 * self.inject_noise * noise - self.inject_noise)
            tsf_masks = self.clamp_func(tsf_masks)

        occ_grid = self.predict_occlusion_grid(x, features)  # LLB
        if self.estimate_minimum:
            class_prob = self.greedy_algo_selection(x, tsf_layers, tsf_masks, tsf_bkgs, occ_grid)  # LKB
            self._class_prob = class_prob  # for monitoring and debug only
        else:
            class_prob = None

        return tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob

    def predict_occlusion_grid(self, x, features):
        B, L = x.size(0), self.n_objects
        if self.pred_occlusion:
            inp = features if features is not None else x
            occ_grid = self.occ_predictor(inp)  # view(-1, L, L)
            occ_grid = torch.sigmoid(occ_grid)
            grid = torch.zeros(B, L, L, device=x.device)
            indices = torch.tril_indices(row=L, col=L, offset=-1)
            grid[:, indices[0], indices[1]] = occ_grid
            occ_grid = grid + torch.triu(1 - grid.transpose(1, 2), diagonal=1)
        else:
            occ_grid = self.occ_grid.unsqueeze(0).expand(B, -1, -1)

        return occ_grid.permute(1, 2, 0)  # LLB

    @torch.no_grad()
    def greedy_algo_selection(self, x, layers, masks, bkgs, occ_grid):
        L, K, B, C, H, W = layers.shape
        if self.add_empty_sprite and self.are_sprite_frozen:
            layers, masks = layers[:, :-1], masks[:, :-1]
            K = K - 1
        x, device = x.unsqueeze(0).expand(K, -1, -1, -1, -1), x.device
        bkgs = torch.zeros(1, B, C, H, W, device=device) if bkgs is None else bkgs
        cur_layers = torch.cat([bkgs, torch.zeros(L, B, C, H, W, device=device)])
        cur_masks = torch.cat([torch.ones(1, B, 1, H, W, device=device), torch.zeros(L, B, 1, H, W, device=device)])
        one, zero = torch.ones(B, L, 1, device=device), torch.zeros(B, 1, L + 1, device=device)
        occ_grid = torch.cat([zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1).permute(1, 2, 0)

        resps, diff_select = torch.zeros(L, K, B, device=device), [[], []]
        for step in range(self.greedy_algo_iter):
            for l, (layer, mask) in enumerate(zip(layers, masks), start=1):
                recons = []
                for k in range(K):
                    tmp_layers = torch.cat([cur_layers[:l], layer[[k]], cur_layers[l+1:]])
                    tmp_masks = torch.cat([cur_masks[:l], mask[[k]], cur_masks[l+1:]])
                    recons.append(layered_composition(tmp_layers, tmp_masks, occ_grid))
                distance = ((x - torch.stack(recons))**2).flatten(2).mean(2)
                if self.add_empty_sprite and not self.are_sprite_frozen:
                    distance += self.lambda_empty_sprite * torch.Tensor([1]*(K-1) + [0]).to(device)[:, None]
                resp = torch.zeros(K, B, device=device).scatter_(0, distance.argmin(0, keepdim=True), 1)
                resps[l - 1] = resp
                cur_layers[l] = (layer * resp[..., None, None, None]).sum(axis=0)
                cur_masks[l] = (mask * resp[..., None, None, None]).sum(axis=0)

            if True:
                # For debug purposes only
                if step == 0:
                    indices = resps.argmax(1).flatten()
                else:
                    new_indices = resps.argmax(1).flatten()
                    diff_select[0].append(str(step))
                    diff_select[1].append((new_indices != indices).float().mean().item())
                    indices = new_indices
        # For debug purposes only
        if step > 0:
            self._diff_selections = diff_select

        if self.add_empty_sprite and self.are_sprite_frozen:
            resps = torch.cat([resps, torch.zeros(L, 1, B, device=device)], dim=1)
        return resps

    def compose(self, layers, masks, occ_grid, backgrounds=None, class_prob=None):
        L, K, B, C, H, W = layers.shape
        device = occ_grid.device

        if class_prob is not None:
            masks = (masks * class_prob[..., None, None, None]).sum(axis=1)
            layers = (layers * class_prob[..., None, None, None]).sum(axis=1)
            size = (B, C, H, W)
            if backgrounds is not None:
                masks = torch.cat([torch.ones(1, B, 1, H, W, device=device), masks])
                layers = torch.cat([backgrounds, layers])
                one, zero = torch.ones(B, L, 1, device=device), torch.zeros(B, 1, L + 1, device=device)
                occ_grid = torch.cat([zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1).permute(1, 2, 0)
            return layered_composition(layers, masks, occ_grid)

        else:
            layers = [layers[k][(None,) * (L-1)].transpose(k, L-1) for k in range(L)]  # L elements of size K1.. 1BCHW
            masks = [masks[k][(None,) * (L-1)].transpose(k, L-1) for k in range(L)]  # L elements of size K1...1BCHW
            size = (K,) * L + (B, C, H, W)
            if backgrounds is not None:
                M = backgrounds.size(0)
                backgrounds = backgrounds[(None,) * L].transpose(0, L)  # M1..1BCHW
                layers = [backgrounds] + [layers[k][None] for k in range(L)]
                masks = [torch.ones((1,) * (L + 1) + (B, C, H, W)).to(device)] + [masks[k][None] for k in range(L)]
                one, zero = torch.ones(B, L, 1, device=device), torch.zeros(B, 1, L + 1, device=device)
                occ_grid = torch.cat([zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1).permute(1, 2, 0)
                size = (M,) + size
            else:
                M = 1

            occ_grid = occ_grid[..., None, None, None]
            res = torch.zeros(size, device=device)
            for k in range(len(layers)):
                if backgrounds is not None:
                    j_start = 1 if self.pred_occlusion else k + 1
                else:
                    j_start = 0 if self.pred_occlusion else k + 1
                occ_masks = torch.ones(size, device=device)
                for j in range(j_start, len(layers)):
                    if j != k:
                        occ_masks *= 1 - occ_grid[j, k] * masks[j]
                res += occ_masks * masks[k] * layers[k]
            return res.view(K**L*M, B, C, H, W).transpose(0, 1)

    def criterion(self, inp, target, weights=None, reduction='mean'):
        dist = self._criterion(inp, target)
        if weights is not None:
            dist = dist * weights
        if reduction == 'mean':
            return dist.flatten(2).mean(2)
        elif reduction == 'sum':
            return dist.flatten(2).sum(2)
        elif reduction == 'none':
            return dist
        else:
            raise NotImplementedError

    @torch.no_grad()
    def transform(self, x, with_composition=False, pred_semantic_labels=False, pred_instance_labels=False,
                  with_bkg=True, hard_occ_grid=False):
        B, C, H, W = x.size()
        L, K = self.n_objects, self.n_sprites
        tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob = self.predict(x)
        if class_prob is not None:
            class_oh = torch.zeros(class_prob.shape, device=x.device).scatter_(1, class_prob.argmax(1, keepdim=True), 1)
        else:
            class_oh = None

        if pred_semantic_labels:
            label_layers = torch.arange(1, K+1, device=x.device)[(None,)*4].transpose(0, 4).expand(L, -1, B, 1, H, W)
            true_occ_grid = (occ_grid > 0.5).float()
            target = self.compose(label_layers, (tsf_masks > 0.5).long(), true_occ_grid, class_prob=class_oh).squeeze(1)
            return target.clamp(0, self.n_sprites).long()

        elif pred_instance_labels:
            label_layers = torch.arange(1, L+1, device=x.device)[(None,)*5].transpose(0, 5).expand(-1, K, B, 1, H, W)
            true_occ_grid = (occ_grid > 0.5).float()
            target = self.compose(label_layers, (tsf_masks > 0.5).long(), true_occ_grid, class_prob=class_oh).squeeze(1)
            target = target.clamp(0, L).long()
            if not with_bkg and class_oh is not None:
                bkg_idx = target == 0
                tsf_layers = (tsf_layers * class_oh[..., None, None, None]).sum(axis=1)
                new_target = ((tsf_layers - x)**2).sum(2).argmin(0).long() + 1
                target[bkg_idx] = new_target[bkg_idx]
            return target

        else:
            occ_grid = (occ_grid > 0.5).float() if hard_occ_grid else occ_grid
            tsf_layers, tsf_masks = tsf_layers.clamp(0, 1), tsf_masks.clamp(0, 1)
            if tsf_bkgs is not None:
                tsf_bkgs = tsf_bkgs.clamp(0, 1)
            target = self.compose(tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob)
            if class_prob is not None:
                target = target.unsqueeze(1)

            if with_composition:
                compo = []
                for k in range(L):
                    compo += [tsf_layers[k].transpose(0, 1), tsf_masks[k].transpose(0, 1)]
                if self.learn_backgrounds:
                    compo.insert(2, tsf_bkgs.transpose(0, 1))
                return target, compo
            else:
                return target

    def step(self):
        self.cur_epoch += 1
        [tsf.step() for tsf in self.sprite_transformers]
        if hasattr(self, 'layer_transformer'):
            self.layer_transformer.step()
        if self.learn_backgrounds:
            self.bkg_transformer.step()

    def set_optimizer(self, opt):
        self.optimizer = opt
        [tsf.set_optimizer(opt) for tsf in self.sprite_transformers]
        if hasattr(self, 'layer_transformer'):
            self.layer_transformer.set_optimizer(opt)
        if self.learn_backgrounds:
            self.bkg_transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                if 'activations' in name and state[name].shape != param.shape:
                    state[name].copy_(torch.Tensor([True] * state[name].size(0)).to(param.device))
                else:
                    state[name].copy_(param)
            elif name == 'prototypes':
                state['prototype_params'].copy_(param)
            elif name == 'backgrounds':
                state['bkg_params'].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster or self.are_sprite_frozen:
            return [], 0
        if self.add_empty_sprite:
            proportions = proportions[:-1] / max(proportions[:-1])

        N, threshold = len(proportions), self.empty_cluster_threshold
        reassigned = []
        idx = torch.argmax(proportions).item()
        for i in range(N):
            if proportions[i] < threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)

        return reassigned, idx

    def restart_branch_from(self, i, j):
        self.prototype_params[i].data.copy_(copy_with_noise(self.prototype_params[j], NOISE_SCALE))
        self.mask_params[i].data.copy_(self.mask_params[j].detach().clone())
        [tsf.restart_branch_from(i, j, noise_scale=0) for tsf in self.sprite_transformers]

        if hasattr(self, 'optimizer'):
            opt = self.optimizer
            params = [self.mask_params]
            if isinstance(opt, (Adam,)):
                for param in params:
                    opt.state[param]['exp_avg'][i] = opt.state[param]['exp_avg'][j]
                    opt.state[param]['exp_avg_sq'][i] = opt.state[param]['exp_avg_sq'][j]
            elif isinstance(opt, (RMSprop,)):
                for param in params:
                    opt.state[param]['square_avg'][i] = opt.state[param]['square_avg'][j]
            else:
                raise NotImplementedError('unknown optimizer: you should define how to reinstanciate statistics if any')
