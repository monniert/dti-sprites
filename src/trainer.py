import argparse
import os
import shutil
import seaborn as sns
import time
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
try:
    import visdom
except ModuleNotFoundError:
    pass

from dataset import get_dataset
from model import get_model
from model.tools import count_parameters, safe_model_state_dict
from optimizer import get_optimizer
from scheduler import get_scheduler
from utils import use_seed, coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from utils.image import convert_to_img, save_gif
from utils.logger import get_logger, print_info, print_warning
from utils.metrics import (AverageTensorMeter, AverageMeter, Metrics, Scores, SegmentationScores, InstanceSegScores)
from utils.path import CONFIGS_PATH, RUNS_PATH
from utils.plot import plot_bar, plot_lines


PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], train_metrics: {}".format
PRINT_VAL_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], val_metrics: {}".format
PRINT_CHECK_CLUSTERS_FMT = "Epoch [{}/{}], Iter [{}/{}]: Reassigned clusters {} from cluster {}".format
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}], LR update: lr = {}".format

TRAIN_METRICS_FILE = "train_metrics.tsv"
VAL_METRICS_FILE = "val_metrics.tsv"
VAL_SCORES_FILE = "val_scores.tsv"
FINAL_SCORES_FILE = 'final_scores.tsv'
FINAL_SEG_SCORES_FILE = 'final_seg_scores.tsv'
FINAL_SEMANTIC_SCORES_FILE = 'final_semantic_scores.tsv'
MODEL_FILE = 'model.pkl'

N_TRANSFORMATION_PREDICTIONS = 4
N_CLUSTER_SAMPLES = 5
MAX_GIF_SIZE = 64
VIZ_HEIGHT = 300
VIZ_WIDTH = 500
VIZ_MAX_IMG_SIZE = 64


class Trainer:
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""

    @use_seed()
    def __init__(self, config_path, run_dir):
        self.config_path = coerce_to_path_and_check_exist(config_path)
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.logger = get_logger(self.run_dir, name="trainer")
        self.print_and_log_info("Trainer initialisation: run directory is {}".format(run_dir))

        shutil.copy(self.config_path, self.run_dir)
        self.print_and_log_info("Config {} copied to run directory".format(self.config_path))

        with open(self.config_path) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

        if torch.cuda.is_available():
            type_device = "cuda"
            nb_device = torch.cuda.device_count()
        else:
            type_device = "cpu"
            nb_device = None
        self.device = torch.device(type_device)
        self.print_and_log_info("Using {} device, nb_device is {}".format(type_device, nb_device))

        # Datasets and dataloaders
        self.dataset_kwargs = cfg["dataset"]
        self.dataset_name = self.dataset_kwargs.pop("name")
        train_dataset = get_dataset(self.dataset_name)("train", **self.dataset_kwargs)
        val_dataset = get_dataset(self.dataset_name)("val", **self.dataset_kwargs)
        self.n_classes = train_dataset.n_classes
        self.is_val_empty = len(val_dataset) == 0
        self.print_and_log_info("Dataset {} instantiated with {}".format(self.dataset_name, self.dataset_kwargs))
        self.print_and_log_info("Found {} classes, {} train samples, {} val samples"
                                .format(self.n_classes, len(train_dataset), len(val_dataset)))

        self.img_size = train_dataset.img_size
        self.batch_size = cfg["training"]["batch_size"]
        self.n_workers = cfg["training"].get("n_workers", 4)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       num_workers=self.n_workers, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        self.print_and_log_info("Dataloaders instantiated with batch_size={} and n_workers={}"
                                .format(self.batch_size, self.n_workers))
        self.seg_eval = getattr(train_dataset, 'seg_eval', False)
        self.instance_eval = getattr(train_dataset, 'instance_eval', False)

        self.n_batches = len(self.train_loader)
        self.n_iterations, self.n_epoches = cfg["training"].get("n_iterations"), cfg["training"].get("n_epoches")
        assert not (self.n_iterations is not None and self.n_epoches is not None)
        if self.n_iterations is not None:
            self.n_epoches = max(self.n_iterations // self.n_batches, 1)
        else:
            self.n_iterations = self.n_epoches * len(self.train_loader)

        # Model
        self.model_kwargs = cfg["model"]
        self.model_name = self.model_kwargs.pop("name")
        self.is_gmm = 'gmm' in self.model_name
        self.model = get_model(self.model_name)(self.train_loader.dataset, **self.model_kwargs).to(self.device)
        self.print_and_log_info("Using model {} with kwargs {}".format(self.model_name, self.model_kwargs))
        self.print_and_log_info('Number of trainable parameters: {}'.format(f'{count_parameters(self.model):,}'))
        self.n_prototypes = self.model.n_prototypes
        self.n_backgrounds = getattr(self.model, 'n_backgrounds', 0)
        self.n_objects = max(self.model.n_objects, 1)
        self.pred_class = getattr(self.model, 'pred_class', False) or getattr(self.model, 'estimate_minimum', False)
        if self.pred_class:
            self.n_clusters = self.n_prototypes * self.n_objects
        else:
            self.n_clusters = self.n_prototypes ** self.n_objects * max(self.n_backgrounds, 1)
        self.learn_masks = getattr(self.model, 'learn_masks', False)
        self.learn_backgrounds = getattr(self.model, 'learn_backgrounds', False)

        # Optimizer
        opt_params = cfg["training"]["optimizer"] or {}
        optimizer_name = opt_params.pop("name")
        cluster_kwargs = opt_params.pop('cluster', {})
        tsf_kwargs = opt_params.pop('transformer', {})
        self.optimizer = get_optimizer(optimizer_name)([
            dict(params=self.model.cluster_parameters(), **cluster_kwargs),
            dict(params=self.model.transformer_parameters(), **tsf_kwargs)],
            **opt_params)
        self.model.set_optimizer(self.optimizer)
        self.print_and_log_info("Using optimizer {} with kwargs {}".format(optimizer_name, opt_params))
        self.print_and_log_info("cluster kwargs {}".format(cluster_kwargs))
        self.print_and_log_info("transformer kwargs {}".format(tsf_kwargs))

        # Scheduler
        scheduler_params = cfg["training"].get("scheduler", {}) or {}
        scheduler_name = scheduler_params.pop("name", None)
        self.scheduler_update_range = scheduler_params.pop("update_range", "epoch")
        assert self.scheduler_update_range in ["epoch", "batch"]
        if scheduler_name == "multi_step" and isinstance(scheduler_params["milestones"][0], float):
            n_tot = self.n_epoches if self.scheduler_update_range == "epoch" else self.n_iterations
            scheduler_params["milestones"] = [round(m * n_tot) for m in scheduler_params["milestones"]]
        self.scheduler = get_scheduler(scheduler_name)(self.optimizer, **scheduler_params)
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.print_and_log_info("Using scheduler {} with parameters {}".format(scheduler_name, scheduler_params))

        # Pretrained / Resume
        checkpoint_path = cfg["training"].get("pretrained")
        checkpoint_path_resume = cfg["training"].get("resume")
        assert not(checkpoint_path is not None and checkpoint_path_resume is not None)
        if checkpoint_path is not None:
            self.load_from_tag(checkpoint_path)
        elif checkpoint_path_resume is not None:
            self.load_from_tag(checkpoint_path_resume, resume=True)
        else:
            self.start_epoch, self.start_batch = 1, 1

        # Train metrics
        metric_names = ['time/img', 'loss']
        metric_names += [f'prop_clus{i}' for i in range(self.n_clusters)]
        train_iter_interval = cfg["training"]["train_stat_interval"]
        self.train_stat_interval = train_iter_interval
        self.train_metrics = Metrics(*metric_names)
        self.train_metrics_path = self.run_dir / TRAIN_METRICS_FILE
        if not self.train_metrics_path.exists():
            with open(self.train_metrics_path, mode="w") as f:
                f.write("iteration\tepoch\tbatch\t" + "\t".join(self.train_metrics.names) + "\n")

        # Val metrics & scores
        val_iter_interval = cfg["training"]["val_stat_interval"]
        self.val_stat_interval = val_iter_interval
        self.val_metrics = Metrics('loss_val')
        self.val_metrics_path = self.run_dir / VAL_METRICS_FILE
        if not self.val_metrics_path.exists():
            with open(self.val_metrics_path, mode="w") as f:
                f.write("iteration\tepoch\tbatch\t" + "\t".join(self.val_metrics.names) + "\n")

        self.eval_semantic = cfg["training"].get("eval_semantic", False)
        self.eval_qualitative = cfg["training"].get("eval_qualitative", False)
        self.eval_with_bkg = cfg["training"].get("eval_with_bkg", False)
        if self.seg_eval:
            self.val_scores = SegmentationScores(self.n_classes)
        elif self.instance_eval:
            self.val_scores = InstanceSegScores(self.n_objects + 1, with_bkg=self.eval_with_bkg)
        else:
            self.val_scores = Scores(self.n_classes, self.n_prototypes)
        self.val_scores_path = self.run_dir / VAL_SCORES_FILE
        if not self.val_scores_path.exists():
            with open(self.val_scores_path, mode="w") as f:
                f.write("iteration\tepoch\tbatch\t" + "\t".join(self.val_scores.names) + "\n")

        # Prototypes
        self.check_cluster_interval = cfg["training"]["check_cluster_interval"]
        self.prototypes_path = coerce_to_path_and_create_dir(self.run_dir / 'prototypes')
        [coerce_to_path_and_create_dir(self.prototypes_path / f'proto{k}') for k in range(self.n_prototypes)]

        if self.learn_masks:
            self.masked_prototypes_path = coerce_to_path_and_create_dir(self.run_dir / 'masked_prototypes')
            [coerce_to_path_and_create_dir(self.masked_prototypes_path / f'proto{k}') for k in range(self.n_prototypes)]
            self.masks_path = coerce_to_path_and_create_dir(self.run_dir / 'masks')
            [coerce_to_path_and_create_dir(self.masks_path / f'mask{k}') for k in range(self.n_prototypes)]
        if self.learn_backgrounds:
            self.backgrounds_path = coerce_to_path_and_create_dir(self.run_dir / 'backgrounds')
            [coerce_to_path_and_create_dir(self.backgrounds_path / f'bkg{k}') for k in range(self.n_backgrounds)]

        # Transformation predictions
        self.transformation_path = coerce_to_path_and_create_dir(self.run_dir / 'transformations')
        self.images_to_tsf = next(iter(self.train_loader))[0][:N_TRANSFORMATION_PREDICTIONS].to(self.device)
        for k in range(self.images_to_tsf.size(0)):
            out = coerce_to_path_and_create_dir(self.transformation_path / f'img{k}')
            convert_to_img(self.images_to_tsf[k]).save(out / 'input.png')
            N = self.n_clusters if self.n_clusters <= 40 else 2 * self.n_prototypes
            [coerce_to_path_and_create_dir(out / f'tsf{k}') for k in range(N)]
            if self.learn_masks:
                [coerce_to_path_and_create_dir(out / f'frg_tsf{k}') for k in range(self.n_prototypes)]
                [coerce_to_path_and_create_dir(out / f'mask_tsf{k}') for k in range(self.n_prototypes)]
            if self.learn_backgrounds:
                [coerce_to_path_and_create_dir(out / f'bkg_tsf{k}') for k in range(self.n_backgrounds)]

        # Visdom
        viz_port = cfg["training"].get("visualizer_port")
        if viz_port is not None:
            os.environ["http_proxy"] = ""
            self.visualizer = visdom.Visdom(port=viz_port, env=f'{self.run_dir.parent.name}_{self.run_dir.name}')
            self.visualizer.delete_env(self.visualizer.env)  # Clean env before plotting
            self.print_and_log_info(f"Visualizer initialised at {viz_port}")
        else:
            self.visualizer = None
            self.print_and_log_info("No visualizer initialized")

    def print_and_log_info(self, string):
        print_info(string)
        self.logger.info(string)

    def load_from_tag(self, tag, resume=False):
        self.print_and_log_info("Loading model from run {}".format(tag))
        path = coerce_to_path_and_check_exist(RUNS_PATH / self.dataset_name / tag / MODEL_FILE)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            state = safe_model_state_dict(checkpoint["model_state"])
            self.model.module.load_state_dict(state, dataset=self.train_loader.dataset)
        self.start_epoch, self.start_batch = 1, 1
        if resume:
            self.start_epoch, self.start_batch = checkpoint["epoch"], checkpoint.get("batch", 0) + 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.cur_lr = self.scheduler.get_last_lr()[0]
        if hasattr(self.model, 'cur_epoch'):
            self.model.cur_epoch = checkpoint['epoch']
        self.print_and_log_info("Checkpoint loaded at epoch {}, batch {}".format(self.start_epoch, self.start_batch-1))
        self.print_and_log_info("LR = {}".format(self.cur_lr))

    @property
    def score_name(self):
        return self.val_scores.score_name

    def print_memory_usage(self, prefix):
        usage = {}
        for attr in ["memory_allocated", "max_memory_allocated", "memory_cached", "max_memory_cached"]:
            usage[attr] = getattr(torch.cuda, attr)() * 0.000001
        self.print_and_log_info("{}:\t{}".format(
            prefix, " / ".join(["{}: {:.0f}MiB".format(k, v) for k, v in usage.items()])))

    @use_seed()
    def run(self):
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter
        prev_check_cluster_iter = cur_iter
        if self.start_epoch == self.n_epoches:
            self.print_and_log_info("No training, only evaluating")
            self.save_metric_plots()
            self.evaluate()
            self.print_and_log_info("Training run is over")
            return None

        for epoch in range(self.start_epoch, self.n_epoches + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1
            for batch, (images, labels) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break

                self.single_train_batch_run(images)
                if self.scheduler_update_range == "batch":
                    self.update_scheduler(epoch, batch=batch)

                if (cur_iter - prev_train_stat_iter) >= self.train_stat_interval:
                    prev_train_stat_iter = cur_iter
                    self.log_train_metrics(cur_iter, epoch, batch)

                if (cur_iter - prev_check_cluster_iter) >= self.check_cluster_interval:
                    prev_check_cluster_iter = cur_iter
                    self.check_cluster(cur_iter, epoch, batch)

                if (cur_iter - prev_val_stat_iter) >= self.val_stat_interval:
                    prev_val_stat_iter = cur_iter
                    if not self.is_val_empty:
                        self.run_val()
                        self.log_val_metrics(cur_iter, epoch, batch)
                    self.save(epoch=epoch, batch=batch)
                    self.log_images(cur_iter)

            self.model.step()
            if self.scheduler_update_range == "epoch" and batch_start == 1:
                self.update_scheduler(epoch + 1, batch=1)

        self.save(epoch=epoch, batch=batch)
        self.save_metric_plots()
        self.evaluate()
        self.print_and_log_info("Training run is over")

    def update_scheduler(self, epoch, batch):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            self.print_and_log_info(PRINT_LR_UPD_FMT(epoch, self.n_epoches, batch, self.n_batches, lr))

    def single_train_batch_run(self, images):
        start_time = time.time()
        B = images.size(0)
        self.model.train()
        images = images.to(self.device)

        self.optimizer.zero_grad()
        loss, distances = self.model(images)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            if self.pred_class:
                proportions = (1 - distances).mean(0)
            else:
                argmin_idx = distances.min(1)[1]
                one_hot = torch.zeros(B, distances.size(1), device=self.device).scatter(1, argmin_idx[:, None], 1)
                proportions = one_hot.sum(0) / B

        self.train_metrics.update({
            'time/img': (time.time() - start_time) / B,
            'loss': loss.item(),
        })
        self.train_metrics.update({f'prop_clus{i}': p.item() for i, p in enumerate(proportions)})

    @torch.no_grad()
    def log_images(self, cur_iter):
        self.model.eval()
        self.save_prototypes(cur_iter)
        self.update_visualizer_images(self.model.prototypes, 'prototypes', nrow=5)
        if self.learn_masks:
            self.save_masked_prototypes(cur_iter)
            self.update_visualizer_images(self.model.prototypes * self.model.masks, 'masked_prototypes', nrow=5)
            self.save_masks(cur_iter)
            self.update_visualizer_images(self.model.masks, 'masks', nrow=5)
        if self.learn_backgrounds:
            self.save_backgrounds(cur_iter)
            self.update_visualizer_images(self.model.backgrounds, 'backgrounds', nrow=5)

        # Transformations
        tsf_imgs, compositions = self.save_transformed_images(cur_iter)
        C, H, W = tsf_imgs.shape[2:]
        self.update_visualizer_images(tsf_imgs.reshape(-1, C, H, W), 'transformations', nrow=tsf_imgs.size(1))

        # Compositions
        if len(compositions) > 0:
            k = 0
            for imgs, name in zip(compositions[:2], ['frg_tsf', 'mask_tsf']):
                self.update_visualizer_images(imgs.view(-1, imgs.size(2), H, W), name, nrow=self.n_prototypes+1)
                k += 1
            if self.learn_backgrounds:
                imgs = compositions[k]
                self.update_visualizer_images(imgs.view(-1, imgs.size(2), H, W), 'bkg_tsf',
                                              nrow=self.n_backgrounds+1)
                k += 1
            if self.n_objects > 1:
                for name in ['frg_tsf_aux', 'mask_tsf_aux']:
                    imgs = compositions[k]
                    self.update_visualizer_images(imgs.view(-1, imgs.size(2), H, W), name, nrow=self.n_prototypes+1)
                    k += 1

    @torch.no_grad()
    def save_prototypes(self, cur_iter=None):
        prototypes = self.model.prototypes
        for k in range(self.n_prototypes):
            img = convert_to_img(prototypes[k])
            if cur_iter is not None:
                img.save(self.prototypes_path / f'proto{k}' / f'{cur_iter}.jpg')
            else:
                img.save(self.prototypes_path / f'prototype{k}.png')

    @torch.no_grad()
    def save_masked_prototypes(self, cur_iter=None):
        prototypes = self.model.prototypes
        masks = self.model.masks
        for k in range(self.n_prototypes):
            img = convert_to_img(prototypes[k] * masks[k])
            if cur_iter is not None:
                img.save(self.masked_prototypes_path / f'proto{k}' / f'{cur_iter}.jpg')
            else:
                img.save(self.masked_prototypes_path / f'prototype{k}.png')

    @torch.no_grad()
    def save_masks(self, cur_iter=None):
        masks = self.model.masks
        for k in range(self.n_prototypes):
            img = convert_to_img(masks[k])
            if cur_iter is not None:
                img.save(self.masks_path / f'mask{k}' / f'{cur_iter}.jpg')
            else:
                img.save(self.masks_path / f'mask{k}.png')

    @torch.no_grad()
    def save_backgrounds(self, cur_iter=None):
        backgrounds = self.model.backgrounds
        for k in range(self.n_backgrounds):
            img = convert_to_img(backgrounds[k])
            if cur_iter is not None:
                img.save(self.backgrounds_path / f'bkg{k}' / f'{cur_iter}.jpg')
            else:
                img.save(self.backgrounds_path / f'background{k}.png')

    @torch.no_grad()
    def save_transformed_images(self, cur_iter=None):
        self.model.eval()
        if self.learn_masks:
            output, compositions = self.model.transform(self.images_to_tsf, with_composition=True)
        else:
            output, compositions = self.model.transform(self.images_to_tsf), []

        transformed_imgs = torch.cat([self.images_to_tsf.unsqueeze(1), output], 1)
        N = self.n_clusters if self.n_clusters <= 40 else 2 * self.n_prototypes
        transformed_imgs = transformed_imgs[:, :N+1]
        for k in range(transformed_imgs.size(0)):
            for j, img in enumerate(transformed_imgs[k][1:]):
                if cur_iter is not None:
                    convert_to_img(img).save(self.transformation_path / f'img{k}' / f'tsf{j}' / f'{cur_iter}.jpg')
                else:
                    convert_to_img(img).save(self.transformation_path / f'img{k}' / f'tsf{j}.png')

        i = 0
        for name in ['frg', 'mask', 'bkg', 'frg_aux', 'mask_aux']:
            if name == 'bkg' and not self.learn_backgrounds:
                continue
            if i == len(compositions):
                break

            layer = compositions[i].expand(-1, -1, self.images_to_tsf.size(1), -1, -1)
            compositions[i] = torch.cat([self.images_to_tsf.unsqueeze(1), layer], 1)
            if name in ['frg', 'mask', 'bkg']:
                for k in range(transformed_imgs.size(0)):
                    tmp_path = self.transformation_path / f'img{k}'
                    for j, img in enumerate(compositions[i][k][1:]):
                        if cur_iter is not None:
                            convert_to_img(img).save(tmp_path / f'{name}_tsf{j}' / f'{cur_iter}.jpg')
                        else:
                            convert_to_img(img).save(tmp_path / f'{name}_tsf{j}.png')
            i += 1

        return transformed_imgs, compositions

    @torch.no_grad()
    def update_visualizer_images(self, images, title, nrow):
        if self.visualizer is None:
            return None

        if max(images.shape[1:]) > VIZ_MAX_IMG_SIZE:
            images = F.interpolate(images, size=VIZ_MAX_IMG_SIZE, mode='bilinear', align_corners=False)
        self.visualizer.images(images.clamp(0, 1), win=title, nrow=nrow,
                               opts=dict(title=title, store_history=True, width=VIZ_WIDTH, height=VIZ_HEIGHT))

    def check_cluster(self, cur_iter, epoch, batch):
        if hasattr(self.model, '_diff_selections') and self.visualizer is not None:
            diff = self.model._diff_selections
            x, y = [[cur_iter] * len(diff[0])], [diff[1]]
            self.visualizer.line(y, x, win='diff selection', update='append', opts=dict(title='diff selection',
                                 legend=diff[0], width=VIZ_WIDTH, height=VIZ_HEIGHT))

        proportions = torch.Tensor([self.train_metrics[f'prop_clus{i}'].avg for i in range(self.n_clusters)])
        if self.n_backgrounds > 1:
            proportions = proportions.view(self.n_prototypes, self.n_backgrounds)
            for axis, is_bkg in zip([1, 0], [False, True]):
                prop = proportions.sum(axis)
                reassigned, idx = self.model.reassign_empty_clusters(prop, is_background=is_bkg)
                msg = PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epoches, batch, self.n_batches, reassigned, idx)
                if is_bkg:
                    msg += ' for backgrounds'
                self.print_and_log_info(msg)
                self.print_and_log_info(', '.join(['prop_{}={:.4f}'.format(k, prop[k]) for k in range(len(prop))]))
        elif self.n_objects > 1:
            k = np.random.randint(0, self.n_objects)
            if self.n_clusters == self.n_prototypes ** self.n_objects:
                prop = proportions.view((self.n_prototypes,) * self.n_objects).transpose(0, k).flatten(1).sum(1)
            else:
                prop = proportions.view(self.n_objects, self.n_prototypes)[k]
            reassigned, idx = self.model.reassign_empty_clusters(prop)
            msg = PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epoches, batch, self.n_batches, reassigned, idx)
            msg += f' for object layer {k}'
            self.print_and_log_info(msg)
            self.print_and_log_info(', '.join(['prop_{}={:.4f}'.format(k, prop[k]) for k in range(len(prop))]))
        else:
            reassigned, idx = self.model.reassign_empty_clusters(proportions)
            msg = PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epoches, batch, self.n_batches, reassigned, idx)
            self.print_and_log_info(msg)
        self.train_metrics.reset(*[f'prop_clus{i}' for i in range(self.n_clusters)])

    def log_train_metrics(self, cur_iter, epoch, batch):
        # Print & write metrics to file
        stat = PRINT_TRAIN_STAT_FMT(epoch, self.n_epoches, batch, self.n_batches, self.train_metrics)
        self.print_and_log_info(stat[:1000])
        with open(self.train_metrics_path, mode="a") as f:
            f.write("{}\t{}\t{}\t".format(cur_iter, epoch, batch) +
                    "\t".join(map("{:.6f}".format, self.train_metrics.avg_values)) + "\n")

        self.update_visualizer_metrics(cur_iter, train=True)
        self.train_metrics.reset('time/img', 'loss')

    def update_visualizer_metrics(self, cur_iter, train):
        if self.visualizer is None:
            return None

        split, metrics = ('train', self.train_metrics) if train else ('val', self.val_metrics)
        losses = list(filter(lambda s: s.startswith('loss'), metrics.names))
        y, x = [[metrics[n].avg for n in losses]], [[cur_iter] * len(losses)]
        self.visualizer.line(y, x, win=f'{split}_losses', update='append',
                             opts=dict(title=f'{split}_losses', legend=losses, width=VIZ_WIDTH, height=VIZ_HEIGHT))

        if train:
            if self.n_prototypes > 1:
                # Cluster proportions
                N = self.n_clusters if self.n_clusters <= 40 else 2 * self.n_prototypes
                proportions = [metrics[f'prop_clus{i}'].avg for i in range(N)]
                self.visualizer.bar(proportions, win='train_cluster_prop',
                                    opts=dict(title='train_cluster_proportions', width=VIZ_HEIGHT, height=VIZ_HEIGHT))
        else:
            names = list(filter(lambda name: 'cls' not in name, self.val_scores.names))
            y, x = [[self.val_scores[n] for n in names]], [[cur_iter] * len(names)]
            self.visualizer.line(y, x, win='global_scores', update='append',
                                 opts=dict(title='global_scores', legend=names, width=VIZ_WIDTH, height=VIZ_HEIGHT))

            if not self.instance_eval:
                name = 'acc' if not self.seg_eval else 'iou'
                N = self.n_classes
                y = [[self.val_scores[f'{name}_cls{i}'] for i in range(N)]]
                x = [[cur_iter] * N]
                self.visualizer.line(y, x, win=f'{name}_by_cls', update='append', opts=dict(title=f'{name}_by_cls',
                                     legend=[f'cls{i}' for i in range(N)], width=VIZ_WIDTH,
                                     heigh=VIZ_HEIGHT))

    @torch.no_grad()
    def run_val(self):
        self.model.eval()
        for images, labels in self.val_loader:
            B, C, H, W = images.shape
            images = images.to(self.device)
            loss_val, distances = self.model(images)

            if not self.pred_class:
                if self.n_backgrounds > 1:
                    distances, bkg_idx = distances.view(B, self.n_prototypes, self.n_backgrounds).min(2)
                if self.n_objects > 1:
                    distances = distances.view(B, *(self.n_prototypes,)*self.n_objects)
                    other_idxs = []
                    for k in range(self.n_objects, 1, -1):
                        distances, idx = distances.min(k)
                        other_idxs.insert(0, idx)
                dist_min_by_sample, argmin_idx = distances.min(1)

            self.val_metrics.update({'loss_val': loss_val.item()})
            if self.seg_eval:
                if self.n_objects == 1:
                    masks = self.model.transform(images, with_composition=True)[1][1]
                    masks = masks[torch.arange(B), argmin_idx]
                    self.val_scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
                else:
                    target = self.model.transform(images, pred_semantic_labels=True).cpu()
                    if not self.pred_class:
                        target = target.view(B, *(self.n_prototypes,)*self.n_objects, H, W)
                        real_idxs = []
                        for idx in [argmin_idx] + other_idxs:
                            for i in real_idxs:
                                idx = idx[torch.arange(B), i]
                            real_idxs.insert(0, idx)
                            target = target[torch.arange(B), idx]
                    self.val_scores.update(labels.long().numpy(), target.long().cpu().numpy())

            elif self.instance_eval:
                if self.n_objects == 1:
                    masks = self.model.transform(images, with_composition=True)[1][1]
                    self.val_scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
                else:
                    target = self.model.transform(images, pred_instance_labels=True, with_bkg=self.eval_with_bkg).cpu()
                    if not self.pred_class:
                        target = target.view(B, *(self.n_prototypes,)*self.n_objects, images.size(2), images.size(3))
                        real_idxs = []
                        for idx in [argmin_idx] + other_idxs:
                            for i in real_idxs:
                                idx = idx[torch.arange(B), i]
                            real_idxs.insert(0, idx)
                            target = target[torch.arange(B), idx]
                        if not self.eval_with_bkg:
                            bkg_idx = target == 0
                            tsf_layers = self.model.predict(images)[0]
                            new_target = ((tsf_layers - images)**2).sum(3).min(1)[0].argmin(0).long() + 1
                            target[bkg_idx] = new_target[bkg_idx]

                    self.val_scores.update(labels.long().numpy(), target.long().numpy())

            else:
                assert self.n_objects == 1
                self.val_scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

    def log_val_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_VAL_STAT_FMT(epoch, self.n_epoches, batch, self.n_batches, self.val_metrics)
        self.print_and_log_info(stat)
        with open(self.val_metrics_path, mode="a") as f:
            f.write("{}\t{}\t{}\t".format(cur_iter, epoch, batch) +
                    "\t".join(map("{:.6f}".format, self.val_metrics.avg_values)) + "\n")

        scores = self.val_scores.compute()
        self.print_and_log_info("val_scores: " + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()]))
        with open(self.val_scores_path, mode="a") as f:
            f.write("{}\t{}\t{}\t".format(cur_iter, epoch, batch) +
                    "\t".join(map("{:.6f}".format, scores.values())) + "\n")

        self.update_visualizer_metrics(cur_iter, train=False)
        self.val_scores.reset()
        self.val_metrics.reset()

    def save(self, epoch, batch):
        state = {
            "epoch": epoch,
            "batch": batch,
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "model_state": self.model.state_dict(),
            "n_prototypes": self.n_prototypes,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log_info("Model saved at {}".format(save_path))

    def save_metric_plots(self):
        self.model.eval()
        # Prototypes & transformation predictions
        size = MAX_GIF_SIZE if MAX_GIF_SIZE < max(self.img_size) else self.img_size
        self.save_prototypes()
        if self.learn_masks:
            self.save_masked_prototypes()
            self.save_masks()
        if self.learn_backgrounds:
            self.save_backgrounds()
        self.save_transformed_images()

        # Train metrics
        df_train = pd.read_csv(self.train_metrics_path, sep="\t", index_col=0)
        df_val = pd.read_csv(self.val_metrics_path, sep="\t", index_col=0)
        df_scores = pd.read_csv(self.val_scores_path, sep="\t", index_col=0)
        if len(df_train) == 0:
            self.print_and_log_info("No metrics or plots to save")
            return

        # Losses
        losses = list(filter(lambda s: s.startswith('loss'), self.train_metrics.names))
        df = df_train.join(df_val[['loss_val']], how="outer")
        fig = plot_lines(df, losses + ['loss_val'], title="Loss")
        fig.savefig(self.run_dir / "loss.pdf")

        # Cluster proportions
        N = self.n_clusters if self.n_clusters <= 40 else 2 * self.n_prototypes
        names = list(filter(lambda s: s.startswith('prop_'), self.train_metrics.names))[:N]
        fig = plot_lines(df, names, title="Cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions.pdf")
        s = df[names].iloc[-1]
        s.index = list(map(lambda n: n.replace('prop_clus', ''), names))
        fig = plot_bar(s, title="Final cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions_final.pdf")

        # Validation
        if not self.is_val_empty:
            names = list(filter(lambda name: 'cls' not in name, self.val_scores.names))
            fig = plot_lines(df_scores, names, title="Global scores", unit_yaxis=True)
            fig.savefig(self.run_dir / 'global_scores.pdf')

            if not self.instance_eval:
                name = 'acc' if not self.seg_eval else 'iou'
                N = self.n_classes
                fig = plot_lines(df_scores, [f'{name}_cls{i}' for i in range(N)],
                                 title="Scores by cls", unit_yaxis=True)
                fig.savefig(self.run_dir / "scores_by_cls.pdf")

        # Save gifs for prototypes
        for k in range(self.n_prototypes):
            save_gif(self.prototypes_path / f'proto{k}', f'prototype{k}.gif', size=size)
            shutil.rmtree(str(self.prototypes_path / f'proto{k}'))
            if self.learn_masks:
                save_gif(self.masked_prototypes_path / f'proto{k}', f'prototype{k}.gif', size=size)
                shutil.rmtree(str(self.masked_prototypes_path / f'proto{k}'))
                save_gif(self.masks_path / f'mask{k}', f'mask{k}.gif', size=size)
                shutil.rmtree(str(self.masks_path / f'mask{k}'))

        for k in range(self.n_backgrounds):
            save_gif(self.backgrounds_path / f'bkg{k}', f'background{k}.gif', size=size)
            shutil.rmtree(str(self.backgrounds_path / f'bkg{k}'))

        # Save gifs for transformation predictions
        for i in range(self.images_to_tsf.size(0)):
            N = self.n_clusters if self.n_clusters <= 40 else 2 * self.n_prototypes
            for k in range(N):
                save_gif(self.transformation_path / f'img{i}' / f'tsf{k}', f'tsf{k}.gif', size=size)
                shutil.rmtree(str(self.transformation_path / f'img{i}' / f'tsf{k}'))

            if self.learn_masks:
                for k in range(self.n_prototypes):
                    save_gif(self.transformation_path / f'img{i}' / f'frg_tsf{k}', f'frg_tsf{k}.gif', size=size)
                    save_gif(self.transformation_path / f'img{i}' / f'mask_tsf{k}', f'mask_tsf{k}.gif', size=size)
                    shutil.rmtree(str(self.transformation_path / f'img{i}' / f'frg_tsf{k}'))
                    shutil.rmtree(str(self.transformation_path / f'img{i}' / f'mask_tsf{k}'))
            if self.learn_backgrounds:
                for k in range(self.n_backgrounds):
                    save_gif(self.transformation_path / f'img{i}' / f'bkg_tsf{k}', f'bkg_tsf{k}.gif', size=size)
                    shutil.rmtree(str(self.transformation_path / f'img{i}' / f'bkg_tsf{k}'))

        self.print_and_log_info("Metrics and plots saved")

    def evaluate(self):
        self.model.eval()
        label = self.train_loader.dataset[0][1]
        empty_label = isinstance(label, (int, np.integer)) and label == -1
        if empty_label:
            self.qualitative_eval()
        elif self.seg_eval or self.instance_eval:
            if (self.seg_eval and self.learn_masks) or self.eval_semantic:
                self.segmentation_quantitative_eval()
                self.segmentation_qualitative_eval()
            if self.instance_eval and self.learn_masks:
                self.instance_seg_quantitative_eval()
                self.instance_seg_qualitative_eval()
        else:
            self.quantitative_eval()
            if self.eval_qualitative:
                self.segmentation_qualitative_eval()

        self.print_and_log_info("Evaluation is over")

    @torch.no_grad()
    def qualitative_eval(self):
        """Routine to save qualitative results"""
        if self.n_objects > 1:
            self.segmentation_qualitative_eval()
            return None

        cluster_path = coerce_to_path_and_create_dir(self.run_dir / 'clusters')
        dataset = self.train_loader.dataset
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)

        # Compute results
        distances, cluster_idx = np.array([]), np.array([], dtype=np.int32)
        averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
        for images, labels in train_loader:
            images = images.to(self.device)
            dist = self.model(images)[1]
            if self.n_backgrounds > 1:
                dist = dist.view(images.size(0), self.n_prototypes, self.n_backgrounds).min(2)[0]
            dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), dist.min(1))
            argmin_idx = argmin_idx.astype(np.int32)
            distances = np.hstack([distances, dist_min_by_sample])
            cluster_idx = np.hstack([cluster_idx, argmin_idx])

            transformed_imgs = self.model.transform(images).cpu()
            for k in range(self.n_prototypes):
                imgs = transformed_imgs[argmin_idx == k, k]
                averages[k].update(imgs)

        # Save results
        with open(cluster_path / 'cluster_counts.tsv', mode='w') as f:
            f.write('\t'.join([str(k) for k in range(self.n_prototypes)]) + '\n')
            f.write('\t'.join([str(averages[k].count) for k in range(self.n_prototypes)]) + '\n')
        for k in range(self.n_prototypes):
            path = coerce_to_path_and_create_dir(cluster_path / f'cluster{k}')
            indices = np.where(cluster_idx == k)[0]
            top_idx = np.argsort(distances[indices])[:N_CLUSTER_SAMPLES]
            for j, idx in enumerate(top_idx):
                inp = dataset[indices[idx]][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(path / f'top{j}_raw.png')
                convert_to_img(self.model.transform(inp)[0, k]).save(path / f'top{j}_tsf.png')
                # convert_to_img(self.model.transform(inp, inverse=True)[0, k]).save(path / f'top{j}_tsf_inp.png')
            if len(indices) <= N_CLUSTER_SAMPLES:
                random_idx = indices
            else:
                random_idx = np.random.choice(indices, N_CLUSTER_SAMPLES, replace=False)
            for j, idx in enumerate(random_idx):
                inp = dataset[idx][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(path / f'random{j}_raw.png')
                convert_to_img(self.model.transform(inp)[0, k]).save(path / f'random{j}_tsf.png')
                # convert_to_img(self.model.transform(inp, inverse=True)[0, k]).save(path / f'random{j}_tsf_inp.png'
            try:
                convert_to_img(averages[k].avg).save(path / 'avg.png')
            except AssertionError:
                print_warning(f'no image found in cluster {k}')

    @torch.no_grad()
    def segmentation_quantitative_eval(self):
        """Run and save evaluation for semantic segmentation"""
        dataset = get_dataset(self.dataset_name)("train", eval_mode=True, eval_semantic=True, **self.dataset_kwargs)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SEMANTIC_SCORES_FILE
        scores = SegmentationScores(self.n_classes)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        for images, labels in train_loader:
            images = images.to(self.device)
            loss_val, distances = self.model(images)
            B, C, H, W = images.shape
            if self.n_objects == 1:
                masks = self.model.transform(images, with_composition=True)[1][1]
                if masks.size(1) > 1:
                    argmin_idx = self.model(images)[1].min(1)[1]
                    masks = masks[torch.arange(B), argmin_idx]
                scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
            else:
                if self.pred_class:
                    target = self.model.transform(images, pred_semantic_labels=True).cpu()
                    scores.update(labels.long().numpy(), target.long().numpy())
                else:
                    distances = self.model(images)[1].view(B, *(self.n_prototypes,)*self.n_objects)
                    other_idxs = []
                    for k in range(self.n_objects, 1, -1):
                        distances, idx = distances.min(k)
                        other_idxs.insert(0, idx)
                    dist_min_by_sample, argmin_idx = distances.min(1)

                    target = self.model.transform(images, pred_semantic_labels=True).cpu()
                    target = target.view(B, *(self.n_prototypes,)*self.n_objects, H, W)
                    real_idxs = []
                    for idx in [argmin_idx] + other_idxs:
                        for i in real_idxs:
                            idx = idx[torch.arange(B), i]
                        real_idxs.insert(0, idx)
                        target = target[torch.arange(B), idx]
                    scores.update(labels.long().numpy(), target.long().cpu().numpy())

            loss.update(loss_val.item(), n=images.size(0))

        scores = scores.compute()
        self.print_and_log_info("final_loss: {:.4f}".format(loss.avg))
        self.print_and_log_info("final_scores: " + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()]))
        with open(scores_path, mode="a") as f:
            f.write("{:.6}\t".format(loss.avg) + "\t".join(map("{:.6f}".format, scores.values())) + "\n")

    @torch.no_grad()
    def segmentation_qualitative_eval(self):
        """Run and save qualitative evaluation for semantic segmentation"""
        out = coerce_to_path_and_create_dir(self.run_dir / 'semantic_seg')
        K = self.n_prototypes if self.model.add_empty_sprite else self.n_prototypes + 1
        colors = sns.color_palette('hls', K)
        colors[0] = tuple((np.asarray(colors[0]) / colors[0][0]) * 0.5)
        dataset = self.train_loader.dataset
        if 32 % self.batch_size == 0:
            N, B = 32 // self.batch_size, self.batch_size
        else:
            N, B = 8, 4
        C, H, W = dataset[0][0].shape
        train_loader = DataLoader(dataset, batch_size=B, num_workers=self.n_workers, shuffle=False)

        iterator = iter(train_loader)
        for j in range(N):
            images, labels = iterator.next()
            images = images.to(self.device)
            if self.pred_class:
                recons = self.model.transform(images, hard_occ_grid=True).cpu()
                infer_seg = self.model.transform(images, pred_semantic_labels=True).cpu()
            else:
                distances = self.model(images)[1].view(B, *(self.n_prototypes,)*self.n_objects)
                other_idxs = []
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)
                dist_min_by_sample, argmin_idx = distances.min(1)

                recons = self.model.transform(images).cpu()
                recons = recons.view(B, *(self.n_prototypes,)*self.n_objects, C, H, W)
                infer_seg = self.model.transform(images, pred_semantic_labels=True).cpu()
                infer_seg = infer_seg.view(B, *(self.n_prototypes,)*self.n_objects, H, W)
                real_idxs = []
                for idx in [argmin_idx] + other_idxs:
                    for i in real_idxs:
                        idx = idx[torch.arange(B), i]
                    real_idxs.insert(0, idx)
                    infer_seg = infer_seg[torch.arange(B), idx]
                    recons = recons[torch.arange(B), idx]

            infer_seg = infer_seg.unsqueeze(1).expand(-1, C, H, W)
            color_seg = torch.zeros(infer_seg.shape).float()
            masks = []
            for k, col in enumerate(colors):
                masks.append(infer_seg == k)
                color_seg[masks[-1]] = torch.Tensor(col)[None, :, None, None].to('cpu').expand(B, C, H, W)[masks[-1]]

            images = images.cpu()
            for k in range(B):
                name = f'{k+j*B}'.zfill(2)
                convert_to_img(images[k]).save(out / f'{name}.png')
                convert_to_img(recons[k]).save(out / f'{name}_recons.png')
                convert_to_img(color_seg[k]).save(out / f'{name}_seg_full.png')

    @torch.no_grad()
    def instance_seg_quantitative_eval(self):
        """Run and save quantitative evaluation for instance segmentation"""
        dataset = get_dataset(self.dataset_name)("train", eval_mode=True, **self.dataset_kwargs)
        if 320 % self.batch_size == 0:
            N, B = 320 // self.batch_size, self.batch_size
        else:
            N, B = 80, 4
        train_loader = DataLoader(dataset, batch_size=B, num_workers=self.n_workers, shuffle=False)
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SEG_SCORES_FILE
        scores = InstanceSegScores(self.n_objects + 1, with_bkg=self.eval_with_bkg)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        iterator = iter(train_loader)
        for k in range(N):
            images, labels = iterator.next()
            images = images.to(self.device)
            loss_val, distances = self.model(images)
            if self.n_objects == 1:
                masks = self.model.transform(images, with_composition=True)[1][1]
                scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
            else:
                if self.pred_class:
                    target = self.model.transform(images, pred_instance_labels=True, with_bkg=self.eval_with_bkg).cpu()
                    scores.update(labels.long().numpy(), target.long().numpy())

                else:
                    distances = distances.view(B, *(self.n_prototypes,)*self.n_objects)
                    other_idxs = []
                    for k in range(self.n_objects, 1, -1):
                        distances, idx = distances.min(k)
                        other_idxs.insert(0, idx)
                    dist_min_by_sample, argmin_idx = distances.min(1)

                    target = self.model.transform(images, pred_instance_labels=True, with_bkg=self.eval_with_bkg).cpu()
                    target = target.view(B, *(self.n_prototypes,)*self.n_objects, images.size(2), images.size(3))
                    real_idxs = []
                    for idx in [argmin_idx] + other_idxs:
                        for i in real_idxs:
                            idx = idx[torch.arange(B), i]
                        real_idxs.insert(0, idx)
                        target = target[torch.arange(B), idx]
                    if not self.eval_with_bkg:
                        bkg_idx = target == 0
                        tsf_layers = self.model.predict(images)[0]
                        new_target = (((tsf_layers - images)**2).sum(3).min(1)[0].argmin(0).long() + 1).cpu()
                        target[bkg_idx] = new_target[bkg_idx]
                    scores.update(labels.long().numpy(), target.long().numpy())

            loss.update(loss_val.item(), n=images.size(0))

        scores = scores.compute()
        self.print_and_log_info("final_loss: {:.4f}".format(loss.avg))
        self.print_and_log_info("final_scores: " + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()]))
        with open(scores_path, mode="a") as f:
            f.write("{:.6}\t".format(loss.avg) + "\t".join(map("{:.6f}".format, scores.values())) + "\n")

    @torch.no_grad()
    def instance_seg_qualitative_eval(self):
        """Run and save qualitative evaluation for instance segmentation"""
        out = coerce_to_path_and_create_dir(self.run_dir / 'instance_seg')
        colors = sns.color_palette('tab10', self.n_objects+1)
        dataset = self.train_loader.dataset
        if 32 % self.batch_size == 0:
            N, B = 32 // self.batch_size, self.batch_size
        else:
            N, B = 8, 4
        C, H, W = dataset[0][0].shape
        train_loader = DataLoader(dataset, batch_size=B, num_workers=self.n_workers, shuffle=False)

        iterator = iter(train_loader)
        for j in range(N):
            images, labels = iterator.next()
            images = images.to(self.device)
            if self.pred_class:
                recons = self.model.transform(images, hard_occ_grid=True).cpu()
                infer_seg = self.model.transform(images, pred_instance_labels=True, with_bkg=self.eval_with_bkg).cpu()
            else:
                distances = self.model(images)[1].view(B, *(self.n_prototypes,)*self.n_objects)
                other_idxs = []
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)
                dist_min_by_sample, argmin_idx = distances.min(1)

                recons = self.model.transform(images).cpu()
                recons = recons.view(B, *(self.n_prototypes,)*self.n_objects, C, H, W)
                infer_seg = self.model.transform(images, pred_instance_labels=True, with_bkg=self.eval_with_bkg).cpu()
                infer_seg = infer_seg.view(B, *(self.n_prototypes,)*self.n_objects, H, W)
                real_idxs = []
                for idx in [argmin_idx] + other_idxs:
                    for i in real_idxs:
                        idx = idx[torch.arange(B), i]
                    real_idxs.insert(0, idx)
                    infer_seg = infer_seg[torch.arange(B), idx]
                    recons = recons[torch.arange(B), idx]

                if not self.eval_with_bkg:
                    bkg_idx = infer_seg == 0
                    tsf_layers = self.model.predict(images)[0]
                    new_target = (((tsf_layers - images)**2).sum(3).min(1)[0].argmin(0).long() + 1).cpu()
                    infer_seg[bkg_idx] = new_target[bkg_idx]

            infer_seg = infer_seg.unsqueeze(1).expand(-1, C, H, W)
            color_seg = torch.zeros(infer_seg.shape).float()
            masks = []
            for k, col in enumerate(colors):
                masks.append(infer_seg == k)
                color_seg[masks[-1]] = torch.Tensor(col)[None, :, None, None].to('cpu').expand(B, C, H, W)[masks[-1]]

            images = images.cpu()
            for k in range(B):
                name = f'{k+j*B}'.zfill(2)
                convert_to_img(images[k]).save(out / f'{name}.png')
                convert_to_img(recons[k]).save(out / f'{name}_recons.png')
                convert_to_img(color_seg[k]).save(out / f'{name}_seg_full.png')
                for l in range(self.n_objects + 1):
                    convert_to_img((images[k] * masks[l][k])).save(out / f'{name}_seg_obj{l}.png')

    @torch.no_grad()
    def quantitative_eval(self):
        """Routine to save quantitative results: loss + scores"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        scores = Scores(self.n_classes, self.n_prototypes)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        dataset = get_dataset(self.dataset_name)("train", eval_mode=True, **self.dataset_kwargs)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        for images, labels in loader:
            B = images.size(0)
            images = images.to(self.device)
            distances = self.model(images)[1]
            if self.n_backgrounds > 1:
                distances, bkg_idx = distances.view(B, self.n_prototypes, self.n_backgrounds).min(2)
            if self.n_objects > 1:
                distances = distances.view(B, *(self.n_prototypes,)*self.n_objects)
                other_idxs = []
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)
            dist_min_by_sample, argmin_idx = distances.min(1)

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            assert self.n_objects == 1
            scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

        scores = scores.compute()
        self.print_and_log_info("final_loss: {:.4f}".format(loss.avg))
        self.print_and_log_info("final_scores: " + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()]))
        with open(scores_path, mode="a") as f:
            f.write("{:.6}\t".format(loss.avg) + "\t".join(map("{:.6f}".format, scores.values())) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, required=True, help="Run tag of the experiment")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    seed = cfg["training"].get("seed", 4321)
    dataset = cfg["dataset"]["name"]

    run_dir = RUNS_PATH / dataset / args.tag

    trainer = Trainer(config, run_dir, seed=seed)
    trainer.run(seed=seed)
