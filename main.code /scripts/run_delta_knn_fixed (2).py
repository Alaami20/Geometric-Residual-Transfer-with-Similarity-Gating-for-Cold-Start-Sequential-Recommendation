import logging
import os
import sys
from typing import Tuple, Optional

sys.path.append("..")

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from clearml import Task
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from source.dataset import SequentialDataModule, load_data
from source.embedding_manager import EmbeddingManager
from source.optimizer import ConstrainedNormAdam
from source.recommender import SASRecModel

from source.winter.evaluation import ColdStartEvaluationPipeline
from source.winter.recommender import ColdStartSequentialRecommender, SASRecModelWithTrainableDelta


# -------------------------
# Helpers: safe config reads
# -------------------------
def cfg_select(cfg: DictConfig, key: str, default):
    """Safe OmegaConf.select with fallback default."""
    try:
        v = OmegaConf.select(cfg, key, default=default)
        return default if v is None else v
    except Exception:
        return default


def get_task_name(config: DictConfig) -> str:
    task_name = "init({})".format("text" if config.use_pretrained_item_embeddings else "rand")

    if config.train_delta:
        task_name += f"-delta-{config.max_delta_norm}"

    # (Optional) add knn tag without breaking old behavior
    if cfg_select(config, "cold_knn.enabled", False) and config.train_delta:
        k = cfg_select(config, "cold_knn.k", 50)
        task_name += f"-coldknn-k{k}"

        gate_on = cfg_select(config, "cold_knn.gate.enabled", True)
        if gate_on:
            gtype = cfg_select(config, "cold_knn.gate.type", "maxsim_sigmoid")
            task_name += f"-gate-{gtype}"

    return task_name


def get_task(config: DictConfig) -> Task:
    task = Task.init(
        project_name=config.project_name,
        task_name=get_task_name(config),
        reuse_last_task_id=False,
    )
    task.connect(OmegaConf.to_container(config))
    return task


def get_datamodule(config: DictConfig) -> SequentialDataModule:
    return SequentialDataModule(
        train_filepath=config.dataset.train_filepath,
        val_filepath=config.dataset.val_filepath,
        max_length=config.dataset.max_length,
    )


def get_model(config: DictConfig) -> SASRecModel:
    model_params = dict(
        num_items=config.model.num_items,
        embedding_dim=config.model.embedding_dim,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        intermediate_dim=config.model.embedding_dim,
        p=config.model.p,
        max_length=config.model.max_length,
    )

    if config.train_delta:
        return SASRecModelWithTrainableDelta(max_delta_norm=config.max_delta_norm, **model_params)
    else:
        return SASRecModel(**model_params)


def get_trainer(config: DictConfig) -> Trainer:
    early_stopping = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        mode=config.early_stopping.mode,
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint_dir, Task.current_task().id, "recommender"),
        monitor=config.model_checkpoint.monitor,
        mode=config.model_checkpoint.mode,
    )
    trainer = Trainer(
        devices=config.trainer.devices,
        callbacks=[early_stopping, model_checkpoint],
        max_epochs=config.trainer.max_epochs,
        val_check_interval=config.trainer.val_check_interval,
    )
    return trainer


def get_item_embeddings(config: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    warm_item_embeddings = np.load(config.dataset.item_embeddings.warm)
    cold_item_embeddings = np.load(config.dataset.item_embeddings.cold)

    embedding_manager = EmbeddingManager(
        config.model.embedding_dim,
        reduce=config.model.embedding_dim != warm_item_embeddings.shape[1],
        normalize=True,
    )
    logging.info(f"EmbeddingManager: reduce = {embedding_manager.reduce}.")

    warm_item_embeddings = embedding_manager.fit_transform(warm_item_embeddings)
    cold_item_embeddings = embedding_manager.transform(cold_item_embeddings)

    embedding_manager.save(os.path.join(config.checkpoint_dir, Task.current_task().id, "embedding_manager.pkl"))

    return torch.tensor(warm_item_embeddings).float(), torch.tensor(cold_item_embeddings).float()


# ------------------------------------------------------
# KNN transfer with Gate: cold_delta from warm_delta
# ------------------------------------------------------
@torch.no_grad()
def compute_cold_delta_knn_with_gate(
    warm_item_embeddings: torch.Tensor,   # [Nw(+pad), D]
    cold_item_embeddings: torch.Tensor,   # [Nc, D]
    warm_delta_embeddings: torch.Tensor,  # [Nw(+pad), D]
    k: int,
    temperature: float,
    alpha: float,
    max_delta_norm: float,
    gate_enabled: bool = True,
    gate_type: str = "maxsim_sigmoid",
    gate_tau: float = 0.25,
    gate_beta: float = 20.0,
    hard_min_sim: Optional[float] = 0.05,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Transfers trainable warm deltas to cold items using KNN over warm item embeddings,
    with a confidence gate. Output is clipped to max_delta_norm per cold item.

    Backward-compatible: can be skipped entirely by caller.
    """

    device = warm_item_embeddings.device
    cold_item_embeddings = cold_item_embeddings.to(device)
    warm_delta_embeddings = warm_delta_embeddings.to(device)

    # Exclude padding row (idx=0) from neighbor candidates to avoid garbage neighbors.
    if warm_item_embeddings.shape[0] > 1:
        warm_e = warm_item_embeddings[1:]
        warm_d = warm_delta_embeddings[1:]
    else:
        warm_e = warm_item_embeddings
        warm_d = warm_delta_embeddings

    # Normalize for cosine sim (EmbManager already normalizes, but keep safe)
    warm_e = F.normalize(warm_e, dim=1)
    cold_e = F.normalize(cold_item_embeddings, dim=1)

    n_warm = warm_e.shape[0]
    k_eff = max(1, min(int(k), int(n_warm)))

    if k_eff != k:
        logging.info(f"cold_knn: adjusted k from {k} to {k_eff} (n_warm={n_warm}).")

    out = torch.empty((cold_e.shape[0], warm_d.shape[1]), device=device, dtype=warm_d.dtype)

    logk = float(np.log(k_eff)) if k_eff > 1 else 1.0

    for start in range(0, cold_e.shape[0], chunk_size):
        end = min(start + chunk_size, cold_e.shape[0])

        # [B, D] @ [D, Nw] -> [B, Nw]
        sim = cold_e[start:end] @ warm_e.T

        topv, topi = torch.topk(sim, k_eff, dim=1, largest=True, sorted=True)  # [B, k]
        w = torch.softmax(topv / float(temperature), dim=1)                    # [B, k]

        # gather deltas: [B, k, D]
        neigh_delta = warm_d[topi]  # topi indexes warm_e (without padding)

        # weighted sum: [B, D]
        transfer = torch.sum(w.unsqueeze(-1) * neigh_delta, dim=1)

        # Gate
        if gate_enabled:
            max_sim = topv[:, 0]  # [B], sorted=True so first is max

            if gate_type == "maxsim_sigmoid":
                g = torch.sigmoid(float(gate_beta) * (max_sim - float(gate_tau)))
            elif gate_type == "entropy_sigmoid":
                # normalized entropy in [0,1], confidence = 1 - entropy
                ent = -(w * (w.clamp_min(1e-12).log())).sum(dim=1)  # [B]
                ent_norm = ent / float(logk)
                conf = 1.0 - ent_norm
                g = torch.sigmoid(float(gate_beta) * (conf - float(gate_tau)))
            else:
                raise ValueError(f"Unknown gate_type: {gate_type}")

            if hard_min_sim is not None:
                g = g * (max_sim >= float(hard_min_sim)).to(g.dtype)

            transfer = transfer * g.unsqueeze(1)

        # Alpha scaling
        cold_delta = float(alpha) * transfer

        # Clip to max_delta_norm (paper-style constraint)
        norms = cold_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(float(max_delta_norm) / norms, max=1.0)
        cold_delta = cold_delta * scale

        out[start:end] = cold_delta

    return out


def add_cold_item_embeddings(model: SASRecModel, cold_item_embeddings: torch.Tensor, config: DictConfig) -> None:
    """
    Original behavior preserved:
      - If NOT trainable-delta model: just append cold item embeddings
      - If trainable-delta model and cold_knn disabled: append cold with zero delta (baseline)
    New behavior:
      - If trainable-delta model and cold_knn enabled: append cold with KNN-transferred delta + gate
    """
    item_embeddings = model.item_embedding.weight[: model.num_items + 1]

    if isinstance(model, SASRecModelWithTrainableDelta):
        delta_embeddings = model.delta_embedding.weight[: model.num_items + 1]

        cold_knn_enabled = bool(cfg_select(config, "cold_knn.enabled", False))
        if cold_knn_enabled:
            k = int(cfg_select(config, "cold_knn.k", 50))
            temperature = float(cfg_select(config, "cold_knn.temperature", 0.07))
            alpha = float(cfg_select(config, "cold_knn.alpha", 1.0))
            chunk_size = int(cfg_select(config, "cold_knn.chunk_size", 4096))

            gate_enabled = bool(cfg_select(config, "cold_knn.gate.enabled", True))
            gate_type = str(cfg_select(config, "cold_knn.gate.type", "maxsim_sigmoid"))
            gate_tau = float(cfg_select(config, "cold_knn.gate.tau", 0.25))
            gate_beta = float(cfg_select(config, "cold_knn.gate.beta", 20.0))
            hard_min_sim = cfg_select(config, "cold_knn.gate.hard_min_sim", 0.05)
            hard_min_sim = None if hard_min_sim is None else float(hard_min_sim)

            logging.info(
                f"cold_knn: enabled=True k={k} temp={temperature} alpha={alpha} "
                f"gate={gate_enabled} type={gate_type} tau={gate_tau} beta={gate_beta} hard_min_sim={hard_min_sim}"
            )

            cold_delta = compute_cold_delta_knn_with_gate(
                warm_item_embeddings=item_embeddings,
                cold_item_embeddings=cold_item_embeddings.to(item_embeddings.device),
                warm_delta_embeddings=delta_embeddings,
                k=k,
                temperature=temperature,
                alpha=alpha,
                max_delta_norm=float(config.max_delta_norm),
                gate_enabled=gate_enabled,
                gate_type=gate_type,
                gate_tau=gate_tau,
                gate_beta=gate_beta,
                hard_min_sim=hard_min_sim,
                chunk_size=chunk_size,
            )
        else:
            # BASELINE: cold deltas = 0
            cold_delta = torch.zeros_like(cold_item_embeddings).to(delta_embeddings.device)

        model.set_pretrained_item_embeddings(
            item_embeddings=torch.vstack([item_embeddings, cold_item_embeddings.to(item_embeddings.device)]),
            delta_embeddings=torch.vstack([delta_embeddings, cold_delta.to(delta_embeddings.device)]),
            add_padding_embedding=False,
            freeze=True,
        )
    else:
        model.set_pretrained_item_embeddings(
            item_embeddings=torch.vstack([item_embeddings, cold_item_embeddings.to(item_embeddings.device)]),
            add_padding_embedding=False,
            freeze=True,
        )


def report_results_to_clearm(results: pl.DataFrame) -> None:
    task = Task.current_task()
    logger = task.get_logger()

    results = results.to_pandas().round(4)

    task.register_artifact(name="cold-start-evaluation", artifact=results)
    logger.report_table(
        title="cold-start-evaluation",
        series="",
        iteration=0,
        table_plot=results,
    )

    results = results.set_index(["recommend-cold-items", "filter-cold-items"])

    for (recommend_cold_items, filter_cold_items), row in results.iterrows():
        for metric_name, metric_value in row.items():
            logger.report_single_value(
                "/".join(
                    [
                        f"recommend-cold-items={recommend_cold_items}",
                        f"filter-cold-items={filter_cold_items}",
                        metric_name,
                    ]
                ),
                metric_value,
            )


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(config: DictConfig) -> None:
    task = get_task(config)

    seed_everything(config.seed)

    if not os.path.exists(os.path.join(config.checkpoint_dir, task.id)):
        os.makedirs(os.path.join(config.checkpoint_dir, task.id))

    datamodule = get_datamodule(config)
    model = get_model(config)

    # Set pre-trained item embeddings if necessary
    warm_item_embeddings = None
    cold_item_embeddings = None
    if config.use_pretrained_item_embeddings:
        warm_item_embeddings, cold_item_embeddings = get_item_embeddings(config)
        model.set_pretrained_item_embeddings(
            warm_item_embeddings.clone(),
            add_padding_embedding=True,
            freeze=False,  # keep original behavior
        )
        logging.info("Set pre-trained item embeddings.")

    recommender = ColdStartSequentialRecommender(
        model,
        learning_rate=config.recommender.learning_rate,
        remove_seen=config.recommender.remove_seen,
        metrics=config.recommender.metrics,
        topk=config.recommender.topk,
    )

    # Change optimizer if necessary (paper-style)
    if isinstance(model, SASRecModelWithTrainableDelta):
        recommender.configure_optimizers = lambda: ConstrainedNormAdam(
            model.parameters(),
            constrained_params=model.delta_embedding.parameters(),
            max_norm=config.max_delta_norm,
            lr=config.recommender.learning_rate,
        )
        logging.info("Switched to ConstrainedNormAdam.")

    # Train recommender
    trainer = get_trainer(config)
    trainer.fit(recommender, datamodule=datamodule)

    # Load checkpoint
    recommender = ColdStartSequentialRecommender.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model=model,
        remove_seen=config.recommender.remove_seen,
        metrics=config.recommender.metrics,
        topk=config.recommender.topk,
    )
    logging.info(f"Loaded checkpoint from {trainer.checkpoint_callback.best_model_path}.")

    # Set cold item embeddings (+ optional cold delta via KNN+gate)
    if config.use_pretrained_item_embeddings:
        add_cold_item_embeddings(recommender.model, cold_item_embeddings, config)

    # Run evaluation
    test_interactions = load_data(config.dataset.test_filepath)
    ground_truth = load_data(config.dataset.gt_filepath)

    results = ColdStartEvaluationPipeline(
        recommender,
        trainer,
        test_interactions,
        ground_truth,
    ).run()
    report_results_to_clearm(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
