"""Training loop: data preparation, WCE with gradient accumulation, validation, model selection.

Same procedure as the completed P2/P3 runs (old ``prithvi_xtseg/training.py``):

* Optimiser: AdamW, two learning rates: encoder 1e-5 (pretrained, small steps),
  neck + decoder + head 1e-4 (new layers); weight decay 0.01.
* One optimiser step = one batch of 16 patches, processed as 16 micro-batches of
  one patch (GPU memory). Each micro-batch loss is weighted by its share of the
  batch's class-weight mass, so the sum equals the WCE of the whole batch.
* fp16 autocast + GradScaler on CUDA; gradient-norm clipping at 1.0.
* Every ``validation_interval_steps`` (156 = half a pass over the 312 batches of the
  4,992 patches): validate on the validation band, keep ``best_any.pt`` (max macro-F1) and ``best_accepted.pt``
  (max macro-F1 among checkpoints that pass the acceptance gate); early stop after
  6 validations without +0.001.
* Resumable: ``latest_resume.pt`` stores model, optimiser, scaler, RNG and history.
"""

from __future__ import annotations

import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .. import __version__
from ..config import load_config
from ..data.sampling import (batches_from_schedule, build_batch_schedule, build_validation_windows,
                             sample_training_patches)
from ..data.scene import TrainingScene, stack_crops
from ..inference.maps import predict_map, validate, write_geotiffs
from ..model import build_model
from ..utils import atomic_json, atomic_torch, sha256_file, stable_hash
from .loss import class_weights_from_counts, weighted_cross_entropy

CHECKPOINT_FORMAT = "prithvi_unet/v1"


def seed_everything(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic_algorithms)


def _versions() -> dict[str, str]:
    out = {"prithvi_unet": __version__}
    for name in ["torch", "terratorch", "timm", "numpy", "rasterio", "pandas"]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


class Trainer:
    def __init__(self, config_path: str | Path, data_root: str | Path, run_dir: str | Path,
                 verify_checksums: bool = True) -> None:
        self.config = load_config(config_path)
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock_run()
        self.scene = TrainingScene(self.config, data_root, verify_checksums)
        self.scene.class_weights = class_weights_from_counts(self.scene.train_class_counts)
        self.manifest, self.batches, self.val_windows = self._prepare_data()
        self.model: torch.nn.Module | None = None

    # ------------------------------------------------------------------ setup
    def _lock_run(self) -> None:
        """A run folder belongs to one config + code version; refuse to mix."""
        method = {"config": self.config, "versions": _versions()}
        self.method_hash = stable_hash(method)
        lock = self.run / "RUN_LOCK.json"
        if lock.exists():
            if json.loads(lock.read_text())["method_hash"] != self.method_hash:
                raise ValueError("This run folder was created with another config or library version; use a new folder")
        else:
            if any(self.run.iterdir()):
                raise ValueError("Non-empty run folder without RUN_LOCK.json; use a new folder")
            atomic_json({"method_hash": self.method_hash, "method": method}, lock)

    def _prepare_data(self) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
        out = self.run / "prepared"
        audit_path = out / "DATA_AUDIT.json"
        if audit_path.exists():  # reuse, after checking nothing changed
            audit = json.loads(audit_path.read_text())
            for name, digest in audit["artifact_sha256"].items():
                if sha256_file(out / name) != digest:
                    raise ValueError(f"Prepared artifact changed: {name}")
            manifest = pd.read_csv(out / "train_manifest.csv")
            schedule = pd.read_csv(out / "batch_schedule.csv")
            windows = pd.read_csv(out / "validation_manifest.csv")
            return manifest, batches_from_schedule(schedule), windows

        out.mkdir(parents=True, exist_ok=True)
        s, c = self.scene, self.config
        class_map = {int(k): v for k, v in c["class_map"].items()}
        manifest = sample_training_patches(s.label, s.common_valid, s.domains["split_mask"], s.domains["train_input"],
                                           c["split"], class_map)
        schedule, support = build_batch_schedule(manifest, c["split"], class_map)
        windows = build_validation_windows(s.label.shape, c["split"], s.image_valid, s.val_mask,
                                           s.domains["validation_input"])
        for name, frame in [("train_manifest", manifest), ("batch_schedule", schedule),
                            ("batch_support", support), ("validation_manifest", windows)]:
            frame.to_csv(out / f"{name}.csv", index=False)
        audit = {"train_pixels": int(s.train_mask.sum()), "validation_pixels": int(s.val_mask.sum()),
                 "train_class_counts": s.train_class_counts.tolist(),
                 "class_weights": s.class_weights.tolist(), "images": s.images,
                 "n_patches": len(manifest), "n_batches": int(schedule.batch_index.nunique()),
                 "n_validation_windows": len(windows),
                 "artifact_sha256": {p.name: sha256_file(p) for p in sorted(out.glob("*.csv"))}}
        atomic_json(audit, audit_path)
        return manifest, batches_from_schedule(schedule), windows

    def load_model(self, pretrained: bool = True) -> torch.nn.Module:
        t = self.config["training"]
        seed_everything(t["seed"], t["deterministic_algorithms"])
        self.model = build_model(self.config, pretrained=pretrained).to(self.device)
        return self.model

    # ------------------------------------------------------------------ training
    def _crop(self, manifest_index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(manifest_index)]
        return self.scene.crop(int(row.row), int(row.col), domain=self.scene.domains["train_input"])

    def _batch_for_step(self, step: int) -> np.ndarray:
        """Batch order is reshuffled each pass with seed + pass index."""
        pass_index, within = divmod(step, len(self.batches))
        order = np.random.default_rng(self.config["training"]["seed"] + pass_index).permutation(len(self.batches))
        return self.batches[int(order[within])]

    def _optimizer(self) -> torch.optim.Optimizer:
        t = self.config["training"]
        encoder = [p for p in self.model.encoder.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder}
        new_layers = [p for p in self.model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        return torch.optim.AdamW([{"params": encoder, "lr": t["encoder_lr"]},
                                  {"params": new_layers, "lr": t["decoder_lr"]}], weight_decay=t["weight_decay"])

    def _payload(self, summary: dict | None = None) -> dict[str, Any]:
        return {"format": CHECKPOINT_FORMAT, "model": self.model.state_dict(), "method_hash": self.method_hash,
                "config": self.config, "summary": summary,
                "pretrained_provenance": getattr(self.model, "pretrained_provenance", None)}

    def train(self, stop_after_steps: int | None = None) -> dict[str, Any]:
        t = self.config["training"]
        if self.device.type != "cuda" and stop_after_steps is None:
            raise RuntimeError("Full training needs a GPU (Colab: Runtime > Change runtime type > GPU)")
        latest = self.run / "latest_resume.pt"
        if self.model is None:
            self.load_model(pretrained=not latest.exists())
        weights = self.scene.class_weights.to(self.device)
        optimizer = self._optimizer()
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        state = {"step": 0, "best_any": -1.0, "best_pass": -1.0, "stale": 0, "patience_best": -1.0,
                 "history": [], "validation": [], "stopped": False}
        if latest.exists():
            saved = torch.load(latest, map_location=self.device, weights_only=False)
            if saved["method_hash"] != self.method_hash:
                raise ValueError("Resume checkpoint belongs to another method")
            self.model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            state = saved["state"]
            self.model.pretrained_provenance = saved["pretrained_provenance"]
            torch.set_rng_state(saved["rng_cpu"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all([x.cpu() for x in saved["rng_cuda"]])

        def save_resume() -> None:
            atomic_torch({**self._payload(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                          "state": state, "rng_cpu": torch.get_rng_state(),
                          "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}, latest)
            pd.DataFrame(state["history"]).to_csv(self.run / "training_history.csv", index=False)
            pd.DataFrame(state["validation"]).to_csv(self.run / "validation_history.csv", index=False)

        end = min(t["maximum_steps"], state["step"] + stop_after_steps) if stop_after_steps else t["maximum_steps"]
        for step in tqdm(range(state["step"], end), desc="Optimizer steps"):
            if state["stopped"]:
                break
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            crops = [self._crop(i) for i in self._batch_for_step(step)]
            micro = t["micro_batch_size"]
            groups = [crops[i : i + micro] for i in range(0, len(crops), micro)]
            # Class-weight mass of each micro-batch -> exact WCE of the whole batch.
            masses = [sum(float(weights.cpu()[b["target"][b["target"] != 255]].sum()) for b in g) for g in groups]
            total_mass = sum(masses)
            if total_mass <= 0:
                raise ValueError("Batch has no labelled pixel")
            batch_loss = 0.0
            for group, mass in zip(groups, masses):
                if mass == 0:
                    continue
                batch = stack_crops(group, self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                    logits = self.model(batch["image"], batch["temporal_coords"], batch["location_coords"])
                loss = weighted_cross_entropy(logits.float(), batch["target"], weights) * (mass / total_mass)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss; stopping before the optimiser update")
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["clip_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            state["step"] = step + 1
            state["history"].append({"step": step + 1, "train_WCE": batch_loss, "grad_norm": float(grad_norm),
                                     "seconds": time.monotonic() - started})

            is_validation = (step + 1) % t["validation_interval_steps"] == 0 or step + 1 == t["maximum_steps"]
            if is_validation:
                summary, table = validate(self.model, self.scene, self.val_windows, self.device)
                summary["step"] = step + 1
                state["validation"].append(summary)
                score = summary["macro_F1_11"]
                if score > state["best_any"]:
                    state["best_any"] = score
                    atomic_torch(self._payload(summary), self.run / "best_any.pt")
                if summary["acceptance_passed"] and score > state["best_pass"]:
                    state["best_pass"] = score
                    atomic_torch(self._payload(summary), self.run / "best_accepted.pt")
                if score > state["patience_best"] + t["min_delta"]:
                    state["patience_best"], state["stale"] = score, 0
                else:
                    state["stale"] += 1
                state["stopped"] = state["stale"] >= t["patience_evaluations"]
                table.to_csv(self.run / "latest_validation_per_class.csv", index=False)
                print(f"step {step + 1}: macro-F1 {score:.4f}, macro-IoU {summary['macro_IoU_11']:.4f}, "
                      f"gate {summary['acceptance_passed']}")
            if is_validation or (step + 1) % t["save_interval_steps"] == 0 or step + 1 == end:
                save_resume()

        complete = state["stopped"] or state["step"] >= t["maximum_steps"]
        result = {"step": state["step"], "completed": complete, "early_stopped": state["stopped"],
                  "best_F1": state["best_any"], "best_accepted_F1": state["best_pass"], "method_hash": self.method_hash}
        atomic_json(result, self.run / ("TRAINING_COMPLETE.json" if complete else "TRAINING_PROGRESS.json"))
        return result

    # ------------------------------------------------------------------ final map
    def selected_checkpoint(self) -> Path:
        accepted, any_ = self.run / "best_accepted.pt", self.run / "best_any.pt"
        if accepted.exists():
            return accepted
        if any_.exists():
            print("No checkpoint passed the acceptance gate: the map is exploratory.")
            return any_
        raise FileNotFoundError("No selected checkpoint in this run")

    def map(self) -> dict[str, str]:
        path = self.selected_checkpoint()
        if self.model is None:
            self.load_model(pretrained=False)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"], strict=True)
        arrays = predict_map(self.model, self.scene, self.device, len(self.config["class_map"]),
                             self.config["split"]["patch_policy"]["inference_stride"])
        tags = {"target_date": self.config["target_date"], "input_dates": ",".join(self.config["active_dates"]),
                "checkpoint": path.name, "checkpoint_sha256": sha256_file(path),
                "label_status": self.config["label"]["status"]}
        return write_geotiffs(arrays, self.scene.profile, self.run / "map", self.config["target_date"], tags)
