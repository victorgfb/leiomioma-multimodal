import gc
import math
import os
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as L
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor
from transformers.models.siglip.modeling_siglip import (
    BaseModelOutputWithPooling,
    SiglipOutput,
)
from pytorch_lightning import seed_everything


class DiagAccurary(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, eval_preds: Tensor) -> None:
        greater_than_mask = torch.gt(probs, eval_preds)

        self.correct_count += torch.sum(greater_than_mask)
        self.total_count += probs.numel()

    def compute(self) -> Tensor:
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.correct_count.device)

        percentage = (self.correct_count.float() / self.total_count.float()) * 100.0
        return percentage


class MostSimDiag(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: Tensor) -> None:
        num_samples = logits.size(0)

        pred_idx_i2t = torch.argmax(logits, dim=1)
        correct_i2t = (
            pred_idx_i2t == torch.arange(num_samples, device=logits.device)
        ).float()

        self.correct_count += torch.sum(correct_i2t).int()
        self.total_count += num_samples

    def compute(self) -> Tensor:
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.correct_count.device)

        percentage = (self.correct_count.float() / self.total_count.float()) * 100.0
        return percentage


class siglipFinetuner(L.LightningModule):
    def __init__(self, siglip_model: SiglipModel, processor, lr, only_projection):
        super().__init__()
        self.transform = processor
        self.siglip_model: SiglipModel = siglip_model
        self.lr = lr

        self.val_most_sim_diag = MostSimDiag().to("cuda")
        self.val_most_sim_diag_2 = MostSimDiag().to("cuda")

        self.train_most_sim_diag = MostSimDiag().to("cuda")
        self.train_most_sim_diag_2 = MostSimDiag().to("cuda")

        if only_projection:
            for param in self.siglip_model.parameters():
                param.requires_grad = False

            for param in self.siglip_model.vision_model.head.parameters():
                param.requires_grad = True

            for param in self.siglip_model.text_model.head.parameters():
                param.requires_grad = True
        else:
            self.siglip_model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        img_path: Optional[list[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Any:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.siglip_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.siglip_model.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.siglip_model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs: BaseModelOutputWithPooling = self.siglip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(
            text_embeds, image_embeds.t().to(text_embeds.device)
        )

        logit_scale, logit_bias = (
            self.siglip_model.logit_scale.to(text_embeds.device),
            self.siglip_model.logit_bias.to(text_embeds.device),
        )
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None

        

        if return_loss:
            # Create a mask where eye[i, j] = 1 if img_path[i] == img_path[j], else 0
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device, dtype=logits_per_text.dtype)
            
            if img_path is not None:
                for i in range(logits_per_text.size(0)):
                    for j in range(logits_per_text.size(0)):
                        if img_path[i] == img_path[j] and img_path[i] is not None and img_path[j] is not None:
                            eye[i, j] = 1
                           
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        output = SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

        logits_per_image = output.logits_per_image

        logits_per_text = output.logits_per_text

        loss = output.loss

        return logits_per_image, logits_per_text, loss

    def training_step(self, batch, batch_idx):
        input = batch

        logits_per_image, logits_per_text, loss = self.forward(
            pixel_values=input["pixel_values"],
            input_ids=input["input_ids"],
            img_path=input["img_path"],
            return_loss=True,
        )

        self.train_most_sim_diag.update(logits_per_image)
        self.train_most_sim_diag_2.update(logits_per_text)

        self.log("train_loss", loss, batch_size=input["pixel_values"].size(0))
        self.log(
            "train_acc",
            self.train_most_sim_diag,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=input["pixel_values"].size(0),
        )
        self.log(
            "train_acc_2",
            self.train_most_sim_diag_2,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=input["pixel_values"].size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input = batch


        logits_per_image, logits_per_text, loss = self.forward(
            pixel_values=input["pixel_values"],
            img_path=input["img_path"],
            input_ids=input["input_ids"],
            return_loss=True,
        )

        self.val_most_sim_diag.update(logits_per_image)
        self.val_most_sim_diag_2.update(logits_per_text)

        self.log("val_loss", loss, batch_size=input["pixel_values"].size(0))
        self.log(
            "val_acc",
            self.val_most_sim_diag,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=input["pixel_values"].size(0),
        )
        self.log(
            "val_acc_2",
            self.val_most_sim_diag_2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=input["pixel_values"].size(0),
        )

        return loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.0001,
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=1000,  # Adjust T_max based on your training steps or epochs
        #     eta_min=1e-6  # Minimum learning rate
        # )

        return optimizer


class DataloaderWrapper:
    def __init__(self, dataset, batch_size, num_workers, collate_fn, shuffle=False):
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)


class imageTextDataset(Dataset):
    def __init__(self, file_path, transform):
        df = pd.read_csv(file_path)
        self.image_paths = df["image_path"].values
        self.captions = df["caption"].values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path: str = self.image_paths[idx]
        caption = self.captions[idx]

        input = self.transform(
            text=[caption],
            images=Image.open(img_path).convert("RGB"),
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )

        if img_path.startswith("/run/media/victor/pessoal/mestrado/codigo/datasets/augmented_leiomioma") or img_path.startswith("/run/media/victor/pessoal/mestrado/dataset/leiomioma"):
            path = img_path.rsplit("_", maxsplit=1)[0].rsplit("leiomioma/", maxsplit=1)[-1].replace(".jpg", "")
        else:
            path = None

        return {
            "pixel_values": input["pixel_values"],
            "input_ids": input["input_ids"],    
            "img_path": path,
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]).squeeze(1),
        "input_ids": torch.stack([x["input_ids"] for x in batch]).squeeze(1),
        "img_path": [x["img_path"] for x in batch]
        #   'attention_mask': torch.stack([x['attention_mask'] for x in batch]).squeeze(1),
    }


class MemoryCleanupCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Optional: Force garbage collection
        gc.collect()
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train(
    train_dataset,
    test_dataset,
    model,
    processor,
    lr,
    batch_size,
    num_workers,
    seed,
    modelo_base,
    folder,
    only_projection=False,
):

    seed_everything(seed, workers=True)

    early_stopping_callback = EarlyStopping("val_loss", patience=3)

    # Create TensorBoard logger first to get version
    logger = TensorBoardLogger(save_dir="./logs/")
    version = logger.version
    
    # Create filename with batch size, learning rate, and version
    filename_template = f"siglip-epoch{{epoch}}-val_loss{{val_loss:.5f}}-batch{batch_size}-lr{lr:.2e}-v{version}-seed{seed}-modelo_base{modelo_base.replace(".ckpt","")}"
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"/run/media/victor/pessoal/mestrado/codigo/train/checkpoint/{folder}",
        filename=filename_template,
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
        save_weights_only=True,
    )

    model.train().to("cuda")

    train_loader = DataloaderWrapper(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True,
    )

    test_loader = DataloaderWrapper(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    siglip_finetuner = siglipFinetuner(model, processor, lr, only_projection)

    memory_cleanup = MemoryCleanupCallback()

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        # max_steps=-1,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        max_epochs=-1,
        logger=logger,
        callbacks=[
            lr_monitor,
            early_stopping_callback,
            checkpoint_callback,
            memory_cleanup,
        ],
        precision="bf16-mixed",
        deterministic=True
    )
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(siglip_finetuner, mode="power")

    trainer.fit(siglip_finetuner, train_loader, test_loader)

    return siglip_finetuner.siglip_model


def check_perfomance(test_dataset, batch_size, model, num_workers):
    da = MostSimDiag().to("cuda")
    model.eval().to("cuda")

    test_loader = DataloaderWrapper(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    with torch.no_grad():
        losses = []
        for inputs in tqdm(test_loader, desc="Evaluating"):
            outputs = model(
                pixel_values=inputs["pixel_values"].to("cuda"),
                input_ids=inputs["input_ids"].to("cuda"),
                return_loss=True,
            )
            losses.append(outputs.loss)
            da.update(outputs.logits_per_image)
            torch.cuda.empty_cache()

    mean_loss = torch.tensor(losses).mean()
    diag = da.compute()

    return mean_loss, diag
