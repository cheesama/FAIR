from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger, model_checkpoint

from argparse import Namespace

from transformers import ElectraModel, ElectraTokenizer

from .FAIR_lightning_model import FAIR_trainer

import os, sys
import torch


def train(
    intent_file_path,
    faq_file_path,
    train_ratio=0.8,
    optimizer="AdamW",
    intent_optimizer_lr=1e-4,
    epochs=20,
    batch_size=None,
    gpu_num=0,
    distributed_backend=None,
    checkpoint_prefix='FAIR_model_'
):
    early_stopping = EarlyStopping('val_loss')
    lr_logger = LearningRateLogger()
    checkpoint_callback = model_checkpoint.ModelCheckpoint(prefix=checkpoint_prefix)

    if batch_size is None:
        trainer = Trainer(
            auto_scale_batch_size="power",
            max_epochs=epochs,
            gpus=gpu_num,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stopping,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback
        )
    else:
        trainer = Trainer(
            max_epochs=epochs,
            gpus=gpu_num,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stopping,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback
        )

    model_args = {}
    model_args["epochs"] = epochs
    model_args["batch_size"] = batch_size
    model_args["nlu_data"] = open(intent_file_path, encoding="utf-8").readlines()
    model_args["faq_data"] = open(faq_file_path, encoding="utf-8").readlines()
    model_args["train_ratio"] = train_ratio
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr

    hparams = Namespace(**model_args)
    model = ElectrasaClassifier(hparams)
    trainer.fit(model)
