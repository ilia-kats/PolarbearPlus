#!/usr/bin/env python3

import logging
from typing import Literal

from jsonargparse import lazy_instance
from jsonargparse.typing import Path_dc, Path_fr
from lightning.pytorch.cli import LightningCLI

from polarbearplus import ATACVAE, RNAVAE, DictLogger, StatsWriter, TranslatorBase
from polarbearplus.datamodules import SNAREDataModule

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class VAELoader:  # noqa D101
    def __init__(
        self,
        encoder: Literal["rna", "atac"],
        decoder: Literal["rna", "atac"],
        encoder_checkpoint: Path_fr,
        decoder_checkpoint: Path_fr,
    ):
        encodercls = RNAVAE if encoder == "rna" else ATACVAE
        decodercls = RNAVAE if decoder == "rna" else ATACVAE

        self.encoder = encodercls.load_from_checkpoint(encoder_checkpoint.absolute, map_location="cpu")
        self.decoder = decodercls.load_from_checkpoint(decoder_checkpoint.absolute, map_location="cpu")

        self._direction = f"{encoder}2{decoder}"

    @property
    def direction(self):  # noqa D102
        return self._direction


class TranslatorCLI(LightningCLI):  # noqa D101
    def add_arguments_to_parser(self, parser):  # noqa D102
        parser.add_class_arguments(VAELoader, "vae")
        parser.add_lightning_class_args(StatsWriter, "prediction")
        parser.add_argument("--save_dir", type=Path_dc, help="Directory to save logs and checkpoints to.")
        parser.link_arguments("save_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("save_dir", "prediction.output_dir")
        parser.link_arguments("vae.encoder", "model.init_args.sourcevae", apply_on="instantiate")
        parser.link_arguments("vae.decoder", "model.init_args.destvae", apply_on="instantiate")
        parser.link_arguments("vae.direction", "data.direction", apply_on="instantiate")


cli = TranslatorCLI(
    TranslatorBase,
    SNAREDataModule,
    subclass_mode_model=True,
    trainer_defaults={"logger": lazy_instance(DictLogger)},
    seed_everything_default=42,
)
