#!/usr/bin/env python3

import logging

from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI

from polarbearplus import RNAVAE, DictLogger
from polarbearplus.datamodules import RnaDataModule

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class VAECLI(LightningCLI):  # noqa D101
    def add_arguments_to_parser(self, parser):  # noqa D102
        parser.link_arguments("data.num_genes", "model.ngenes", apply_on="instantiate")
        parser.link_arguments("data.num_batches", "model.nbatches", apply_on="instantiate")
        parser.link_arguments("data.logbatchmean", "model.logbatchmeans", apply_on="instantiate")
        parser.link_arguments("data.logbatchvar", "model.logbatchvars", apply_on="instantiate")


cli = VAECLI(RNAVAE, RnaDataModule, trainer_defaults={"logger": lazy_instance(DictLogger)}, seed_everything_default=42)
