"""
Data module for CIAO explanations using counterfactuals datasets.
"""

from typing import TYPE_CHECKING, TypeAlias, cast

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from rationai.mlkit.data.datasets import MetaTiledSlides

PartialConf: TypeAlias = DictConfig
Sample = tuple  # Image, label, metadata tuple


class CIAODataModule(LightningDataModule):
    """
    Data module for CIAO explanations.
    Uses the same datasets as counterfactuals for consistency.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        """Setup datasets for different stages."""
        match stage:
            case "fit":
                self.train = cast(
                    "MetaTiledSlides[Sample]", instantiate(self.datasets["train"])
                )
                self.val = cast(
                    "MetaTiledSlides[Sample]", instantiate(self.datasets["val"])
                )
            case "validate":
                self.val = cast(
                    "MetaTiledSlides[Sample]", instantiate(self.datasets["val"])
                )
            case "test" | "explain":
                self.test = cast(
                    "MetaTiledSlides[Sample]", instantiate(self.datasets["test"])
                )
            case "predict":
                self.predict_dataset = cast(
                    "MetaTiledSlides[Sample]", instantiate(self.datasets["test"])
                )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test,
            batch_size=self.batch_size,  # Usually 1 for CIAO
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def explain_dataloader(self) -> DataLoader:
        """
        Special dataloader for CIAO explanations.
        Always uses batch_size=1 since CIAO processes single images.
        """
        return DataLoader(
            self.test,
            batch_size=1,  # CIAO processes one image at a time
            num_workers=self.num_workers,
            shuffle=False,
        )
