from collections.abc import Iterable
from typing import cast

import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ciao.typing import Sample


class ColorectalCancer(MetaTiledSlides[Sample]):
    """Colorectal cancer dataset class.

    This is a copy of counterfactuals.data.datasets.ColorectalCancer that can be used
    independently without requiring the counterfactuals package.
    """

    def __init__(
        self,
        uris: Iterable[str],
        transforms: BasicTransform | None = None,
        cancer_threshold: float | None = None,
    ) -> None:
        self.transforms = transforms
        self.cancer_threshold = cancer_threshold
        self.slide_level = self.cancer_threshold is None
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        if self.slide_level:
            id_to_cancer = pd.Series(
                self.slides["cancer"].values, index=self.slides["id"]
            ).to_dict()
            self.tiles["cancer"] = self.tiles["slide_id"].map(id_to_cancer)
        else:
            self.tiles["cancer"] = (
                self.tiles["cancer_percentage"] > self.cancer_threshold
            )
        return (
            cast(
                "Dataset[Sample]",
                _ColorectalCancerSlideTiles(
                    slide,
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    transforms=self.transforms,
                ),
            )
            for _, slide in self.slides.iterrows()
        )


class _ColorectalCancerSlideTiles(Dataset[Sample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: BasicTransform | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.slide_label = slide_metadata.cancer if "cancer" in slide_metadata else None
        self.transforms = transforms
        self.to_tensor = ToTensorV2()
        self.slide_metadata = slide_metadata

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample:
        # OpenSlideTilesDataset only returns the image as NDArray
        image = self.slide_tiles[idx]  # This is just the numpy array image

        # Get metadata from the tiles DataFrame
        tile_row = self.slide_tiles.tiles.iloc[idx]
        metadata = tile_row.to_dict()

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        else:
            image = self.to_tensor(image=image)["image"]

        # Get label from tile metadata
        label_value = metadata.get("cancer", 0)
        label = torch.tensor(label_value, dtype=torch.float32).unsqueeze(0)
        # Return image, label, metadata for CIAO main script
        return image, label, metadata
