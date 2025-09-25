from collections.abc import Iterable
from typing import cast

import pandas as pd
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ciao.typing import PredictSample, Sample

TARGET_COL: str = "cancer"


class ProstateCancer(MetaTiledSlides[Sample]):
    """Prostate cancer class for generating datasets for training, validation, and testing.

    This is a copy of counterfactuals.data.datasets.ProstateCancer that can be used
    independently without requiring the counterfactuals package.

    Attributes:
        uris (Iterable[str]): List of URIs to the slides.
        slide_level (bool): If `True`, the dataset is used for slide-level classification. If `False`,
            the negative tiles are dropped from positive slides. This ensures that when calculating
            tile-level metrics, they are not biased by incomplete annotations. If set to `False` during
            slide-level classification, the metrics will be **incorrect**, as there are only positive
            tiles in the positive slides. Defaults to `False`.
        transforms (TransformType | None, optional): Data augmentation transforms. Defaults to `None`.
    """

    def __init__(
        self,
        uris: Iterable[str],
        slide_level: bool = False,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        self.slide_level = slide_level
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        if "isup_grade" in self.slides.columns:
            self._update_by_isup()
        self._preprocess_tiles()
        return (
            cast(
                "Dataset[Sample]",
                _ProstateCancerSlideTiles(
                    slide,
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    include_label=True,
                    transforms=self.transforms,
                ),
            )
            for _, slide in self.slides.iterrows()
        )

    def _update_by_isup(self) -> None:
        """Set cancer column to 1 if ISUP > 0. Used for testing on the data from Gleason dataset."""
        if "isup_grade" not in self.tiles.columns:
            raise ValueError("Column 'isup_grade' not found in tiles.")
        self.slides[TARGET_COL] = (self.slides["isup_grade"] > 0).astype(int)
        self.tiles[TARGET_COL] = (self.tiles["isup_grade"] > 0).astype(int)

    def _preprocess_tiles(self) -> None:
        """Preprocess the tiles."""
        if TARGET_COL not in self.tiles.columns:
            raise ValueError("Column 'cancer' not found in tiles.")
        # Set cancer column to int
        self.tiles[TARGET_COL] = self.tiles[TARGET_COL].astype(int)
        if self.slide_level:
            # Consider all tiles in cancerous slides to be cancerous
            self._update_cancerous_tiles_in_cancerous_slides()
        else:  # Tile level
            self._drop_non_cancerous_tiles()

    def _update_cancerous_tiles_in_cancerous_slides(self) -> None:
        """Update cancerous tiles in cancerous slides to be all cancerous."""
        # Map slide cancer to tiles
        tiles_slide_cancer = self._get_tiles_slide_cancer_map()
        # Set cancer column to 1 for all tiles in cancerous slides
        self.tiles.loc[tiles_slide_cancer == 1, TARGET_COL] = 1

    def _get_tiles_slide_cancer_map(self) -> pd.Series:
        """Return a Series mapping each tile to its slide's cancer status.

        Returns:
            pd.Series: A Series where the index is the tile DataFrame index and the values are the cancer status
            of the corresponding slide (1 for cancerous, 0 for non-cancerous).
        """
        return (
            self.tiles["slide_id"]
            .map(dict(zip(self.slides["id"], self.slides[TARGET_COL], strict=True)))
            .astype(int)
        )

    def _drop_non_cancerous_tiles(self) -> None:
        """Drop non-cancerous tiles in cancerous slides."""
        # Map slide cancer to tiles
        tiles_slide_cancer = self._get_tiles_slide_cancer_map()
        self.tiles = self.tiles[
            # I.e., remove tiles where the slide is cancerous but the tile is not
            ~((tiles_slide_cancer == 1) & (self.tiles[TARGET_COL] == 0))
        ]


class ProstateCancerPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            cast(
                "Dataset[PredictSample]",
                _ProstateCancerSlideTiles(
                    slide,
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    include_label=False,
                    transforms=self.transforms,
                ),
            )
            for _, slide in self.slides.iterrows()
        )


class _ProstateCancerSlideTiles(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.include_label = include_label
        self.to_tensor = ToTensorV2()
        if len(tiles) == 0:
            raise ValueError(f"No tiles found for slide {slide_metadata.id}")

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        image, metadata = self.slide_tiles[idx]

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        else:
            image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label_value = metadata["cancer"]
            label = torch.tensor(label_value, dtype=torch.float32).unsqueeze(0)
            return image, label
        else:
            return image, metadata
