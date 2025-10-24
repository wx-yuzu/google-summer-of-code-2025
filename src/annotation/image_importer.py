from pathlib import Path
from typing import List, Optional

from PIL import Image


# [TODO] Replace with torch.dataloader
class ImageImporter:
    """Load images from a directory tree by supported file extensions.

    Notes
    -----
    - The list of supported extensions is matched case-insensitively against each
      file's suffix. Subdirectories are scanned recursively.
    - Be cautious with a mutable default argument for ``extensions``; consider
      passing an explicit list from call sites to avoid accidental mutation.
    """

    def __init__(
        self,
        image_dir: Path,
        extensions: Optional[List[str]] = [".jpg", ".jpeg", ".png"],
    ):
        """Initialize the importer and scan the directory for images.

        Parameters
        ----------
        image_dir : pathlib.Path
            Root directory to search for images (recursively).
        extensions : list[str] | None, optional
            File suffixes (including the dot) to accept, case-insensitive.
            Defaults to ``[".jpg", ".jpeg", ".png"]``.

        Attributes
        ----------
        image_dir : pathlib.Path
            The root directory provided at initialization.
        extensions : list[str] | None
            The accepted extensions used during traversal.
        image_paths : list[pathlib.Path]
            Sorted list of discovered image paths in the directory tree.
        """
        self.image_dir = image_dir
        self.extensions = extensions
        # Build the initial cache of image paths.
        self.image_paths = self._traverse_image_paths()

    def _traverse_image_paths(self) -> List[Path]:
        """Traverse the directory tree and collect image file paths.

        Returns
        -------
        list[pathlib.Path]
            A lexicographically sorted list of paths whose suffix matches one of
            ``self.extensions`` (comparison done in lowercase).
        """
        return sorted(
            [
                p
                for p in self.image_dir.rglob("*")
                if p.suffix.lower() in self.extensions
            ]
        )

    def read_image_at(self, idx: int) -> Image.Image:
        """Open and return a single image by index.

        Parameters
        ----------
        idx : int
            Zero-based index into ``self.image_paths``.

        Returns
        -------
        PIL.Image.Image
            The lazily opened PIL image corresponding to the indexed path.

        Raises
        ------
        IndexError
            If ``idx`` is outside the range of ``self.image_paths``.
        FileNotFoundError
            If the underlying file no longer exists at read time.
        """
        return Image.open(self.image_paths[idx])

    def read_all_images(self) -> List[Image.Image]:
        """Read and return all discovered images as a list.

        Notes
        -----
        - This eagerly loads every image into memory. For large datasets,
          prefer a generator-based approach or a DataLoader-style iterator.
          (TODO in code: convert to a generator.)
        - The order of returned images matches the sorted order of paths.

        Returns
        -------
        list[PIL.Image.Image]
            A list of PIL images in the same order as ``self.image_paths``.
        """
        # [TODO] Convert this to a generator for memory efficiency.
        return [self.read_image_at(i) for i in range(len(self.image_paths))]


def test_construct_importer():
    importer = ImageImporter(image_dir=Path("./src/test/mock_datasets"))
    images = importer.read_all_images()
    assert isinstance(images, list), "images should be a list"
    assert all(
        isinstance(img, Image.Image) for img in images
    ), "all items must be PIL.Image"
    assert len(images) > 0, "no images found in test_images"
