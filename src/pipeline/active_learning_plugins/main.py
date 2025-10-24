import argparse
from pathlib import Path
from typing import Union

from pipeline.active_learning_plugins.collect_bbox_by_item import (
    DatumaroBBoxIndexBuilder
)
from pipeline.active_learning_plugins.replace_dataset import (
    replace_then_export_dataset
)
from pipeline.active_learning_plugins.sample_dataset import (
    get_missing_datasets,
    sample_dataset_item,
    sample_unlabeled_dataset_item
)

PathLike = Union[str, Path]


def main():
    """Run CLI entry point to synchronize/patch source annotations and export GT.

    The command reads CLI arguments, prepares the output directory, determines
    which dataset items should be corrected (either missing/unlabeled images or
    IoU-sampled items), and then replaces/exports the dataset accordingly.

    Notes
    -----
    - This function does not return a value; it performs side effects:
      filesystem I/O, dataset loading, sampling, and export.
    - Argument validation is minimal; downstream functions are expected to raise
      if inputs are invalid.

    Raises
    ------
    FileNotFoundError
        If required files/directories referenced by arguments are missing (raised
        indirectly by downstream calls).
    ValueError
        If argument values are inconsistent (raised indirectly by downstream calls).
    """

    def build_parser() -> argparse.ArgumentParser:
        """Build and return the argument parser for this CLI.

        Returns
        -------
        argparse.ArgumentParser
            Configured parser defining required CLI arguments for dataset paths,
            formats, and the export destination.

        Notes
        -----
        The parser enforces presence of:
        - ``--original_dataset_path``: base dataset directory.
        - ``--src_annotation_path``/``--src_annotation_format``: source annotations.
        - ``--target_annotation_path``/``--target_annotation_format``: target annotations.
        - ``--output_dir``: destination directory for the corrected dataset.
        """
        parser = argparse.ArgumentParser(
            prog="update-src-dataset",
            description="Synchronize/patch source annotations with target results and write corrected GT.",
        )
        parser.add_argument(
            "--original_dataset_path",
            required=True,
            help="Path to the original dataset directory (e.g., ./dump/dump_yolow_ws_all_train).",
        )
        parser.add_argument(
            "--src_annotation_path",
            required=True,
            help="Path to the SOURCE annotation dataset directory (e.g., ./dump/dump_yolow_ws_all_train).",
        )
        parser.add_argument(
            "--src_annotation_format",
            required=True,
        )
        parser.add_argument(
            "--target_annotation_path",
            required=True,
        )
        parser.add_argument(
            "--target_annotation_format",
            required=True,
        )
        parser.add_argument(
            "--output_dir",
            required=True,
            help="Directory to write the corrected ground-truth dataset.",
        )
        # Optional flag (disabled for now):
        # parser.add_argument(
        #     "--overwrite",
        #     action="store_true",
        #     help="Allow writing into a non-empty output directory.",
        # )
        return parser

    parser = build_parser()
    args = parser.parse_args()

    # Ensure the output directory exists before writing any files.
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which items need correction.
    # NOTE: The following call expects a function that computes the set of images
    # missing annotations. Ensure argument names/types match the function's spec.
    missing_dataset = get_missing_datasets(
        image_path_list=str(args.original_dataset_path),
        src_annotation_path=str(args.src_annotation_path),
        src_annotation_format=args.src_annotation_format,
    )

    # Choose sampling strategy:
    # - If there are unlabeled items, sample from those first.
    # - Otherwise, fall back to IoU-based sampling between source and target datasets.
    if len(missing_dataset) > 0:
        dataset_item_to_be_corrected = sample_unlabeled_dataset_item(
            missing_image_paths=missing_dataset
        )
    else:
        dataset_item_to_be_corrected = sample_dataset_item(
            src_annotation_path=str(args.src_annotation_path),
            src_annotation_format=args.src_annotation_format,
            target_annotation_path=str(args.target_annotation_path),
            target_annotation_format=args.target_annotation_format,
        )

    # Load the source dataset to be patched/replaced.
    index_builder = DatumaroBBoxIndexBuilder()

    src_ds = index_builder.datumaro_import(
        str(args.src_annotation_path), args.src_annotation_format
    )

    # Apply replacements and export the corrected dataset to `out_dir`.
    # NOTE: `replace_then_export_dataset` is expected to:
    # - update items in-place (or produce a modified dataset),
    # - export using the desired format,
    # - and respect `save_media` or similar flags as implemented.
    replace_then_export_dataset(
        output_dir=str(out_dir),
        dataset=src_ds,
        items_to_be_replaced=dataset_item_to_be_corrected,
    )
