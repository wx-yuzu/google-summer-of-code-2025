# [TODO] delete this code. This method is for debugging.
def save_raw_outputs(self, raw_outputs, paths) -> None:
    """Optionally persist raw model outputs for inspection.

    Parameters
    ----------
    raw_outputs : Iterable[Any]
        Collection of raw outputs; expected to have an `orig_img` field.
    paths : Iterable[pathlib.Path]
        Destination file names (or sources) aligned with `raw_outputs`.

    Notes
    -----
    - This method currently flips images BGR↔RGB and leaves saving commented
        out. Adjust to your needs (e.g., enable `.save(...)`).
    """
    for out, path in zip(raw_outputs, paths):
        out.orig_img = out.orig_img[..., ::-1]
        # out.save(self.raw_dir_path / path.name)
