from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml


@dataclass
class DetectionOntology:
    """
    Map prompts (aliases/synonyms) to a canonical class name.

    Rules:
      - Many prompts → one class (M:1) is allowed.
      - One prompt → many classes is NOT allowed (each prompt has exactly one class).
    """

    prompt_to_class: Dict[Any, str] = field(default_factory=dict)

    def __post_init__(self):
        """
        Build helper indices and the prompt-ID → class-ID migration map.
        Outputs (attributes):
            - id_to_prompt: Dict[int, Any]
            - id_to_class_name: Dict[int, str]
            - reverse_class_names: Dict[str, int]
            - id_migration_map: Dict[int, int]

        Example:
            {p1 -> c1, p2 -> c2, p3 -> c1}
            prompts = [p1, p2, p3]
            class_names = [c1, c2]
        """
        prompts = list(dict.fromkeys(self.prompt_to_class.keys()))
        self.id_to_prompt = {
            prompt_id: prompt for prompt_id, prompt in enumerate(prompts)
        }
        class_names = list(dict.fromkeys(self.prompt_to_class.values()))
        self.id_to_class_name = {
            cls_id: cls_name for cls_id, cls_name in enumerate(class_names)
        }
        self.reverse_class_names = {v: k for k, v in self.id_to_class_name.items()}
        self.id_migration_map = self.build_migration_map()

    def prompts(self) -> List[Any]:
        """
        Return:
            List[Any]: Prompts ordered by their assigned prompt IDs.
        """
        return list(self.id_to_prompt.values())

    def classes(self) -> List[str]:
        """
        Return:
            List[str]: Class names ordered by their assigned class IDs.
        """
        return list(self.id_to_class_name.values())

    def get_class(self, prompt: str) -> str:
        """
        Resolve a prompt to its canonical class.

        Args:
            prompt (str): A prompt/alias to resolve.

        Returns:
            str: The canonical class name for the given prompt.

        Raises:
            ValueError: If the prompt does not exist in this ontology.
        """
        try:
            return self.prompt_to_class[prompt]
        except KeyError as e:
            raise ValueError(f"Prompt not found in ontology: {prompt}") from e

    def build_migration_map(self) -> Dict[int, int]:
        """
        Build a mapping from prompt IDs to class IDs.

        Returns:
            Dict[int, int]: Mapping old prompt IDs → new class IDs.

        Example:
            id_to_prompts = {0: "green grapes", 1: "purple grapes", 2: "apples"}
            id_to_classes = {0: "grapes", 1: "apples"}
            => migration_map = {0: 0, 1: 0, 2: 1}, newid_to_label = {0:'grapes', 1:'apples'}

        Note:
            Keep `sorted(...)` for deterministic order by old_id; use `.items()` to
            preserve insertion order if preferred.
        """
        migration_map: dict[int, int] = {}

        # Note: Keep `sorted(...)` for deterministic order by old_id; use `.items()` to
        # preserve insertion order if preferred.
        for old_id, prompt in sorted(self.id_to_prompt.items()):
            cls_name = self.prompt_to_class[prompt]
            new_id = self.reverse_class_names[cls_name]
            migration_map[old_id] = new_id

        return migration_map

    @classmethod
    def from_prompt_map(cls, prompt_map: List[tuple]) -> "DetectionOntology":
        """
        Construct ontology from a list of (prompt, class_name) with validation.

        Args:
            prompt_map (List[Tuple[Any, str]]): Sequence of (prompt, class_name) pairs.

        Returns:
            DetectionOntology: Instance with duplicates rejected.

        Raises:
            ValueError: If the same prompt appears more than once.
        """
        prompt_to_class: Dict[Any, str] = {}
        for prompt, cls_name in prompt_map:
            if prompt in prompt_to_class:
                raise ValueError(f"Duplicate prompt detected: {prompt}")
            prompt_to_class[prompt] = cls_name
        return cls(prompt_to_class=prompt_to_class)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        normalize: Callable[[str], str] = lambda s: s.strip(),
        forbid_duplicates: bool = True,
    ) -> "DetectionOntology":
        """
        Load ontology from a YAML mapping: {prompt: class_name}.

        Args:
            yaml_path (str | Path): YAML file path whose top-level must be a mapping.
            normalize (Callable[[str], str]): Normalizer applied to prompt keys (default: strip).
            forbid_duplicates (bool): If True, raise on normalized key collision.

        Returns:
            DetectionOntology: Instance built from the YAML mapping.

        Raises:
            ValueError:
                - If YAML top-level is not a mapping.
                - If any key/value is not a string.
                - If duplicate prompts are found after normalization (when `forbid_duplicates` is True).

        Example:
            YAML file:
                purple grapes: ripe grapes
                lightgreen grapes: unripe grapes
            -> { "purple grapes": "ripe grapes", ... }
        """
        # Normalize the input path to a Path object.
        yaml_path = Path(yaml_path)

        # Open the YAML file in UTF-8 and parse safely.
        # If the file is empty, `safe_load` may return None → coerce to {}.
        with yaml_path.open("r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f) or {}

        # Validate that the top-level YAML structure is a mapping (dict-like).
        if not isinstance(data, dict):
            raise ValueError("The top-level YAML document must be a mapping (dict).")

        # Prepare the result mapping: prompt (normalized) → class name (stripped).
        prompt_to_class: dict[str, str] = {}

        # Iterate over all YAML key-value pairs.
        for prompt, cls_name in data.items():
            # Enforce both key and value are strings to avoid ambiguous ontology states.
            if not isinstance(prompt, str) or not isinstance(cls_name, str):
                raise ValueError(
                    f"Both key and value must be strings: {prompt!r} -> {cls_name!r}"
                )

            # Apply normalization to the prompt (e.g., trimming, lowercasing if provided).
            prompt = normalize(prompt)

            # Optionally detect and block duplicate prompts after normalization.
            if forbid_duplicates and prompt in prompt_to_class:
                raise ValueError(
                    f"Duplicate prompt detected after normalization: {prompt!r}"
                )

            # Strip whitespace from class labels for consistency.
            prompt_to_class[prompt] = cls_name.strip()

        # Construct and return a fully-initialized ontology instance.
        return cls(prompt_to_class=prompt_to_class)
