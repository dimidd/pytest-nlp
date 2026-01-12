"""AMR (Abstract Meaning Representation) module.

This module provides functions for parsing sentences to AMR graphs and
asserting semantic structure using amrlib and penman.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.request import urlretrieve

from pytest_nlp.models import DEFAULT_AMR_MODEL

if TYPE_CHECKING:
    import penman


# =============================================================================
# AMR Model Registry
# =============================================================================

# Maps model name -> (version, tarball name)
_AMR_MODEL_REGISTRY: Final[dict[str, tuple[str, str]]] = {
    "parse_xfm_bart_base": ("v0_1_0", "model_parse_xfm_bart_base-v0_1_0"),
    "parse_xfm_bart_large": ("v0_1_0", "model_parse_xfm_bart_large-v0_1_0"),
    "parse_spring": ("v0_1_0", "model_parse_spring-v0_1_0"),
    "parse_t5": ("v0_2_0", "model_parse_t5-v0_2_0"),
}

_AMRLIB_MODELS_BASE_URL: Final[str] = (
    "https://github.com/bjascob/amrlib-models/releases/download"
)


@dataclass
class AMRMatch:
    """Result of an AMR pattern match.

    :param concept: The matched concept.
    :param variable: The variable name in the AMR graph.
    :param roles: Dictionary of roles and their values.
    :param subgraph: The matched subgraph as a string.
    """

    concept: str
    variable: str
    roles: dict[str, Any] = field(default_factory=dict)
    subgraph: str = ""


@dataclass
class AMRGraph:
    """Wrapper around a penman Graph with convenience methods.

    :param graph: The penman Graph object.
    :param sentence: The original sentence.
    """

    graph: "penman.Graph"
    sentence: str

    @property
    def top(self) -> str:
        """Get the top/root variable of the graph."""
        return self.graph.top

    @property
    def concepts(self) -> list[tuple[str, str]]:
        """Get all (variable, concept) pairs in the graph.

        :returns: List of (variable, concept) tuples.
        """
        return [(t.source, t.target) for t in self.graph.instances()]

    @property
    def roles(self) -> list[tuple[str, str, str]]:
        """Get all (source, role, target) triples in the graph.

        :returns: List of (source, role, target) tuples.
        """
        return [(t.source, t.role, t.target) for t in self.graph.edges()]

    @property
    def attributes(self) -> list[tuple[str, str, Any]]:
        """Get all (source, role, value) attribute triples.

        :returns: List of (source, role, value) tuples.
        """
        return [(t.source, t.role, t.target) for t in self.graph.attributes()]

    def has_concept(self, concept: str) -> bool:
        """Check if the graph contains a specific concept.

        :param concept: Concept to search for (e.g., 'discontinue-01').
        :returns: True if concept exists in the graph.
        """
        return any(c == concept for _, c in self.concepts)

    def has_role(self, role: str, source_concept: str | None = None, target: str | None = None) -> bool:
        """Check if the graph contains a specific role.

        :param role: Role to search for (e.g., ':ARG0', ':polarity').
        :param source_concept: Optional source concept to filter by.
        :param target: Optional target to filter by.
        :returns: True if role exists.
        """
        # Normalize role name
        if not role.startswith(":"):
            role = f":{role}"

        for src, r, tgt in self.roles + self.attributes:
            if r == role:
                if source_concept is not None:
                    # Check if source has the required concept
                    src_concepts = [c for v, c in self.concepts if v == src]
                    if source_concept not in src_concepts:
                        continue
                if target is not None and tgt != target:
                    continue
                return True
        return False

    def is_negated(self, concept: str | None = None) -> bool:
        """Check if the graph (or a specific concept) is negated.

        :param concept: Optional concept to check negation for.
        :returns: True if negated (:polarity - exists).
        """
        for src, role, tgt in self.attributes:
            if role == ":polarity" and tgt == "-":
                if concept is None:
                    return True
                # Check if source has the concept
                src_concepts = [c for v, c in self.concepts if v == src]
                if concept in src_concepts:
                    return True
        return False

    def get_role_fillers(self, role: str, concept: str | None = None) -> list[str]:
        """Get all fillers for a specific role.

        :param role: Role to search for.
        :param concept: Optional source concept to filter by.
        :returns: List of role filler values.
        """
        if not role.startswith(":"):
            role = f":{role}"

        fillers = []
        for src, r, tgt in self.roles + self.attributes:
            if r == role:
                if concept is not None:
                    src_concepts = [c for v, c in self.concepts if v == src]
                    if concept not in src_concepts:
                        continue
                fillers.append(tgt)
        return fillers

    def __str__(self) -> str:
        """Return the AMR graph as a string."""
        import penman
        return penman.encode(self.graph)


def _get_amrlib_data_dir() -> Path:
    """Get the amrlib data directory where models are stored.

    :returns: Path to the amrlib data directory.
    """
    import amrlib

    return Path(amrlib.__file__).parent / "data"


def _get_model_dir(model: str) -> Path | None:
    """Get the model directory path if it exists.

    :param model: Model name (e.g., 'parse_xfm_bart_base').
    :returns: Path to model directory if it exists, None otherwise.
    """
    if model not in _AMR_MODEL_REGISTRY:
        return None

    _, tarball_name = _AMR_MODEL_REGISTRY[model]
    data_dir = _get_amrlib_data_dir()

    # Check for the extracted model directory
    model_dir = data_dir / tarball_name
    if model_dir.exists():
        return model_dir

    # Also check for model_stog symlink (amrlib default)
    stog_link = data_dir / "model_stog"
    if stog_link.exists():
        return stog_link

    return None


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def download_amr_model(model: str = DEFAULT_AMR_MODEL, force: bool = False) -> Path:
    """Download an AMR model if not already present.

    Downloads the model from the amrlib-models GitHub releases and extracts
    it to the amrlib data directory.

    :param model: Model name to download. Options:
        - ``parse_xfm_bart_base`` (default, 492MB)
        - ``parse_xfm_bart_large`` (1.4GB)
        - ``parse_spring`` (1.5GB)
        - ``parse_t5`` (785MB)
    :param force: If True, download even if model already exists.
    :returns: Path to the extracted model directory.
    :raises ValueError: If model name is not recognized.
    :raises OSError: If download or extraction fails.

    >>> # Download default model (skips if already present)
    >>> model_dir = download_amr_model()  # doctest: +SKIP
    >>> model_dir.exists()  # doctest: +SKIP
    True
    """
    if model not in _AMR_MODEL_REGISTRY:
        available = ", ".join(sorted(_AMR_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown AMR model: {model!r}. Available models: {available}"
        )

    version, tarball_name = _AMR_MODEL_REGISTRY[model]
    data_dir = _get_amrlib_data_dir()
    model_dir = data_dir / tarball_name

    # Check if already downloaded
    if model_dir.exists() and not force:
        return model_dir

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build download URL
    # Format: https://github.com/bjascob/amrlib-models/releases/download/{model}-{version}/{tarball_name}.tar.gz
    release_tag = f"{model}-{version}"
    tarball_url = f"{_AMRLIB_MODELS_BASE_URL}/{release_tag}/{tarball_name}.tar.gz"

    # Download to temp file
    tarball_path = data_dir / f"{tarball_name}.tar.gz"

    print(f"Downloading AMR model '{model}' from GitHub...")
    print(f"  URL: {tarball_url}")

    try:
        _ = urlretrieve(tarball_url, tarball_path, reporthook=_download_progress)
        print()  # Newline after progress
    except Exception as e:
        raise OSError(f"Failed to download model from {tarball_url}: {e}") from e

    # Extract tarball
    print(f"Extracting to {data_dir}...")
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
    except Exception as e:
        raise OSError(f"Failed to extract model tarball: {e}") from e
    finally:
        # Clean up tarball
        if tarball_path.exists():
            tarball_path.unlink()

    if not model_dir.exists():
        raise OSError(
            f"Model extraction completed but directory not found: {model_dir}"
        )

    print(f"Model '{model}' installed successfully at {model_dir}")
    return model_dir


@lru_cache(maxsize=2)
def get_amr_parser(model: str = DEFAULT_AMR_MODEL):
    """Load and cache an AMR parser, downloading if necessary.

    :param model: Model name to load. Options include:
        - ``parse_xfm_bart_base`` (default, good balance of quality/speed)
        - ``parse_xfm_bart_large`` (best quality, larger)
        - ``parse_spring`` (alternative architecture)
        - ``parse_t5`` (T5-based)
    :returns: The loaded AMR parser (stog model).

    >>> parser = get_amr_parser()
    >>> parser is not None
    True
    """
    try:
        import amrlib
    except ImportError as e:
        raise ImportError(
            "amrlib is required for AMR functions. "
            "Install it with: pip install pytest-nlp[amr]"
        ) from e

    # Check if model exists, download if not
    model_dir = _get_model_dir(model)
    if model_dir is None:
        model_dir = download_amr_model(model)

    # Load the model from the specific directory
    stog = amrlib.load_stog_model(model_dir=str(model_dir))
    return stog


def parse_amr(sentence: str, model: str = DEFAULT_AMR_MODEL) -> AMRGraph:
    """Parse a sentence into an AMR graph.

    :param sentence: Sentence to parse.
    :param model: AMR parser model to use.
    :returns: AMRGraph object.

    >>> graph = parse_amr("The boy wants to go.")
    >>> graph.has_concept("want-01")
    True
    """
    import penman

    parser = get_amr_parser(model)
    amr_strings = parser.parse_sents([sentence])

    if not amr_strings or not amr_strings[0]:
        raise ValueError(f"Failed to parse sentence: {sentence!r}")

    graph = penman.decode(amr_strings[0])
    return AMRGraph(graph=graph, sentence=sentence)


def parse_amr_batch(sentences: list[str], model: str = DEFAULT_AMR_MODEL) -> list[AMRGraph]:
    """Parse multiple sentences into AMR graphs.

    :param sentences: List of sentences to parse.
    :param model: AMR parser model to use.
    :returns: List of AMRGraph objects.
    """
    import penman

    parser = get_amr_parser(model)
    amr_strings = parser.parse_sents(sentences)

    graphs = []
    for sent, amr_str in zip(sentences, amr_strings):
        if amr_str:
            graph = penman.decode(amr_str)
            graphs.append(AMRGraph(graph=graph, sentence=sent))
        else:
            raise ValueError(f"Failed to parse sentence: {sent!r}")

    return graphs


def _strip_amr_comments(amr_str: str) -> str:
    """Strip comment lines from an AMR string.

    :param amr_str: AMR string potentially containing comment lines.
    :returns: AMR string with comment lines removed.
    """
    lines = amr_str.strip().split("\n")
    # Filter out lines starting with # (metadata comments like "# ::snt ...")
    filtered = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(filtered)


def amr_similarity(graph1: AMRGraph, graph2: AMRGraph) -> float:
    """Compute Smatch similarity between two AMR graphs.

    Smatch is the standard metric for comparing AMR graphs.

    :param graph1: First AMR graph.
    :param graph2: Second AMR graph.
    :returns: Smatch F1 score between 0 and 1.
    """
    try:
        import amrlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "amrlib is required for AMR functions. "
            "Install it with: pip install pytest-nlp[amr]"
        ) from e

    # Use smatch to compute similarity
    from amrlib.evaluate.smatch_enhanced import compute_smatch

    # Strip metadata comments (e.g., "# ::snt ...") that smatch can't parse
    amr1_str = _strip_amr_comments(str(graph1))
    amr2_str = _strip_amr_comments(str(graph2))

    # compute_smatch expects lists of AMR strings
    _, _, f1 = compute_smatch([amr1_str], [amr2_str])
    return f1


def sentence_amr_similarity(sentence1: str, sentence2: str, model: str = DEFAULT_AMR_MODEL) -> float:
    """Compute AMR-based semantic similarity between two sentences.

    :param sentence1: First sentence.
    :param sentence2: Second sentence.
    :param model: AMR parser model to use.
    :returns: Smatch F1 score between 0 and 1.
    """
    graph1 = parse_amr(sentence1, model)
    graph2 = parse_amr(sentence2, model)
    return amr_similarity(graph1, graph2)


def find_concepts(graph: AMRGraph, concept_pattern: str) -> list[AMRMatch]:
    """Find all instances of a concept in the AMR graph.

    :param graph: AMR graph to search.
    :param concept_pattern: Concept to find (can use '*' as wildcard).
    :returns: List of AMRMatch objects.

    >>> graph = parse_amr("The boy wants to go.")
    >>> matches = find_concepts(graph, "want-01")
    >>> len(matches) >= 1
    True
    """
    import fnmatch

    matches = []
    for var, concept in graph.concepts:
        if fnmatch.fnmatch(concept, concept_pattern):
            # Get roles for this variable
            roles = {}
            for src, role, tgt in graph.roles + graph.attributes:
                if src == var:
                    roles[role] = tgt

            matches.append(AMRMatch(
                concept=concept,
                variable=var,
                roles=roles,
            ))

    return matches


def match_amr_pattern(
    graph: AMRGraph,
    concept: str,
    roles: dict[str, str] | None = None,
) -> list[AMRMatch]:
    """Match a pattern against an AMR graph.

    :param graph: AMR graph to search.
    :param concept: Concept to match (supports '*' wildcard).
    :param roles: Optional dict of roles to match (role -> value or concept).
    :returns: List of matching AMRMatch objects.

    >>> graph = parse_amr("The boy wants the girl to believe him.")
    >>> matches = match_amr_pattern(graph, "want-01", {":ARG0": "*"})
    >>> len(matches) >= 1
    True
    """
    import fnmatch

    concept_matches = find_concepts(graph, concept)

    if roles is None:
        return concept_matches

    # Filter by roles
    filtered = []
    for match in concept_matches:
        all_match = True
        for role, expected in roles.items():
            if not role.startswith(":"):
                role = f":{role}"

            if role not in match.roles:
                all_match = False
                break

            actual = match.roles[role]
            # Check if expected matches (could be a variable, concept, or literal)
            if expected != "*" and not fnmatch.fnmatch(str(actual), expected):
                # Also check if the variable's concept matches
                var_concepts = [c for v, c in graph.concepts if v == actual]
                if not any(fnmatch.fnmatch(c, expected) for c in var_concepts):
                    all_match = False
                    break

        if all_match:
            filtered.append(match)

    return filtered


# Assertion functions


def assert_has_concept(
    sentence: str,
    concept: str,
    model: str = DEFAULT_AMR_MODEL,
    msg: str | None = None,
) -> None:
    """Assert that a sentence's AMR contains a specific concept.

    :param sentence: Sentence to parse.
    :param concept: Concept to find (supports '*' wildcard).
    :param model: AMR parser model.
    :param msg: Custom error message.
    :raises AssertionError: If concept not found.
    """
    graph = parse_amr(sentence, model)
    matches = find_concepts(graph, concept)

    if not matches:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Concept {concept!r} not found in AMR.\n"
            f"  Sentence: {sentence!r}\n"
            f"  AMR:\n{graph}"
        )


def assert_has_role(
    sentence: str,
    role: str,
    source_concept: str | None = None,
    target: str | None = None,
    model: str = DEFAULT_AMR_MODEL,
    msg: str | None = None,
) -> None:
    """Assert that a sentence's AMR contains a specific role.

    :param sentence: Sentence to parse.
    :param role: Role to find (e.g., ':ARG0', ':polarity').
    :param source_concept: Optional source concept filter.
    :param target: Optional target filter.
    :param model: AMR parser model.
    :param msg: Custom error message.
    :raises AssertionError: If role not found.
    """
    graph = parse_amr(sentence, model)

    if not graph.has_role(role, source_concept, target):
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Role {role!r} not found in AMR"
            f"{f' (source: {source_concept})' if source_concept else ''}"
            f"{f' (target: {target})' if target else ''}.\n"
            f"  Sentence: {sentence!r}\n"
            f"  AMR:\n{graph}"
        )


def assert_is_negated(
    sentence: str,
    concept: str | None = None,
    model: str = DEFAULT_AMR_MODEL,
    msg: str | None = None,
) -> None:
    """Assert that a sentence's AMR is negated.

    :param sentence: Sentence to parse.
    :param concept: Optional concept to check negation for.
    :param model: AMR parser model.
    :param msg: Custom error message.
    :raises AssertionError: If not negated.
    """
    graph = parse_amr(sentence, model)

    if not graph.is_negated(concept):
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Sentence is not negated"
            f"{f' (concept: {concept})' if concept else ''}.\n"
            f"  Sentence: {sentence!r}\n"
            f"  AMR:\n{graph}"
        )


def assert_not_negated(
    sentence: str,
    concept: str | None = None,
    model: str = DEFAULT_AMR_MODEL,
    msg: str | None = None,
) -> None:
    """Assert that a sentence's AMR is NOT negated.

    :param sentence: Sentence to parse.
    :param concept: Optional concept to check.
    :param model: AMR parser model.
    :param msg: Custom error message.
    :raises AssertionError: If negated.
    """
    graph = parse_amr(sentence, model)

    if graph.is_negated(concept):
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"Sentence is negated"
            f"{f' (concept: {concept})' if concept else ''}.\n"
            f"  Sentence: {sentence!r}\n"
            f"  AMR:\n{graph}"
        )


def assert_amr_pattern(
    sentence: str,
    concept: str,
    roles: dict[str, str] | None = None,
    model: str = DEFAULT_AMR_MODEL,
    min_matches: int = 1,
    msg: str | None = None,
) -> None:
    """Assert that a sentence's AMR matches a pattern.

    :param sentence: Sentence to parse.
    :param concept: Concept to match.
    :param roles: Optional dict of roles to match.
    :param model: AMR parser model.
    :param min_matches: Minimum number of matches required.
    :param msg: Custom error message.
    :raises AssertionError: If pattern doesn't match.
    """
    graph = parse_amr(sentence, model)
    matches = match_amr_pattern(graph, concept, roles)

    if len(matches) < min_matches:
        if msg:
            raise AssertionError(msg)
        raise AssertionError(
            f"AMR pattern not matched.\n"
            f"  Concept: {concept!r}\n"
            f"  Roles: {roles}\n"
            f"  Expected at least {min_matches} match(es), found {len(matches)}\n"
            f"  Sentence: {sentence!r}\n"
            f"  AMR:\n{graph}"
        )


def assert_amr_similarity(
    sentence1: str,
    sentence2: str,
    threshold: float = 0.7,
    model: str = DEFAULT_AMR_MODEL,
    msg: str | None = None,
) -> None:
    """Assert that two sentences have similar AMR structure.

    :param sentence1: First sentence.
    :param sentence2: Second sentence.
    :param threshold: Minimum Smatch F1 score.
    :param model: AMR parser model.
    :param msg: Custom error message.
    :raises AssertionError: If similarity below threshold.
    """
    score = sentence_amr_similarity(sentence1, sentence2, model)

    if score < threshold:
        if msg:
            raise AssertionError(msg)
        graph1 = parse_amr(sentence1, model)
        graph2 = parse_amr(sentence2, model)
        raise AssertionError(
            f"AMR structures not similar enough.\n"
            f"  Sentence 1: {sentence1!r}\n"
            f"  Sentence 2: {sentence2!r}\n"
            f"  Smatch F1: {score:.4f} (threshold: {threshold})\n"
            f"  AMR 1:\n{graph1}\n"
            f"  AMR 2:\n{graph2}"
        )


def clear_amr_cache() -> None:
    """Clear the cached AMR parser."""
    get_amr_parser.cache_clear()

