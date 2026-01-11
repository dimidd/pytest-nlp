"""Constituency parsing module using Stanza.

This module provides functions for matching constituency (phrase structure) tree
patterns using S-expression syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pytest_nlp.models import get_stanza_pipeline


@dataclass
class PatternNode:
    """Represents a node in a constituency pattern.

    :param label: The node label (e.g., 'NP', 'VP', 'NNP').
    :param text: Optional exact text to match (for terminal nodes).
    :param variable: Optional variable name for capture (e.g., 'drug' from '?drug').
    :param children: Child pattern nodes.
    :param allow_partial: If True, matches even if tree has additional children.
    :param is_wildcard: If True, matches any POS tag with same prefix (e.g., 'VB*').
    """

    label: str
    text: str | None = None
    variable: str | None = None
    children: list["PatternNode"] = field(default_factory=list)
    allow_partial: bool = False
    is_wildcard: bool = False


@dataclass
class ConstituencyMatch:
    """Result of a constituency pattern match.

    :param text: Full text of the matched subtree.
    :param captures: Dictionary of captured variables.
    :param tree_str: String representation of the matched subtree.
    """

    text: str
    captures: dict[str, str]
    tree_str: str


def _tokenize_sexpr(sexpr: str) -> list[str]:
    """Tokenize an S-expression into tokens.

    :param sexpr: S-expression string.
    :returns: List of tokens.
    """
    # Add spaces around parentheses for splitting
    sexpr = sexpr.replace("(", " ( ").replace(")", " ) ")
    tokens = sexpr.split()
    return [t for t in tokens if t]


def _parse_pattern(tokens: list[str], pos: int = 0) -> tuple[PatternNode | None, int]:
    """Parse an S-expression pattern into a PatternNode tree.

    :param tokens: List of tokens.
    :param pos: Current position in token list.
    :returns: Tuple of (PatternNode, next position).
    """
    if pos >= len(tokens):
        return None, pos

    token = tokens[pos]

    if token == "(":
        pos += 1
        if pos >= len(tokens):
            return None, pos

        # Parse label
        label = tokens[pos]
        pos += 1

        # Check for wildcard labels (e.g., VB*)
        is_wildcard = label.endswith("*")
        if is_wildcard:
            label = label[:-1]

        node = PatternNode(label=label, is_wildcard=is_wildcard)
        children = []

        while pos < len(tokens) and tokens[pos] != ")":
            if tokens[pos] == "...":
                # Partial match marker
                node.allow_partial = True
                pos += 1
            elif tokens[pos] == "(":
                # Child node
                child, pos = _parse_pattern(tokens, pos)
                if child:
                    children.append(child)
            else:
                # Terminal text or variable
                text = tokens[pos]
                if text.startswith("?"):
                    # Variable capture
                    var_name = text[1:]
                    child = PatternNode(label="", variable=var_name)
                    children.append(child)
                else:
                    # Exact text match
                    child = PatternNode(label="", text=text)
                    children.append(child)
                pos += 1

        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1

        node.children = children
        return node, pos

    return None, pos


def parse_pattern(sexpr: str) -> PatternNode | None:
    """Parse an S-expression pattern string.

    :param sexpr: S-expression pattern string.
    :returns: Root PatternNode or None if parsing fails.

    >>> node = parse_pattern("(NP (NNP Ibuprofen))")
    >>> node.label
    'NP'
    >>> node.children[0].label
    'NNP'
    """
    tokens = _tokenize_sexpr(sexpr)
    node, _ = _parse_pattern(tokens)
    return node


def _get_tree_text(tree: Any) -> str:
    """Extract text from a Stanza constituency tree node.

    :param tree: Stanza Tree object.
    :returns: Concatenated text of all leaves.
    """
    if tree.is_leaf():
        return tree.label
    return " ".join(_get_tree_text(child) for child in tree.children)


def _match_node(
    tree: Any,
    pattern: PatternNode,
    captures: dict[str, str],
) -> bool:
    """Match a pattern node against a constituency tree node.

    :param tree: Stanza Tree node.
    :param pattern: Pattern node to match.
    :param captures: Dictionary to store captured variables.
    :returns: True if pattern matches.
    """
    # Handle terminal text/variable patterns
    if not pattern.label:
        if tree.is_leaf():
            if pattern.variable:
                captures[pattern.variable] = tree.label
                return True
            elif pattern.text:
                return tree.label.lower() == pattern.text.lower()
        return False

    # Check label match
    if pattern.is_wildcard:
        if not tree.label.startswith(pattern.label):
            return False
    else:
        if tree.label != pattern.label:
            return False

    # Handle leaf nodes
    if tree.is_leaf():
        # Pattern has children but tree is a leaf - no match
        if pattern.children:
            return False
        return True

    # If pattern has no children, match any subtree with correct label
    if not pattern.children:
        return True

    # Match children
    pattern_children = pattern.children
    tree_children = list(tree.children)

    if pattern.allow_partial:
        # Partial matching: find pattern children in tree children (order matters)
        tree_idx = 0
        for p_child in pattern_children:
            found = False
            while tree_idx < len(tree_children):
                if _match_node(tree_children[tree_idx], p_child, captures):
                    found = True
                    tree_idx += 1
                    break
                tree_idx += 1
            if not found:
                return False
        return True
    else:
        # Exact matching (all children must match in order)
        if len(pattern_children) != len(tree_children):
            return False
        return all(
            _match_node(t_child, p_child, captures)
            for t_child, p_child in zip(tree_children, pattern_children)
        )


def _find_matches(
    tree: Any,
    pattern: PatternNode,
) -> list[ConstituencyMatch]:
    """Find all matches of pattern in constituency tree.

    :param tree: Root of Stanza constituency tree.
    :param pattern: Pattern to match.
    :returns: List of ConstituencyMatch objects.
    """
    matches = []

    def search(node: Any) -> None:
        captures: dict[str, str] = {}
        if _match_node(node, pattern, captures):
            matches.append(
                ConstituencyMatch(
                    text=_get_tree_text(node),
                    captures=captures,
                    tree_str=str(node),
                )
            )
        # Continue searching in children
        if not node.is_leaf():
            for child in node.children:
                search(child)

    search(tree)
    return matches


def _select_sentences(
    sentences: list[Any],
    indices: list[int] | slice | None,
) -> list[Any]:
    """Select a subset of sentences based on indices or slice.

    :param sentences: List of Stanza Sentence objects.
    :param indices: List of indices, a slice, or None for all.
    :returns: Selected subset of sentences.
    """
    if indices is None:
        return sentences
    if isinstance(indices, slice):
        return sentences[indices]
    return [sentences[i] for i in indices]


def match_constituency(
    doc: str,
    pattern: str,
    sentences: list[int] | slice | None = None,
    stanza_lang: str = "en",
) -> list[dict[str, str]]:
    """Match constituency patterns and return captured variables.

    :param doc: Document text to search in.
    :param pattern: S-expression pattern with optional variable captures.
    :param sentences: Sentence indices or slice to filter.
    :param stanza_lang: Stanza language model to use.
    :returns: List of dictionaries with captured variables for each match.

    >>> matches = match_constituency("The cat sat.", "(S (NP (DT ?det) (NN ?noun)) ...)")
    >>> isinstance(matches, list)
    True
    """
    pipeline = get_stanza_pipeline(stanza_lang)
    stanza_doc = pipeline(doc)

    selected_sentences = _select_sentences(stanza_doc.sentences, sentences)

    pattern_node = parse_pattern(pattern)
    if pattern_node is None:
        return []

    all_matches = []
    for sent in selected_sentences:
        if sent.constituency is not None:
            matches = _find_matches(sent.constituency, pattern_node)
            for m in matches:
                all_matches.append(m.captures)

    return all_matches


def get_constituency_matches(
    doc: str,
    pattern: str,
    sentences: list[int] | slice | None = None,
    stanza_lang: str = "en",
) -> list[ConstituencyMatch]:
    """Match constituency patterns and return full match objects.

    :param doc: Document text to search in.
    :param pattern: S-expression pattern with optional variable captures.
    :param sentences: Sentence indices or slice to filter.
    :param stanza_lang: Stanza language model to use.
    :returns: List of ConstituencyMatch objects.
    """
    pipeline = get_stanza_pipeline(stanza_lang)
    stanza_doc = pipeline(doc)

    selected_sentences = _select_sentences(stanza_doc.sentences, sentences)

    pattern_node = parse_pattern(pattern)
    if pattern_node is None:
        return []

    all_matches = []
    for sent in selected_sentences:
        if sent.constituency is not None:
            matches = _find_matches(sent.constituency, pattern_node)
            all_matches.extend(matches)

    return all_matches


def assert_matches_constituency(
    doc: str,
    pattern: str,
    sentences: list[int] | slice | None = None,
    stanza_lang: str = "en",
    min_matches: int = 1,
    msg: str | None = None,
) -> None:
    """Assert that constituency pattern matches in the document.

    :param doc: Document text to search in.
    :param pattern: S-expression pattern with optional variable captures.
    :param sentences: Sentence indices or slice to filter.
    :param stanza_lang: Stanza language model to use.
    :param min_matches: Minimum number of matches required.
    :param msg: Custom error message.
    :raises AssertionError: If pattern does not match.
    """
    matches = get_constituency_matches(doc, pattern, sentences, stanza_lang)

    if len(matches) < min_matches:
        if msg:
            raise AssertionError(msg)

        # Get constituency trees for error message
        pipeline = get_stanza_pipeline(stanza_lang)
        stanza_doc = pipeline(doc)
        trees = []
        selected = _select_sentences(stanza_doc.sentences, sentences)
        for sent in selected:
            if sent.constituency:
                trees.append(str(sent.constituency))

        raise AssertionError(
            f"Constituency pattern did not match.\n"
            f"  Document: {doc!r}\n"
            f"  Pattern: {pattern}\n"
            f"  Expected at least {min_matches} match(es), found {len(matches)}\n"
            f"  Sentences filter: {sentences}\n"
            f"  Constituency trees:\n" + "\n".join(f"    {t}" for t in trees)
        )


def get_constituency_tree(
    doc: str,
    sentences: list[int] | slice | None = None,
    stanza_lang: str = "en",
) -> list[str]:
    """Get constituency tree representations for sentences.

    Useful for debugging and constructing patterns.

    :param doc: Document text.
    :param sentences: Sentence indices or slice to filter.
    :param stanza_lang: Stanza language model to use.
    :returns: List of constituency tree strings (one per sentence).

    >>> trees = get_constituency_tree("The cat sat on the mat.")
    >>> len(trees) > 0
    True
    """
    pipeline = get_stanza_pipeline(stanza_lang)
    stanza_doc = pipeline(doc)

    selected_sentences = _select_sentences(stanza_doc.sentences, sentences)

    trees = []
    for sent in selected_sentences:
        if sent.constituency is not None:
            trees.append(str(sent.constituency))
    return trees

