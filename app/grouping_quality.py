# app/grouping_quality.py

import re
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for title and entity comparison.
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(text: str) -> set:
    """
    Convert text into a set of meaningful words.
    """

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "has",
        "have",
        "this",
        "that",
        "after",
        "before",
        "into",
        "over",
        "under",
        "as",
        "it",
        "its",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
    }

    words = normalize_text(text).split()

    return {
        word
        for word in words
        if len(word) > 2 and word not in stop_words
    }


# ---------------------------------------------------------
# Title similarity
# ---------------------------------------------------------

def title_similarity(
    title_a: str,
    title_b: str
) -> float:
    """
    Calculate Jaccard similarity between two titles.
    """

    words_a = tokenize(title_a)
    words_b = tokenize(title_b)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)

    if not union:
        return 0.0

    return len(intersection) / len(union)


# ---------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------

def extract_entities(text: str) -> set:
    """
    Extract simple capitalized named entities.

    This lightweight method does not require an
    additional NLP model.
    """

    if not text:
        return set()

    text = str(text)

    pattern = (
        r"\b[A-Z][a-z]+"
        r"(?:\s+[A-Z][a-z]+){0,3}\b"
    )

    entities = re.findall(pattern, text)

    ignored = {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "After",
        "Before",
        "According",
        "Government",
        "Officials",
        "People",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }

    return {
        normalize_text(entity)
        for entity in entities
        if entity not in ignored
    }


def entity_similarity(
    text_a: str,
    text_b: str
) -> float:
    """
    Calculate Jaccard similarity between extracted entities.
    """

    entities_a = extract_entities(text_a)
    entities_b = extract_entities(text_b)

    if not entities_a or not entities_b:
        return 0.0

    intersection = entities_a.intersection(entities_b)
    union = entities_a.union(entities_b)

    if not union:
        return 0.0

    return len(intersection) / len(union)


# ---------------------------------------------------------
# Combined article similarity
# ---------------------------------------------------------

def combined_similarity(
    embedding_a,
    embedding_b,
    title_a: str,
    title_b: str,
    body_a: str = "",
    body_b: str = ""
) -> float:
    """
    Calculate combined similarity between two articles.

    Similarity weights:
    - 85% sentence-embedding similarity
    - 15% title similarity

    Entity similarity is calculated separately in this
    module but is not used in the final score because
    lightweight entity extraction can be unreliable.
    """

    embedding_a = np.asarray(
        embedding_a,
        dtype=float
    ).reshape(1, -1)

    embedding_b = np.asarray(
        embedding_b,
        dtype=float
    ).reshape(1, -1)

    embedding_score = float(
        cosine_similarity(
            embedding_a,
            embedding_b
        )[0][0]
    )

    title_score = title_similarity(
        title_a,
        title_b
    )

    final_score = (
        0.85 * embedding_score
        + 0.15 * title_score
    )

    return round(
        float(final_score),
        4
    )


# ---------------------------------------------------------
# Article field helpers
# ---------------------------------------------------------

def get_article_title(
    article: Dict[str, Any]
) -> str:
    """
    Read the article title from supported field names.
    """

    return str(
        article.get(
            "title",
            article.get(
                "headline",
                ""
            )
        )
        or ""
    )


def get_article_body(
    article: Dict[str, Any]
) -> str:
    """
    Read the article body from supported field names.
    """

    return str(
        article.get(
            "content",
            article.get(
                "body",
                article.get(
                    "clean_text",
                    ""
                )
            )
        )
        or ""
    )


# ---------------------------------------------------------
# Duplicate/event decision
# ---------------------------------------------------------

def are_same_event(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
    embedding_a,
    embedding_b,
    threshold: float = 0.35
) -> bool:
    """
    Decide whether two articles probably describe
    the same event.
    """

    title_a = get_article_title(article_a)
    title_b = get_article_title(article_b)

    body_a = get_article_body(article_a)
    body_b = get_article_body(article_b)

    score = combined_similarity(
        embedding_a=embedding_a,
        embedding_b=embedding_b,
        title_a=title_a,
        title_b=title_b,
        body_a=body_a,
        body_b=body_b,
    )

    return score >= threshold


# ---------------------------------------------------------
# Pairwise grouping
# ---------------------------------------------------------

def pairwise_event_groups(
    articles: List[Dict[str, Any]],
    embeddings,
    threshold: float = 0.35
) -> List[List[int]]:
    """
    Group articles using pairwise similarity.

    Returns lists of article indexes, for example:

    [
        [0, 2, 5],
        [1],
        [3, 4]
    ]

    Articles are connected when their similarity is
    greater than or equal to the selected threshold.
    """

    article_count = len(articles)

    if article_count == 0:
        return []

    embeddings = np.asarray(
        embeddings,
        dtype=float
    )

    if len(embeddings) != article_count:
        raise ValueError(
            "The number of embeddings must match "
            "the number of articles."
        )

    # Each article initially belongs to its own group.
    parent = list(range(article_count))

    def find(index: int) -> int:
        """
        Find the root parent of an article.
        """

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(
        index_a: int,
        index_b: int
    ) -> None:
        """
        Merge two article groups.
        """

        root_a = find(index_a)
        root_b = find(index_b)

        if root_a != root_b:
            parent[root_b] = root_a

    # Compare every pair of articles.
    for i in range(article_count):
        for j in range(i + 1, article_count):

            same_event = are_same_event(
                article_a=articles[i],
                article_b=articles[j],
                embedding_a=embeddings[i],
                embedding_b=embeddings[j],
                threshold=threshold,
            )

            if same_event:
                union(i, j)

    # Build final groups.
    groups = defaultdict(list)

    for index in range(article_count):
        root = find(index)
        groups[root].append(index)

    # Sort groups by size, largest first.
    event_groups = sorted(
        groups.values(),
        key=len,
        reverse=True
    )

    return event_groups


# ---------------------------------------------------------
# Group statistics
# ---------------------------------------------------------

def group_statistics(
    groups: List[List[int]]
) -> Dict[str, Any]:
    """
    Return grouping statistics for the dashboard.
    """

    if not groups:
        return {
            "number_of_groups": 0,
            "largest_group": 0,
            "smallest_group": 0,
            "average_group_size": 0.0,
        }

    sizes = [
        len(group)
        for group in groups
    ]

    return {
        "number_of_groups": len(groups),
        "largest_group": max(sizes),
        "smallest_group": min(sizes),
        "average_group_size": round(
            sum(sizes) / len(sizes),
            2
        ),
    }


# ---------------------------------------------------------
# Possible duplicate detection
# ---------------------------------------------------------

def find_possible_duplicates(
    articles: List[Dict[str, Any]],
    embeddings,
    threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Return possible duplicate or same-event article pairs.

    A higher threshold is used here than for general
    event grouping because duplicate review should be
    more strict.
    """

    possible_duplicates = []

    article_count = len(articles)

    if article_count == 0:
        return possible_duplicates

    embeddings = np.asarray(
        embeddings,
        dtype=float
    )

    if len(embeddings) != article_count:
        raise ValueError(
            "The number of embeddings must match "
            "the number of articles."
        )

    for i in range(article_count):
        for j in range(i + 1, article_count):

            title_a = get_article_title(
                articles[i]
            )

            title_b = get_article_title(
                articles[j]
            )

            body_a = get_article_body(
                articles[i]
            )

            body_b = get_article_body(
                articles[j]
            )

            score = combined_similarity(
                embedding_a=embeddings[i],
                embedding_b=embeddings[j],
                title_a=title_a,
                title_b=title_b,
                body_a=body_a,
                body_b=body_b,
            )

            if score >= threshold:
                possible_duplicates.append({
                    "article_a": i,
                    "article_b": j,
                    "similarity": score,
                    "title_a": title_a,
                    "title_b": title_b,
                })

    possible_duplicates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return possible_duplicates