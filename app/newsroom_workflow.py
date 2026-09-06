from datetime import datetime
from copy import deepcopy
import hashlib


WORKFLOW_STATUSES = [
    "Draft",
    "In Review",
    "Approved",
    "Published",
]


def generate_draft_id(group_id):
    """
    Generate a stable draft ID for an event group.
    """
    raw_id = f"group-{group_id}"
    return hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:10]


def create_draft(group_id, articles, summary):
    """
    Create one draft brief for one event group.
    """

    if not articles:
        return None

    first_article = articles[0]

    headline = (
        first_article.get("title")
        or first_article.get("headline")
        or "Untitled News Event"
    )

    source_articles = []

    for article in articles:
        source_articles.append(
            {
                "title": (
                    article.get("title")
                    or article.get("headline")
                    or "Untitled Article"
                ),
                "content": (
                    article.get("content")
                    or article.get("clean_text")
                    or ""
                ),
                "source": article.get("source", "Dataset"),
                "category": article.get("category", "Unknown"),
            }
        )

    draft = {
        "draft_id": generate_draft_id(group_id),
        "group_id": group_id,
        "headline": headline,
        "summary": summary or "No summary generated.",
        "source_articles": source_articles,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "published_at": None,
        "status": "Draft",
        "original_headline": headline,
        "original_summary": summary or "No summary generated.",
        "versions": [],
    }

    return draft


def update_draft_status(draft, new_status):
    """
    Update a draft's workflow status.
    """

    if draft["status"] == "Published":
        return draft

    if new_status not in WORKFLOW_STATUSES:
        raise ValueError(f"Invalid workflow status: {new_status}")

    draft["status"] = new_status
    draft["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if new_status == "Published":
        draft["published_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return draft


def rewrite_draft(draft, new_headline, new_summary):
    """
    Rewrite a draft before publication.

    Published stories cannot be directly edited.
    """

    if draft["status"] == "Published":
        raise ValueError(
            "Published stories are immutable. Create a correction/version instead."
        )

    old_version = {
        "headline": draft["headline"],
        "summary": draft["summary"],
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    draft["versions"].append(old_version)

    draft["headline"] = new_headline.strip()
    draft["summary"] = new_summary.strip()
    draft["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return draft


def create_correction_version(draft, new_headline, new_summary):
    """
    Create a new correction/version for a published story.
    The original published story remains unchanged.
    """

    correction = deepcopy(draft)

    correction["draft_id"] = (
        f"{draft['draft_id']}-correction-"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    correction["headline"] = new_headline.strip()
    correction["summary"] = new_summary.strip()
    correction["original_draft_id"] = draft["draft_id"]
    correction["status"] = "Draft"
    correction["created_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    correction["updated_at"] = correction["created_at"]
    correction["published_at"] = None
    correction["versions"] = []

    return correction


def get_published_stories(drafts):
    """
    Return only published stories.
    """
    return [
        draft
        for draft in drafts
        if draft.get("status") == "Published"
    ]


def get_stories_published_yesterday(drafts):
    """
    Return stories published yesterday.
    """

    yesterday = datetime.now().date().fromordinal(
        datetime.now().date().toordinal() - 1
    )

    published_stories = []

    for draft in drafts:
        if draft.get("status") != "Published":
            continue

        published_at = draft.get("published_at")

        if not published_at:
            continue

        try:
            published_date = datetime.strptime(
                published_at,
                "%Y-%m-%d %H:%M:%S"
            ).date()

            if published_date == yesterday:
                published_stories.append(draft)

        except ValueError:
            continue

    return published_stories


def calculate_waiting_time_minutes(draft):
    """
    Calculate time from draft creation to publication.
    """

    if not draft.get("published_at"):
        return None

    try:
        created_at = datetime.strptime(
            draft["created_at"],
            "%Y-%m-%d %H:%M:%S"
        )

        published_at = datetime.strptime(
            draft["published_at"],
            "%Y-%m-%d %H:%M:%S"
        )

        difference = published_at - created_at

        return round(difference.total_seconds() / 60, 2)

    except ValueError:
        return None