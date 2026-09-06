import sys
import os

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd


# ---------------------------------------------------------
# Ensure project root is on the Python path
# ---------------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="News Clustering & Summarization",
    page_icon="📰",
    layout="wide",
)


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from src.preprocess import load_dataset
from src.embedder import (
    load_embedding_model,
    embed_texts,
)

from src.clustering import (
    kmeans_cluster,
    dbscan_cluster,
    evaluate_all_cluster_metrics,
)

from src.summarizer import summarize_cluster

from newsroom_workflow import (
    create_draft,
    update_draft_status,
    rewrite_draft,
    get_published_stories,
    create_correction_version,
    get_stories_published_yesterday,
    calculate_waiting_time_minutes,
)

# --------------------------------------------------
# NEWSROOM SESSION STATE
# --------------------------------------------------

if "drafts" not in st.session_state:
    st.session_state["drafts"] = []

if "selected_role" not in st.session_state:
    st.session_state["selected_role"] = "Reporter"

if "merged_groups" not in st.session_state:
    st.session_state["merged_groups"] = []

# ---------------------------------------------------------
# Main title
# ---------------------------------------------------------

st.title(
    "News Clustering & Summarization Prototype"
)

st.write(
    "This prototype groups similar news articles and "
    "generates cluster summaries using sentence embeddings, "
    "clustering, and extractive summarization."
)


# ---------------------------------------------------------
# Ensure src/ is on the Python path
# ---------------------------------------------------------

from grouping_quality import (
    combined_similarity,
    pairwise_event_groups,
    group_statistics,
    find_possible_duplicates,
)

# ---------------------------------------------------------
# Cached helper functions
# ---------------------------------------------------------

@st.cache_data
def load_data():
    """
    Load and preprocess the news dataset.
    """
    df = load_dataset(
        "data/News_Category_Dataset_v3.json"
    )

    return df


@st.cache_resource
def load_model_cached():
    """
    Load the sentence embedding model once.
    """
    return load_embedding_model()


@st.cache_data
def compute_embeddings(_model, texts):
    """
    Generate sentence embeddings.
    """
    return embed_texts(_model, texts)


@st.cache_data
def perform_clustering(
    embeddings,
    k,
    algorithm="KMeans",
    eps=0.8,
    min_samples=5,
):
    """
    Perform KMeans or DBSCAN clustering.
    """

    if algorithm == "KMeans":

        labels, model = kmeans_cluster(
            embeddings,
            k,
        )

    else:

        labels, model = dbscan_cluster(
            embeddings,
            eps=eps,
            min_samples=min_samples,
        )

    return labels

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

try:

    df = load_data()

except Exception as error:

    st.error(
        f"Unable to load the dataset: {error}"
    )

    st.stop()


if df.empty:

    st.warning(
        "The dataset is empty."
    )

    st.stop()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Dataset Controls")

sample_size = st.sidebar.slider(
    "Select sample size",
    min_value=200,
    max_value=min(2000, len(df)),
    value=min(800, len(df)),
    step=100,
)

algorithm = st.sidebar.selectbox(
    "Clustering algorithm",
    ["KMeans", "DBSCAN"],
)

# ---------------------------------------------------------
# Improved event grouping controls
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("Improved Event Grouping")

event_threshold = st.sidebar.slider(
    "Same-event similarity threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.35,
    step=0.01,
    help=(
        "Lower values create broader event groups. "
        "Higher values create stricter event groups."
    ),
)

# --------------------------------------------------
# NEWSROOM ROLE SELECTOR
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Newsroom Role")

selected_role = st.sidebar.selectbox(
    "Select your role",
    ["Reporter", "Editor", "Desk Head"],
    key="selected_role",
)

st.sidebar.info(
    f"Current role: {selected_role}"
)

st.sidebar.caption(
    f"Current threshold: {event_threshold:.2f}"
)


# ---------------------------------------------------------
# Algorithm-specific controls
# ---------------------------------------------------------

if algorithm == "KMeans":

    k = st.sidebar.slider(
        "Number of clusters (k)",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
    )

    eps = None
    min_samples = None

else:

    st.sidebar.info(
        "DBSCAN does not use k. "
        "It discovers the number of clusters automatically."
    )

    eps = st.sidebar.slider(
        "DBSCAN eps",
        min_value=0.05,
        max_value=1.50,
        value=0.40,
        step=0.05,
    )

    min_samples = st.sidebar.slider(
        "DBSCAN min_samples",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
    )

    k = None


# ---------------------------------------------------------
# Sample dataset
# ---------------------------------------------------------

sample_df = df.sample(
    n=sample_size,
    random_state=42,
).copy()


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

model = load_model_cached()


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

st.write(
    "Generating embeddings... "
    "(takes a few seconds)"
)

try:

    embeddings = compute_embeddings(
        model,
        sample_df["clean_text"].tolist(),
    )

except Exception as error:

    st.error(
        f"Unable to generate embeddings: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# Convert dataframe into article dictionaries
# ---------------------------------------------------------

articles = sample_df.rename(
    columns={
        "headline": "title",
        "clean_text": "content",
        "category": "category",
    }
).to_dict("records")


# Add a source field because the dataset does not
# contain a separate publisher/source column.

for article in articles:
    if not article.get("source"):
        article["source"] = "HuffPost Kaggle Dataset"

# ---------------------------------------------------------
# Improved event grouping
# ---------------------------------------------------------

st.markdown("---")
st.header("Improved Event Grouping")

st.write(
    "This grouping method compares articles using "
    "embedding similarity and title similarity. "
    "It is independent of the KMeans or DBSCAN labels."
)


# ---------------------------------------------------------
# Similarity diagnostics
# ---------------------------------------------------------

st.subheader("Similarity Diagnostics")

if len(articles) >= 2:

    diagnostic_score = combined_similarity(
        embedding_a=embeddings[0],
        embedding_b=embeddings[1],
        title_a=articles[0].get("title", ""),
        title_b=articles[1].get("title", ""),
        body_a=articles[0].get("content", ""),
        body_b=articles[1].get("content", ""),
    )

    st.write(
        "Similarity between the first two sampled articles: "
        f"**{diagnostic_score:.3f}**"
    )

    st.caption(
        "Use this diagnostic value to understand whether "
        "the selected threshold is too high or too low."
    )


# ---------------------------------------------------------
# Calculate improved event groups
# ---------------------------------------------------------

with st.spinner(
    "Grouping articles using pairwise event similarity..."
):

    improved_groups = pairwise_event_groups(
        articles=articles,
        embeddings=embeddings,
        threshold=event_threshold,
    )

# --------------------------------------------------
# NEWSROOM DRAFT GENERATION
# --------------------------------------------------

st.markdown("---")
st.header("Newsroom Draft Generation")

st.write(
    "Generate one draft brief for each detected event group."
)

if st.button("Generate Drafts from Event Groups"):

    existing_group_ids = {
        draft["group_id"]
        for draft in st.session_state["drafts"]
    }

    generated_count = 0

    for group_id, group_indexes in enumerate(improved_groups):

        # Skip groups that already have a draft
        if group_id in existing_group_ids:
            continue

        # Skip empty groups
        if len(group_indexes) == 0:
            continue

        # Get all articles belonging to this event group
        group_articles = [
            articles[index]
            for index in group_indexes
        ]

        # Extract article text for summarization
        group_texts = []

        for article in group_articles:

            article_text = (
                article.get("content")
                or article.get("clean_text")
                or article.get("title")
                or ""
            )

            group_texts.append(article_text)

        # Generate the event-group summary
        try:

            group_summary = summarize_cluster(
                group_texts
            )

        except Exception as error:

            group_summary = (
                "Summary generation failed. "
                "Please review the source articles. "
                f"Error: {error}"
            )

        # Create the newsroom draft object
        draft = create_draft(
            group_id=group_id,
            articles=group_articles,
            summary=group_summary,
        )

        # Save the draft in Streamlit session state
        if draft is not None:

            st.session_state["drafts"].append(
                draft
            )

            generated_count += 1

    if generated_count > 0:

        st.success(
            f"{generated_count} new draft(s) generated successfully."
        )

    else:

        st.info(
            "No new drafts generated. "
            "Drafts may already exist."
        )

# --------------------------------------------------
# DRAFT REVIEW DESK
# --------------------------------------------------

st.markdown("---")
st.header("Draft Review Desk")

drafts = st.session_state["drafts"]

if not drafts:

    st.info(
        "No drafts available yet. "
        "Click 'Generate Drafts from Event Groups'."
    )

else:

    st.write(f"Total drafts: {len(drafts)}")

    for draft_index, draft in enumerate(drafts):

        with st.expander(
            f"{draft['status']} | "
            f"Group {draft['group_id']} | "
            f"{draft['headline']}"
        ):

            st.markdown(
                f"### {draft['headline']}"
            )

            st.write(
                f"**Draft ID:** `{draft['draft_id']}`"
            )

            st.write(
                f"**Event Group:** {draft['group_id']}"
            )

            st.write(
                f"**Created:** {draft['created_at']}"
            )

            st.write(
                f"**Updated:** {draft['updated_at']}"
            )

            st.write(
                f"**Status:** `{draft['status']}`"
            )

            st.markdown("#### Draft Summary")

            st.write(draft["summary"])

            st.markdown("#### Source Articles")

            st.write(
                f"Number of sources: "
                f"{len(draft['source_articles'])}"
            )

            for source_index, source in enumerate(
                draft["source_articles"],
                start=1
            ):

                st.markdown(
                    f"**{source_index}. {source['title']}**"
                )

                st.caption(
                    f"Source: {source.get('source', 'Dataset')}"
                )

            # --------------------------------------------------
            # REPORTER ACTIONS
            # --------------------------------------------------

            if selected_role == "Reporter":

                st.markdown("#### Reporter Actions")

                if draft["status"] == "Draft":

                    if st.button(
                        "Send for Editor Review",
                        key=f"review_{draft_index}",
                    ):

                        update_draft_status(
                            draft,
                            "In Review",
                        )

                        st.success(
                            "Draft sent for editor review."
                        )

                        st.rerun()

                elif draft["status"] == "In Review":

                    st.info(
                        "This draft is waiting for editor review."
                    )

                elif draft["status"] == "Approved":
                    st.success(
                        "This draft has been approved by an editor."
                    )

                elif draft["status"] == "Published":
                    st.success(
                        "This story is published and immutable."
                    )
                    
            # --------------------------------------------------
            # EDITOR ACTIONS
            # --------------------------------------------------

            if selected_role == "Editor":

                st.markdown("#### Editor Actions")

                # Published stories cannot be edited directly
                if draft["status"] == "Published":

                    st.success(
                        "This story is published and immutable."
                    )

                    st.write(
                        f"Published at: {draft['published_at']}"
                    )

                    st.info(
                        "Create a correction version if changes are required."
                    )

                else:

                    edited_headline = st.text_input(
                        "Edit headline",
                        value=draft["headline"],
                        key=f"headline_{draft_index}",
                    )

                    edited_summary = st.text_area(
                        "Edit summary",
                        value=draft["summary"],
                        height=180,
                        key=f"summary_{draft_index}",
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        if st.button(
                            "Save Rewrite",
                            key=f"rewrite_{draft_index}",
                        ):

                            rewrite_draft(
                                draft,
                                edited_headline,
                                edited_summary,
                            )

                            st.success(
                                "Draft rewritten successfully."
                            )

                            st.rerun()

                    with col2:

                        if st.button(
                            "Approve Draft",
                            key=f"approve_{draft_index}",
                        ):

                            if draft["status"] in [
                                "Draft",
                                "In Review",
                            ]:

                                update_draft_status(
                                    draft,
                                    "Approved",
                                )

                                st.success(
                                    "Draft approved successfully."
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    "This draft cannot be approved "
                                    "from its current status."
                                )

                    with col3:

                        if st.button(
                            "Publish Story",
                            key=f"publish_{draft_index}",
                        ):

                            if draft["status"] != "Approved":

                                st.error(
                                    "Only approved drafts can be published."
                                )

                            else:

                                update_draft_status(
                                    draft,
                                    "Published",
                                )

                                st.success(
                                    "Story published successfully."
                                )

                                st.rerun()         

# --------------------------------------------------
# PUBLISHED STORIES
# --------------------------------------------------

st.markdown("---")
st.header("Published Stories")

published_stories = get_published_stories(
    st.session_state["drafts"]
)

if not published_stories:

    st.info("No stories have been published yet.")

else:

    for published_index, published_story in enumerate(
        published_stories
    ):

        with st.expander(
            f"Published | {published_story['headline']}"
        ):

            st.markdown(
                f"### {published_story['headline']}"
            )

            st.write(
                published_story["summary"]
            )

            st.write(
                f"**Published at:** "
                f"{published_story['published_at']}"
            )

            st.write(
                f"**Original draft ID:** "
                f"{published_story['draft_id']}"
            )

            st.success(
                "This published story is immutable."
            )

            if selected_role == "Editor":

                st.markdown(
                    "#### Create Correction Version"
                )

                correction_headline = st.text_input(
                    "Correction headline",
                    value=published_story["headline"],
                    key=f"correction_headline_{published_index}",
                )

                correction_summary = st.text_area(
                    "Correction summary",
                    value=published_story["summary"],
                    height=180,
                    key=f"correction_summary_{published_index}",
                )

                if st.button(
                    "Create Correction Draft",
                    key=f"correction_{published_index}",
                ):

                    correction = create_correction_version(
                        published_story,
                        correction_headline,
                        correction_summary,
                    )

                    st.session_state["drafts"].append(
                        correction
                    )

                    st.success(
                        "Correction draft created. "
                        "The original published story remains unchanged."
                    )

                    st.rerun()

# ---------------------------------------------------------
# Display grouping statistics
# ---------------------------------------------------------

group_stats = group_statistics(
    improved_groups
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Improved Groups",
    group_stats["number_of_groups"],
)

col2.metric(
    "Largest Group",
    group_stats["largest_group"],
)

col3.metric(
    "Smallest Group",
    group_stats["smallest_group"],
)

col4.metric(
    "Average Group Size",
    group_stats["average_group_size"],
)

st.success(
    f"Created {group_stats['number_of_groups']} "
    f"improved event groups using threshold "
    f"{event_threshold:.2f}"
)


# ---------------------------------------------------------
# Interpretation of grouping result
# ---------------------------------------------------------

if group_stats["number_of_groups"] == len(articles):

    st.warning(
        "Every article is currently in its own group. "
        "Try reducing the similarity threshold, for example "
        "to 0.25 or 0.20."
    )

elif group_stats["largest_group"] == 1:

    st.warning(
        "No multi-article event groups were detected. "
        "Try reducing the threshold."
    )

else:

    st.info(
        "Multi-article event groups were detected. "
        "Review them below to check whether the articles "
        "describe the same event."
    )


# ---------------------------------------------------------
# Display improved event groups
# ---------------------------------------------------------

st.subheader("Improved Event Groups")

for group_number, group_indexes in enumerate(
    improved_groups,
    start=1,
):

    group_articles = [
        articles[index]
        for index in group_indexes
    ]

    with st.expander(
        f"Event Group {group_number} "
        f"({len(group_articles)} articles)",
        expanded=False,
    ):

        st.markdown("### Articles in this event")

        for article in group_articles:

            title = article.get(
                "title",
                "Untitled article",
            )

            source = article.get(
                "source",
                article.get(
                    "publisher",
                    "Unknown source",
                ),
            )

            category = article.get(
                "category",
                "Unknown category",
            )

            st.markdown(
                f"**{title}**"
            )

            st.caption(
                f"Source: {source} | "
                f"Category: {category}"
            )

            st.divider()

# ---------------------------------------------------------
# Event Group Merge
# ---------------------------------------------------------

st.markdown("---")
st.header("Merge Event Groups")

if selected_role != "Editor":
    st.info("Only editors can merge event groups.")
else:
    group_options = {
        f"Group {group_number + 1} "
        f"({len(group_indexes)} articles)": group_number
        for group_number, group_indexes in enumerate(improved_groups)
    }

    if len(group_options) < 2:
        st.info("At least two event groups are required for merging.")
    else:
        selected_group_names = st.multiselect(
            "Select two or more event groups to merge",
            options=list(group_options.keys()),
            key="selected_groups_for_merge",
        )

        if st.button("Merge Selected Event Groups"):
            if len(selected_group_names) < 2:
                st.warning("Select at least two event groups.")
            else:
                selected_group_numbers = [
                    group_options[name]
                    for name in selected_group_names
                ]

                merged_article_indexes = []

                for group_number in selected_group_numbers:
                    merged_article_indexes.extend(
                        improved_groups[group_number]
                    )

                merged_article_indexes = list(
                    dict.fromkeys(merged_article_indexes)
                )

                merged_articles = [
                    articles[index]
                    for index in merged_article_indexes
                ]

                merged_texts = [
                    article.get("content")
                    or article.get("title")
                    or ""
                    for article in merged_articles
                ]

                try:
                    merged_summary = summarize_cluster(
                        merged_texts,
                        num_sentences=3,
                    )
                except Exception as error:
                    merged_summary = (
                        "Merged summary generation failed. "
                        f"Please review the source articles. Error: {error}"
                    )

                merged_group = {
                    "group_id": (
                        "Merged-"
                        + "-".join(
                            str(number + 1)
                            for number in selected_group_numbers
                        )
                    ),
                    "articles": merged_articles,
                    "summary": merged_summary,
                }

                st.session_state["merged_groups"].append(
                    merged_group
                )

                st.success(
                    "Selected event groups were merged successfully."
                )

        if st.session_state["merged_groups"]:
            st.subheader("Merged Event Groups")

            for merged_group in st.session_state["merged_groups"]:
                with st.expander(
                    f"{merged_group['group_id']} "
                    f"({len(merged_group['articles'])} articles)"
                ):
                    st.markdown("### Merged Summary")
                    st.write(merged_group["summary"])

                    st.markdown("### Source Articles")

                    for article in merged_group["articles"]:
                        st.write(
                            f"- {article.get('title', 'Untitled')}"
                        )

                    if st.button(
                        "Create Draft from Merged Group",
                        key=f"create_merged_{merged_group['group_id']}",
                    ):
                        merged_draft = create_draft(
                            group_id=merged_group["group_id"],
                            articles=merged_group["articles"],
                            summary=merged_group["summary"],
                        )

                        if merged_draft is not None:
                            st.session_state["drafts"].append(
                                merged_draft
                            )

                            st.success(
                                "Draft created from merged event group."
                            )

                            st.rerun()

# ---------------------------------------------------------
# Possible duplicate review
# ---------------------------------------------------------

st.markdown("---")
st.header("Possible Duplicate or Same-Event Articles")

with st.spinner(
    "Checking for possible duplicate stories..."
):

    duplicate_pairs = find_possible_duplicates(
        articles=articles,
        embeddings=embeddings,
        threshold=event_threshold,
    )


if not duplicate_pairs:

    st.info(
        "No possible duplicate or same-event pairs detected "
        "at the current threshold."
    )

else:

    st.write(
        f"Detected {len(duplicate_pairs)} possible "
        f"same-event pairs."
    )

    for pair_number, pair in enumerate(
        duplicate_pairs[:20],
        start=1,
    ):

        with st.expander(
            f"Possible Match {pair_number} "
            f"— Similarity: {pair['similarity']:.2f}"
        ):

            st.markdown("**Article 1**")

            st.write(
                pair["title_a"]
            )

            st.markdown("**Article 2**")

            st.write(
                pair["title_b"]
            )

            st.caption(
                "These articles may describe the same event. "
                "Review them before merging."
            )


# ---------------------------------------------------------
# Standard clustering
# ---------------------------------------------------------

st.markdown("---")
st.header("Standard Article Clustering")

st.write(
    "The following section uses the selected clustering "
    "algorithm to group the sampled articles."
)

with st.spinner(
    "Clustering articles..."
):

    labels = perform_clustering(
        embeddings=embeddings,
        k=k if k is not None else 0,
        algorithm=algorithm,
        eps=eps if eps is not None else 0.8,
        min_samples=(
            min_samples
            if min_samples is not None
            else 5
        ),
    )


sample_df["cluster"] = labels


# ---------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------

metrics = evaluate_all_cluster_metrics(
    embeddings,
    labels,
)

st.subheader("Clustering Metrics")

st.write(
    f"Number of clusters: "
    f"{metrics['num_clusters']}"
)

if metrics["silhouette_score"] is not None:

    st.write(
        f"Silhouette Score: "
        f"{metrics['silhouette_score']:.3f}"
    )

else:

    st.write(
        "Silhouette Score: not defined. "
        "At least two non-noise clusters are required."
    )


if metrics["davies_bouldin_index"] is not None:

    st.write(
        f"Davies-Bouldin Index: "
        f"{metrics['davies_bouldin_index']:.3f}"
    )

else:

    st.write(
        "Davies-Bouldin Index: not defined."
    )


if metrics["calinski_harabasz_index"] is not None:

    st.write(
        f"Calinski-Harabasz Index: "
        f"{metrics['calinski_harabasz_index']:.1f}"
    )

else:

    st.write(
        "Calinski-Harabasz Index: not defined."
    )


# ---------------------------------------------------------
# DBSCAN special messages
# ---------------------------------------------------------

unique_labels = set(labels)

if algorithm == "DBSCAN":

    if unique_labels == {-1}:

        st.warning(
            "DBSCAN marked all articles as noise. "
            "Try increasing eps or decreasing min_samples."
        )

    elif len(unique_labels) == 1:

        st.info(
            "DBSCAN found only one cluster. "
            "Try reducing eps or increasing min_samples."
        )


# ---------------------------------------------------------
# Cluster statistics
# ---------------------------------------------------------

cluster_ids = sorted(
    sample_df["cluster"].unique()
)

cluster_stats = []

for cluster_id in cluster_ids:

    cluster_df = sample_df[
        sample_df["cluster"] == cluster_id
    ]

    cluster_size = len(cluster_df)

    if cluster_size > 0:

        top_category = (
            cluster_df["category"]
            .value_counts()
            .idxmax()
        )

    else:

        top_category = "N/A"

    cluster_stats.append(
        {
            "cluster_id": cluster_id,
            "size": cluster_size,
            "top_category": top_category,
        }
    )


# ---------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------

st.markdown("---")
st.header("Cluster Summaries")

for cluster_id in cluster_ids:

    stats = next(
        item
        for item in cluster_stats
        if item["cluster_id"] == cluster_id
    )

    cluster_df = sample_df[
        sample_df["cluster"] == cluster_id
    ]

    if (
        algorithm == "DBSCAN"
        and cluster_id == -1
    ):

        cluster_name = "Noise / Outliers"

    else:

        cluster_name = f"Cluster {cluster_id}"

    header_text = (
        f"{cluster_name} - "
        f"{stats['top_category']} "
        f"({stats['size']} articles)"
    )

    with st.expander(
        header_text,
        expanded=False,
    ):

        # -------------------------------------------------
        # Generate summary
        # -------------------------------------------------

        texts = cluster_df[
            "clean_text"
        ].tolist()

        summary = summarize_cluster(
            texts,
            num_sentences=3,
        )

        max_len = 400

        if (
            isinstance(summary, str)
            and len(summary) > max_len
        ):

            short_summary = (
                summary[:max_len]
                + "..."
            )

        else:

            short_summary = summary

        st.markdown(
            f"**Summary (truncated):** "
            f"{short_summary}"
        )

        if (
            isinstance(summary, str)
            and len(summary) > max_len
        ):

            with st.expander(
                "Show full summary"
            ):

                st.write(summary)

        # -------------------------------------------------
        # Category distribution
        # -------------------------------------------------

        st.markdown(
            "**Category distribution in this cluster:**"
        )

        if len(cluster_df) > 0:

            category_counts = (
                cluster_df["category"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "category",
                "count",
            ]

            st.dataframe(
                category_counts,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.write(
                "No articles in this cluster."
            )

        # -------------------------------------------------
        # Top headlines
        # -------------------------------------------------

        st.subheader(
            "Top 10 Headlines"
        )

        headlines = cluster_df[
            "headline"
        ].tolist()

        for headline in headlines[:10]:

            st.write(
                f"- {headline}"
            )

        # -------------------------------------------------
        # Random article
        # -------------------------------------------------

        if st.button(
            "Show a random article from this cluster",
            key=f"random_{cluster_id}",
        ):

            random_article = (
                cluster_df
                .sample(1)
                .iloc[0]
            )

            st.markdown(
                "**Random article headline:** "
                f"{random_article['headline']}"
            )

            st.markdown(
                "**Category:** "
                f"{random_article['category']}"
            )

            st.markdown(
                "**Text:** "
                f"{random_article['clean_text']}"
            )

# ---------------------------------------------------------
# Analytics Dashboard - Desk Head
# ---------------------------------------------------------

if selected_role == "Desk Head":

    st.markdown("---")
    st.header("Newsroom Analytics")

    all_published_stories = get_published_stories(
        st.session_state["drafts"]
    )

    published_yesterday = get_stories_published_yesterday(
        st.session_state["drafts"]
    )

    waiting_times = []

    for story in all_published_stories:

        waiting_time = calculate_waiting_time_minutes(
            story
        )

        if waiting_time is not None:
            waiting_times.append(waiting_time)

    average_waiting_time = (
        sum(waiting_times) / len(waiting_times)
        if waiting_times
        else 0
    )

    # -----------------------------------------------------
    # Summary Metrics
    # -----------------------------------------------------

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Total Published Stories",
            len(all_published_stories),
        )

    with metric_col2:
        st.metric(
            "Published Yesterday",
            len(published_yesterday),
        )

    with metric_col3:
        st.metric(
            "Average Waiting Time",
            f"{average_waiting_time:.1f} min",
        )

    # -----------------------------------------------------
    # Topics Covered
    # -----------------------------------------------------

    st.subheader("Topics Covered")

    topic_counts = {}

    for story in all_published_stories:

        for article in story.get("source_articles", []):

            topic = article.get(
                "category",
                "Unknown",
            )

            if not topic:
                topic = "Unknown"

            topic_counts[topic] = (
                topic_counts.get(topic, 0) + 1
            )

    if topic_counts:

        topic_df = pd.DataFrame(
            list(topic_counts.items()),
            columns=[
                "Topic",
                "Article Count",
            ],
        ).sort_values(
            by="Article Count",
            ascending=False,
        )

        st.dataframe(
            topic_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            topic_df.set_index("Topic")
        )

    else:
        st.info(
            "No topic information is available yet."
        )

    # -----------------------------------------------------
    # Stories Published Yesterday
    # -----------------------------------------------------

    st.subheader("Stories Published Yesterday")

    if published_yesterday:

        yesterday_rows = []

        for story in published_yesterday:

            yesterday_rows.append(
                {
                    "Headline": story.get(
                        "headline",
                        story.get(
                            "title",
                            "Untitled Story",
                        ),
                    ),
                    "Status": story.get(
                        "status",
                        "Published",
                    ),
                    "Published At": story.get(
                        "published_at",
                        "Unknown",
                    ),
                    "Source Articles": len(
                        story.get(
                            "source_articles",
                            [],
                        )
                    ),
                }
            )

        yesterday_df = pd.DataFrame(
            yesterday_rows
        )

        st.dataframe(
            yesterday_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No stories were published yesterday."
        )

    # -----------------------------------------------------
    # Publication Waiting Time
    # -----------------------------------------------------

    st.subheader("Publication Waiting Time")

    waiting_rows = []

    for story in all_published_stories:

        waiting_time = calculate_waiting_time_minutes(
            story
        )

        waiting_rows.append(
            {
                "Headline": story.get(
                    "headline",
                    story.get(
                        "title",
                        "Untitled Story",
                    ),
                ),
                "Draft ID": story.get(
                    "draft_id",
                    "Unknown",
                ),
                "Published At": story.get(
                    "published_at",
                    "Unknown",
                ),
                "Waiting Time": (
                    f"{waiting_time:.1f} minutes"
                    if waiting_time is not None
                    else "Unavailable"
                ),
            }
        )

    if waiting_rows:

        waiting_df = pd.DataFrame(
            waiting_rows
        )

        st.dataframe(
            waiting_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No publication waiting-time data is available."
        )