# News Brief Desk

An AI-powered newsroom application that groups related news articles, identifies possible duplicate coverage, generates concise summaries, and supports a complete editorial workflow for creating, reviewing, approving, publishing, and analyzing news stories.

The project combines Natural Language Processing, sentence embeddings, machine learning clustering, similarity-based event grouping, extractive summarization, and role-based newsroom management in an interactive Streamlit application.

---

## Project Overview

News organizations receive a large number of articles from different sources every day. Several articles may discuss the same real-world event while using different headlines, descriptions, and writing styles.

For example, multiple articles may report different aspects of the same event:

- A major political announcement
- A natural disaster
- A sports match
- A government policy
- A business development
- An entertainment event

Reading every article separately can be time-consuming and may result in:

- Repetitive news coverage
- Duplicate stories
- Difficulty identifying the main event
- Long editorial review time
- Inefficient newsroom publishing
- Difficulty tracking related articles

The **News Brief Desk** application helps organize incoming news articles into meaningful groups and converts related coverage into structured newsroom stories.

The application performs the following tasks:

1. Loads and preprocesses news articles.
2. Generates semantic embeddings using Sentence-Transformers.
3. Groups articles using K-Means or DBSCAN clustering.
4. Evaluates clustering quality using standard metrics.
5. Groups articles that may describe the same real-world event.
6. Detects possible duplicate article pairs.
7. Generates extractive summaries using TextRank.
8. Creates newsroom drafts from event groups.
9. Supports Reporter, Editor, and Desk Head roles.
10. Allows Editors to review and publish stories.
11. Tracks published stories.
12. Supports correction version tracking.
13. Provides newsroom analytics.

---

## Problem Statement

Traditional news aggregation systems often display related articles separately, even when they describe the same real-world event.

For example:

```text
Article 1:
A major earthquake hits the northern region.

Article 2:
Rescue teams are deployed after the earthquake.

Article 3:
Several buildings are damaged in the earthquake.
```

These articles may belong to the same event even though their headlines and descriptions are different.

The objective of this project is to build an intelligent newsroom assistant that can:

- Understand the semantic meaning of news articles.
- Group related articles.
- Identify possible same-event coverage.
- Detect possible duplicate articles.
- Generate concise summaries.
- Assist reporters in creating drafts.
- Allow editors to review and publish stories.
- Provide management-level newsroom analytics.

---

## Project Objectives

The main objectives of this project are:

- Build an NLP-based news processing pipeline.
- Generate semantic representations of news articles.
- Apply machine learning clustering algorithms.
- Compare K-Means and DBSCAN clustering.
- Evaluate clustering quality using standard metrics.
- Improve broad topic clustering through event-based grouping.
- Identify possible duplicate article pairs.
- Generate extractive summaries.
- Create a newsroom draft workflow.
- Support Reporter, Editor, and Desk Head roles.
- Track draft and publication status.
- Support correction version tracking.
- Provide newsroom analytics.
- Include evaluation scripts and unit tests.
- Build an interactive Streamlit application.

---

## Main Features

### 1. News Article Processing

The application loads news articles from the Kaggle News Category Dataset.

The dataset contains fields such as:

- Headline
- Short description
- Category
- Author
- Publication date
- Article URL

The application combines the headline and short description into a single text field.

The text is then cleaned and prepared for embedding generation.

---

### 2. Semantic Embedding Generation

The project uses a pre-trained Sentence-Transformers model to convert article text into numerical vector representations.

The model used in the project is:

```text
sentence-transformers/all-mpnet-base-v2
```

The model captures the semantic meaning of the text.

Articles with different wording but similar meaning may have similar embeddings.

For example:

```text
Government announces a new economic policy.

Officials introduce new financial rules.

New economic changes are announced by the government.
```

Although these sentences use different words, they describe similar concepts.

The embedding model converts each article into a numerical vector that can be compared with other articles.

Embeddings are used for:

- Clustering
- Event grouping
- Duplicate detection
- Similarity comparison

---

### 3. K-Means Clustering

K-Means is used to group articles into a user-selected number of clusters.

The user can select the number of clusters through the Streamlit interface.

For example:

```text
Number of clusters: 6
```

The resulting clusters may represent broad news topics such as:

- Politics
- Sports
- Travel
- Wellness
- Entertainment
- Business
- Technology

The application displays:

- Cluster ID
- Article count
- Dominant category
- Category distribution
- Top headlines
- Cluster-level summary
- Random article inspection

---

### 4. DBSCAN Clustering

DBSCAN is included as a density-based clustering alternative.

The implementation uses cosine distance because cosine distance is suitable for sentence embeddings.

DBSCAN uses the following parameters:

- `eps`
- `min_samples`

DBSCAN can identify:

- Dense groups of similar articles
- Outlier articles
- Articles that do not belong to a strong cluster

Articles labeled as `-1` are treated as noise or outliers.

---

### 5. Clustering Evaluation

The application calculates standard clustering metrics:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Number of clusters

These metrics provide quantitative information about cluster quality.

The metrics are interpreted together with:

- Article headlines
- Article categories
- Cluster summaries
- Event groups
- Similarity relationships

Numerical metrics alone do not guarantee that every cluster represents one real-world event.

---

### 6. Improved Event Grouping

Traditional clustering identifies broad semantic topics. However, articles belonging to the same topic may describe different real-world events.

For example, the topic `Politics` may contain articles about:

- Elections
- Government policies
- International relations
- Political speeches
- Court decisions

The project therefore includes an improved event grouping process.

The event grouping logic uses information such as:

- Semantic similarity
- Article embeddings
- Headline similarity
- Shared important words
- Article category
- Similarity thresholds
- Relationships between related articles

This helps group articles that are more likely to describe the same event.

The event grouping section displays:

- Total event groups
- Largest event group
- Smallest event group
- Average event group size
- Articles inside each event group
- Representative headlines

---

### 7. Duplicate Article Detection

The application identifies possible duplicate or near-duplicate articles.

Duplicate detection compares article embeddings and similarity scores.

The system identifies article pairs that may contain:

- Similar wording
- Similar information
- Repeated coverage
- Highly overlapping content

The duplicate detection section displays:

- First article headline
- Second article headline
- Similarity score
- Article categories
- Article indexes or identifiers

The system does not automatically delete articles.

Duplicate detection is intended to support human editorial review because two similar articles may still contain different information or perspectives about the same event.

---

### 8. Extractive Summarization

The project uses TextRank-based extractive summarization through the `sumy` library.

Extractive summarization selects important sentences from the original article text.

It does not generate completely new information. Instead, it extracts sentences that are considered important according to the TextRank algorithm.

The summarizer can be used for:

- Individual articles
- Clusters
- Event groups
- Newsroom drafts

---

### 9. Newsroom Draft Creation

The Reporter can select an event group and generate a newsroom draft.

A draft may contain:

- Draft ID
- Headline
- Summary
- Source articles
- Reporter information
- Creation timestamp
- Draft status

The draft is then submitted to the Editor for review.

---

### 10. Editorial Review

The Editor can review submitted drafts.

The Editor can:

- Read the headline
- Review the summary
- Inspect source articles
- Approve the draft
- Return the draft for revision
- Publish an approved story

The Editor checks whether:

- The headline accurately represents the event.
- The summary is supported by the source articles.
- The story contains sufficient information.
- Duplicate coverage has been avoided.
- The draft is ready for publication.

---

### 11. Story Publishing

Approved drafts can be published through the Editor workflow.

Published stories are displayed separately from unpublished drafts.

A published story may contain:

- Headline
- Summary
- Source articles
- Draft ID
- Publication timestamp
- Publication status
- Correction history

---

### 12. Correction Version Tracking

The application supports correction handling for published stories.

When a published story needs an update, the application can create a correction version instead of silently replacing the original content.

A correction version may contain:

- Original story reference
- Updated headline
- Updated summary
- Correction reason
- Correction timestamp
- Version number
- Updated source articles

This helps maintain a basic history of changes made to published stories.

---

### 13. Desk Head Analytics

The Desk Head dashboard provides a high-level view of newsroom activity.

It displays:

- Total published stories
- Stories published yesterday
- Average publication waiting time
- Topics covered
- Published story details
- Event grouping statistics
- Duplicate detection results
- Clustering metrics

---

## Application Workflow

The complete workflow is:

```text
News Dataset
     |
     v
Data Loading and Preprocessing
     |
     v
Sentence Embedding Generation
     |
     v
Clustering
     |
     v
Improved Event Grouping
     |
     v
Duplicate Article Detection
     |
     v
Summary Generation
     |
     v
Reporter Draft Creation
     |
     v
Editor Review
     |
     v
Story Approval
     |
     v
Story Publication
     |
     v
Desk Head Analytics
```

---

## Newsroom Roles

The application provides three newsroom roles:

```text
Reporter
Editor
Desk Head
```

---

### Reporter Role

The Reporter is responsible for creating and preparing drafts.

The Reporter can:

1. Review event groups.
2. Select a relevant event.
3. Generate a headline.
4. Generate a summary.
5. Select source articles.
6. Save the draft.
7. Submit the draft for editorial review.

The Reporter workflow converts grouped news articles into a structured newsroom story.

---

### Editor Role

The Editor is responsible for reviewing drafts before publication.

The Editor can:

1. View submitted drafts.
2. Read the headline.
3. Review the summary.
4. Inspect source articles.
5. Approve the draft.
6. Return the draft for revision.
7. Publish the approved story.

The Editor checks whether:

- The headline represents the event accurately.
- The summary is supported by the source articles.
- The story contains sufficient information.
- Duplicate coverage has been avoided.
- The draft is ready for publication.

---

### Desk Head Role

The Desk Head monitors newsroom performance.

The Desk Head can view:

- Total published stories
- Stories published yesterday
- Average publication waiting time
- Topics covered
- Published story details
- Event grouping statistics
- Duplicate detection results
- Clustering metrics

The Desk Head role provides a management-level view of the newsroom workflow.

---

## Draft Status Management

Drafts may move through the following statuses:

```text
Draft
Submitted
Approved
Published
Correction Required
Corrected
```

These statuses help distinguish between:

- Stories being prepared
- Stories waiting for review
- Approved stories
- Published stories
- Stories requiring correction

---

## Technology Stack

### Programming Language

- Python

### User Interface

- Streamlit

### Data Processing

- Pandas
- NumPy

### Natural Language Processing

- Sentence-Transformers
- Text preprocessing
- Cosine similarity

### Machine Learning

- Scikit-learn
- K-Means
- DBSCAN
- Clustering evaluation metrics

### Summarization

- Sumy
- TextRank

### Evaluation

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- ROUGE-1
- ROUGE-L

### Testing

- Python unittest
- Pytest, if installed

### Visualization

- Streamlit charts
- Pandas DataFrames
- Metric cards
- Expanders
- Tables

---

## Project Structure

```text
News-Clustering-and-Summarization/
│
├── app/
│   ├── app.py
│   ├── grouping_quality.py
│   └── newsroom_workflow.py
│
├── data/
│   └── News_Category_Dataset_v3.json
│
├── src/
│   ├── preprocess.py
│   ├── embedder.py
│   ├── clustering.py
│   └── summarizer.py
│
├── evaluation.py
├── test_clustering.py
├── test_embedding.py
├── test_preprocess.py
├── test_summarizer.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The `venv/` and `__pycache__/` directories are local development files and should not be uploaded to GitHub.

---

## System Architecture

```text
                         News Dataset
                              |
                              v
                  Data Loading and Cleaning
                              |
                              v
                  Sentence Embedding Model
                              |
                              v
                    Article Embeddings
                              |
              +---------------+---------------+
              |                               |
              v                               v
       K-Means / DBSCAN                Similarity Analysis
              |                               |
              v                               v
       Topic Clustering                Event Grouping
              |                               |
              v                               v
      Cluster Summaries               Duplicate Detection
              |                               |
              +---------------+---------------+
                              |
                              v
                       Draft Generation
                              |
                              v
                       Reporter Review
                              |
                              v
                        Editor Review
                              |
                              v
                       Story Approval
                              |
                              v
                         Publication
                              |
                              v
                     Desk Head Analytics
```

---

## Module Description

### `app/app.py`

This is the main entry point of the Streamlit application.

It is responsible for:

- Displaying the Streamlit interface
- Loading the dataset
- Providing dataset controls
- Running preprocessing
- Generating embeddings
- Running clustering
- Displaying clustering metrics
- Displaying cluster summaries
- Displaying event groups
- Displaying duplicate article pairs
- Managing Reporter, Editor, and Desk Head roles
- Creating drafts
- Reviewing drafts
- Approving stories
- Publishing stories
- Displaying published stories
- Displaying newsroom analytics

This file acts as the main orchestration layer of the project.

---

### `app/grouping_quality.py`

This module contains the improved event grouping logic.

Its responsibilities include:

- Comparing article embeddings
- Measuring article similarity
- Applying grouping rules
- Creating event groups
- Calculating event group statistics
- Identifying related articles
- Supporting duplicate detection

The module helps distinguish broad topic similarity from possible same-event relationships.

---

### `app/newsroom_workflow.py`

This module manages the editorial workflow.

Its responsibilities include:

- Creating Reporter drafts
- Managing draft metadata
- Managing draft statuses
- Submitting drafts for review
- Reviewing drafts
- Approving stories
- Publishing stories
- Retrieving published stories
- Creating correction versions
- Calculating publication waiting time
- Finding stories published yesterday

This module separates newsroom workflow logic from the Streamlit interface.

---

### `src/preprocess.py`

This module loads and preprocesses the news dataset.

Its responsibilities include:

- Reading the JSON-lines dataset
- Loading articles into a Pandas DataFrame
- Combining headlines and descriptions
- Cleaning article text
- Normalizing whitespace
- Preparing standardized text for embedding generation

The processed data contains fields such as:

```text
headline
short_description
category
clean_text
```

The `clean_text` field is used as the primary input for the embedding model.

---

### `src/embedder.py`

This module generates sentence embeddings.

Its responsibilities include:

- Loading the Sentence-Transformers model
- Encoding article text
- Processing articles in batches
- Returning NumPy embedding arrays

The embeddings represent the semantic meaning of each article and are used for:

- Clustering
- Event grouping
- Duplicate detection
- Similarity comparison

---

### `src/clustering.py`

This module implements clustering algorithms and evaluation metrics.

It supports:

- K-Means clustering
- DBSCAN clustering
- Cosine-distance-based comparison
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Cluster count calculation
- Noise and edge-case handling

---

### `src/summarizer.py`

This module implements extractive summarization.

Its responsibilities include:

- Parsing article text
- Ranking sentences using TextRank
- Selecting important sentences
- Generating article summaries
- Generating cluster-level summaries
- Supporting event and newsroom summaries

---

### `evaluation.py`

This script provides optional offline evaluation.

It can calculate:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- ROUGE-1 score
- ROUGE-L score

The script allows clustering and summarization to be evaluated outside the Streamlit interface.

---

### Test Files

The project contains unit tests for the main processing components.

Test files:

```text
test_clustering.py
test_embedding.py
test_preprocess.py
test_summarizer.py
```

These tests are used to verify:

- Data preprocessing
- Embedding generation
- Clustering behavior
- Summarization behavior
- Output formats
- Basic edge cases

---

## Dataset

The project uses the Kaggle **News Category Dataset**.

The dataset was created from real news articles collected from HuffPost and contains news articles organized into different categories.

The dataset includes fields such as:

- `headline`
- `short_description`
- `category`
- `authors`
- `date`
- `link`

The expected local dataset file is:

```text
data/News_Category_Dataset_v3.json
```

### Dataset Source

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/rmisra/news-category-dataset?resource=download

Dataset owner:

```text
rmisra
```

### Is This a Realistic Dataset?

Yes. The dataset is a realistic, real-world news dataset because:

- It contains actual news articles.
- The articles were collected from HuffPost.
- It contains multiple news categories.
- Articles have realistic headlines and descriptions.
- It represents the type of text processed by news aggregation systems.
- It is suitable for NLP, classification, clustering, similarity analysis, and summarization experiments.

However, the dataset has some limitations:

- It is a historical dataset rather than a live news feed.
- It does not continuously update with new articles.
- It does not provide perfect same-event labels.
- It does not provide verified duplicate-article labels.
- Some articles may discuss similar topics but different events.
- Some articles may contain incomplete article text.

Therefore, this project uses semantic similarity and clustering techniques to estimate related articles and possible duplicate coverage.

### Dataset Preparation

After downloading the dataset, place the JSON file inside the `data/` directory.

Expected structure:

```text
data/
└── News_Category_Dataset_v3.json
```

The application reads the dataset and creates a cleaned text field by combining the headline and short description.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yhuir/News-Clustering-and-Summarization.git
cd News-Clustering-and-Summarization
```

If you are using your own GitHub repository, replace the repository URL with your repository URL.

---

### 2. Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies separate from other Python projects.

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The dependencies include packages required for:

- Streamlit
- Sentence embeddings
- Clustering
- Data processing
- Summarization
- Evaluation
- Testing

---

## Running the Application

From the project root, run:

```bash
streamlit run app/app.py
```

The application will start a local Streamlit server.

Open the local URL displayed in the terminal in your web browser.

---

## Running the Application on Windows

From the project directory:

```powershell
cd "C:\Users\S NITHIN\OneDrive\Desktop\News-Clustering-and-Summarization-main\News-Clustering-and-Summarization-main"
```

Activate the virtual environment:

```powershell
..\venv\Scripts\Activate.ps1
```

Run the application:

```powershell
streamlit run app/app.py
```

---

## Application Usage

### Step 1: Select the Dataset

Select the available dataset and specify the number of articles to process.

For example:

```text
Sample size: 800
```

Using a smaller sample size can reduce processing time during experimentation.

---

### Step 2: Select a Clustering Algorithm

Choose between:

```text
KMeans
DBSCAN
```

For K-Means, select the number of clusters.

For DBSCAN, configure:

- Epsilon
- Minimum samples

---

### Step 3: Generate Embeddings

The application loads the Sentence-Transformers model and converts article text into embeddings.

The model and embeddings are cached to reduce repeated computation.

---

### Step 4: Explore Clusters

The application displays:

- Number of clusters
- Cluster IDs
- Dominant categories
- Article counts
- Category distributions
- Top headlines
- Cluster summaries
- Random article inspection

---

### Step 5: Review Event Groups

The improved event grouping section displays articles that may describe the same real-world event.

Users can inspect the articles within each event group.

---

### Step 6: Inspect Possible Duplicates

The duplicate detection section displays article pairs that may contain duplicate or highly similar coverage.

Users can manually inspect the article pairs before deciding whether they represent duplicate content.

---

### Step 7: Generate a Newsroom Draft

The Reporter role allows users to select an event group and generate a draft story.

The draft includes:

- Headline
- Summary
- Source articles
- Draft metadata

---

### Step 8: Review and Publish

The Editor role allows users to review the draft and approve or publish it.

After publication, the story appears in the Published Stories section.

---

### Step 9: View Desk Head Analytics

The Desk Head role provides a newsroom-level summary of:

- Published stories
- Recent publication activity
- Topic coverage
- Publication waiting time
- Clustering results
- Event grouping results
- Duplicate detection results

---

## Clustering Methods

### K-Means

K-Means groups articles into a fixed number of clusters.

The number of clusters is selected by the user.

Advantages:

- Simple to understand
- Fast for sampled datasets
- Easy to visualize
- Produces stable topic groups when the number of clusters is appropriate

Limitations:

- Requires the number of clusters in advance
- May force unrelated articles into the same cluster
- Assumes clusters can be represented by centroids

---

### DBSCAN

DBSCAN groups articles based on density.

Advantages:

- Does not require the number of clusters in advance
- Can identify outliers
- Can find irregularly shaped groups

Limitations:

- Sensitive to `eps` and `min_samples`
- May label many articles as noise
- May produce one large cluster
- Can be difficult to apply to high-dimensional embeddings

---

## Clustering Evaluation Metrics

### Silhouette Score

The Silhouette Score measures how well an article fits within its assigned cluster compared with other clusters.

The score generally ranges from:

```text
-1 to 1
```

A higher score generally indicates better-separated clusters.

---

### Davies-Bouldin Index

The Davies-Bouldin Index measures the similarity between clusters.

A lower value is generally better because it indicates more compact and better-separated clusters.

---

### Calinski-Harabasz Index

The Calinski-Harabasz Index compares:

- Between-cluster dispersion
- Within-cluster dispersion

A higher value generally indicates better-defined clusters.

---

## Improved Event Grouping

Traditional clustering focuses on broad semantic similarity.

However, two articles may belong to the same topic while describing different events.

The improved event grouping process attempts to identify articles that describe the same real-world event.

The grouping process uses:

- Article embeddings
- Semantic similarity
- Headline similarity
- Shared important terms
- Category information
- Similarity thresholds
- Relationships between related articles

For example:

```text
Article 1:
A major earthquake hits the northern region.

Article 2:
Rescue teams are deployed after the earthquake.

Article 3:
Several buildings are damaged in the earthquake.
```

These articles may be grouped into one event.

Event grouping is different from broad topic clustering:

```text
Topic Clustering:
Groups articles about similar subjects.

Event Grouping:
Groups articles that may describe the same incident.
```

The event grouping dashboard displays:

- Total event groups
- Largest event group
- Smallest event group
- Average event group size
- Article lists for each group
- Representative headlines

---

## Duplicate Article Detection

Duplicate detection compares articles using semantic similarity.

The system identifies possible duplicate or near-duplicate article pairs.

Possible duplicate articles may:

- Use similar wording
- Describe the same event
- Contain overlapping information
- Repeat the same coverage

The system displays similarity information to help the Editor review the articles.

Duplicate detection does not automatically delete articles because two articles may have similar wording but still provide different information.

---

## Summarization

The project uses TextRank extractive summarization.

TextRank ranks sentences according to their importance and selects the most relevant sentences.

The summarizer can be used for:

- Individual articles
- Clusters
- Event groups
- Newsroom drafts

### Advantages

- Does not require supervised training
- Does not require manually labeled summaries
- Relatively lightweight
- Extracts information directly from source text

### Limitations

- Summaries may be generic
- Sentences may not flow naturally
- Important context may be missing
- Mixed event groups may produce less focused summaries

---

## Newsroom Workflow

The newsroom workflow follows these stages:

```text
Event Group
    |
    v
Reporter Creates Draft
    |
    v
Draft Submitted
    |
    v
Editor Reviews Draft
    |
    v
Draft Approved
    |
    v
Story Published
    |
    v
Published Story Analytics
```

### Reporter Workflow

The Reporter:

1. Selects an event group.
2. Reviews source articles.
3. Generates a headline.
4. Generates a summary.
5. Creates a draft.
6. Submits the draft for review.

### Editor Workflow

The Editor:

1. Views submitted drafts.
2. Reviews the headline.
3. Reviews the summary.
4. Inspects source articles.
5. Approves or returns the draft.
6. Publishes the approved story.

### Published Story Workflow

After publication:

- The story appears in Published Stories.
- Publication time is recorded.
- Source articles remain associated with the story.
- Correction versions can be created when required.

---

## Desk Head Analytics

The Desk Head dashboard provides an overview of newsroom activity.

### Total Published Stories

Displays the total number of stories published through the application.

---

### Stories Published Yesterday

Counts stories published on the previous day.

This metric provides a view of recent publishing activity.

---

### Average Publication Waiting Time

The application calculates the average time between draft creation and publication.

The value is displayed in minutes.

For example:

```text
Average Waiting Time: 33.5 minutes
```

This metric helps estimate how long stories remain in the editorial pipeline.

---

### Topics Covered

The dashboard analyzes the categories of source articles used in published stories.

It displays:

- Topic names
- Article counts
- Topic distribution
- Bar chart visualization

---

## Offline Evaluation

The project includes an optional offline evaluation script.

Run:

```bash
python evaluation.py
```

The script can calculate:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- ROUGE-1 score
- ROUGE-L score

The evaluation results depend on:

- Dataset sample
- Number of articles
- Embedding model
- Clustering algorithm
- Hyperparameters
- Article content
- Summary length

---

## Testing

The project includes test files for the main processing components.

Run the tests using:

```bash
python -m unittest discover
```

If Pytest is installed, run:

```bash
pytest
```

The test files are:

```text
test_clustering.py
test_embedding.py
test_preprocess.py
test_summarizer.py
```

The tests verify:

- Data preprocessing
- Embedding generation
- Clustering behavior
- Summarization behavior
- Output formats
- Basic edge cases

### Manual Testing Checklist

- Verify that the dataset loads correctly.
- Verify that article text is cleaned.
- Verify that embeddings are generated.
- Test K-Means with different cluster counts.
- Test DBSCAN with different parameters.
- Check clustering metrics.
- Inspect event groups.
- Inspect possible duplicate pairs.
- Generate a Reporter draft.
- Submit the draft for review.
- Approve the draft as an Editor.
- Publish the story.
- Verify that the story appears under Published Stories.
- Test correction version creation.
- Verify Desk Head analytics.

---

## Example Results

The results depend on the selected dataset sample and clustering parameters.

During one application run, the following results were observed:

```text
Number of clusters: 6
Silhouette Score: 0.017
Davies-Bouldin Index: 4.836
Calinski-Harabasz Index: 3.6
```

The improved event grouping section displayed:

```text
Total Event Groups: 161
Largest Event Group: 7 articles
Smallest Event Group: 1 article
Average Event Group Size: 1.24 articles
```

The application also identified:

```text
Possible Same-Event Article Pairs: 46
```

The Desk Head dashboard displayed:

```text
Total Published Stories: 1
Stories Published Yesterday: 0
Average Waiting Time: 33.5 minutes
```

These values are examples from a particular application run. They may change when the dataset sample, similarity threshold, clustering algorithm, or hyperparameters are changed.

---

## Application Screenshots

Create a `screenshots/` directory in the project root and place the application screenshots inside it.

Suggested structure:

```text
screenshots/
├── 01-reporter-dashboard.png
├── 02-editor-review-desk.png
├── 03-published-story.png
├── 04-desk-head-analytics.png
├── 05-event-grouping.png
└── 06-clustering-metrics.png
```

## Caching and Performance

The application uses Streamlit caching to reduce unnecessary computation.

Caching is used for resources such as:

- Dataset loading
- Embedding model loading
- Embedding generation
- Clustering results

This is important because sentence embedding generation can be computationally expensive.

Caching helps reduce repeated processing when users change interface parameters.

The application is primarily designed for interactive experimentation with sampled datasets.

---

## Limitations

### Dataset Limitations

The current application uses the Kaggle News Category Dataset for experimentation.

The dataset may not contain:

- Multiple live sources for every event
- Complete article bodies
- Consistent publication timestamps
- Ground-truth event labels
- Ground-truth duplicate labels

Therefore, event grouping and duplicate detection are based on semantic similarity and available article metadata.

---

### Clustering Limitations

News articles often contain overlapping topics.

For example, a political article may also discuss:

- Business
- Technology
- International relations
- Public policy

As a result, some clusters may contain mixed topics.

K-Means requires the number of clusters to be selected in advance.

DBSCAN may produce:

- Too many noise points
- One large cluster
- Very few meaningful groups

This is common when density-based clustering is applied to high-dimensional sentence embeddings.

---

### Event Grouping Limitations

The improved event grouping process uses similarity-based rules.

It may sometimes:

- Group articles from related but different events
- Separate articles that describe the same event
- Miss relationships when articles use very different language
- Depend strongly on similarity thresholds

Event grouping should therefore be reviewed by a human editor.

---

### Summarization Limitations

The application uses extractive TextRank summarization.

Potential limitations include:

- Repetitive sentences
- Generic summaries
- Missing important context
- Less natural sentence flow
- Weak summaries for very short articles
- Mixed summaries when an event group contains multiple subtopics

---

### Duplicate Detection Limitations

Duplicate detection identifies possible duplicate articles using similarity scores.

It does not automatically determine whether two articles are legally or editorially identical.

Two articles may have similar wording but provide different information.

Therefore, duplicate pairs should be manually reviewed before removing or rejecting coverage.

---

### Scalability Limitations

The current application is designed for interactive experimentation with sampled datasets.

Processing a very large news dataset may require:

- Vector databases
- Approximate nearest-neighbor search
- Background processing
- Pagination
- Lazy loading
- A backend API
- Distributed embedding generation
- Persistent storage

---

### Newsroom Workflow Limitations

The current newsroom workflow is a functional prototype.

It does not yet include:

- User authentication
- Multi-user database storage
- Real-time collaboration
- Advanced permission management
- External publishing integrations
- Production-grade audit logs
- Production deployment infrastructure

---

## Future Improvements

Possible future improvements include:

1. Integrate real-time news APIs.
2. Add a database for drafts and published stories.
3. Add user authentication.
4. Add multi-user newsroom collaboration.
5. Add human-in-the-loop summary editing.
6. Add abstractive summarization.
7. Add event timeline generation.
8. Add source reliability scoring.
9. Add fact-checking support.
10. Add multilingual news processing.
11. Add HDBSCAN or hierarchical clustering.
12. Add vector database support.
13. Add automatic topic labeling.
14. Add publication scheduling.
15. Add advanced correction and audit history.
16. Add automated editorial quality checks.
17. Add real-time notifications.
18. Add story search and filtering.
19. Add persistent storage for newsroom data.
20. Deploy the application using a cloud hosting platform.

---

## GitHub Repository Guidelines

The repository should contain:

- Source code
- Streamlit application
- Newsroom workflow
- Event grouping logic
- Duplicate detection logic
- Clustering modules
- Summarization modules
- Evaluation script
- Unit tests
- Requirements file
- README documentation
- License
- Application screenshots


## License

This project is licensed under the MIT License.

See the `LICENSE` file for more information.

---

## Author

Developed as an applied Natural Language Processing and Machine Learning project demonstrating:

- Semantic text embeddings
- News article clustering
- Improved event grouping
- Duplicate article detection
- Extractive summarization
- Streamlit application development
- Reporter and Editor workflows
- Desk Head analytics
- Story approval and publication
- Correction version tracking
- Python testing and evaluation
