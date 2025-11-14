import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

dataset_name_aliases = {
    "amazon_counterfactual_classification": "amazon_counterfactual",
    "amazon_review_classification": "amazon_review",
    "massive_intent_classification": "massive_intent",
    "massive_scenario_classification": "massive_scenario",
    "japanese_sentiment_classification": "jpn_sentiment",
    "sib200_japanese_classification": "sib200_jpn_cls",
    "sib200_japanese_clustering": "sib200_jpn_clust",
    "nlp_journal_abs_article": "nlp_abs_article",
    "nlp_journal_abs_intro": "nlp_abs_intro",
    "nlp_journal_title_abs": "nlp_title_abs",
    "nlp_journal_title_intro": "nlp_title_intro",
}

TASK_ORDER = ["Retrieval", "STS", "Classification", "Reranking", "Clustering"]
SUMMARY_KEY = "Summary"

"""
Collects the results from the results folder.
"""
# Load reference structure from sbintuitions/sarashina-embedding-v1-1b/summary.json
reference_file = Path("docs/results/sbintuitions/sarashina-embedding-v1-1b/summary.json")
with open(reference_file) as f:
    reference_structure = json.load(f)

# Extract the expected structure
expected_structure = {}
for task_name, task_results in reference_structure.items():
    expected_structure[task_name] = set(task_results.keys())


def has_same_structure(summary: dict, expected: dict) -> bool:
    """Check if summary has exactly the same structure as expected."""
    if set(summary.keys()) != set(expected.keys()):
        return False

    for task_name, datasets in expected.items():
        if set(summary[task_name].keys()) != datasets:
            return False

    return True


# {task_name: {model_signature: {(dataset_name, metric_name): score}}}
all_results: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
for summary_file in Path("docs/results").rglob("summary.json"):
    if not summary_file.exists():
        continue

    with open(summary_file) as f:
        summary = json.load(f)

    # Skip models that don't have the same structure as reference
    if not has_same_structure(summary, expected_structure):
        org_name = summary_file.parent.parent.name
        model_name = summary_file.parent.name
        print(f"Skipping {org_name}/{model_name}: different structure")
        continue

    org_name = summary_file.parent.parent.name
    model_name = summary_file.parent.name
    model_signature = f"{org_name}/{model_name}"

    for task_name, task_results in summary.items():
        task_results_formatted: dict[str, float] = {}
        task_scores: list[float] = []
        for dataset_name, metric_dict in task_results.items():
            metric_name, score = next(iter(metric_dict.items()))
            dataset_name = dataset_name_aliases.get(dataset_name, dataset_name)
            task_results_formatted[f"{dataset_name}<br>({metric_name})"] = score
            task_scores.append(score)
        all_results[task_name][model_signature] = task_results_formatted
        all_results[SUMMARY_KEY][model_signature][task_name] = sum(task_scores) / len(task_scores)

"""
Creates markdown tables for each task.
"""


def format_score(score: float) -> str:
    return f"{score * 100:.2f}"


AVG_COLUMN_NAME = "Avg."
markdown_tables: dict[str, str] = {}
for task_name, task_results in all_results.items():
    # format to markdown table
    dataset_keys = list(task_results[next(iter(task_results))].keys())
    if task_name == SUMMARY_KEY:
        # Only include existing tasks in the summary
        dataset_keys = [task for task in TASK_ORDER if task in all_results]

    header = ["Model", AVG_COLUMN_NAME, *dataset_keys]
    table_list: list[list[str | float]] = []
    for model_signature, dataset_scores in task_results.items():
        # Skip models that don't have all required datasets
        if not all(k in dataset_scores for k in dataset_keys):
            continue

        model_scores = [dataset_scores[k] for k in dataset_keys]
        if task_name == SUMMARY_KEY:
            scores_by_dataset = []
            for _task_name, _task_results in all_results.items():
                if _task_name != SUMMARY_KEY and model_signature in _task_results:
                    scores_by_dataset.extend(list(_task_results[model_signature].values()))
            if not scores_by_dataset:  # Skip if no scores available
                continue
            average_score = sum(scores_by_dataset) / len(scores_by_dataset)
        else:
            average_score = sum(model_scores) / len(model_scores)
        table_list.append([model_signature, average_score, *model_scores])

    # sort by the average score
    avg_idx = header.index(AVG_COLUMN_NAME)
    table_list.sort(key=lambda x: x[avg_idx], reverse=True)

    # make the highest score in each dataset bold
    for dataset_name in [AVG_COLUMN_NAME, *dataset_keys]:
        task_idx = header.index(dataset_name)
        max_score = max(row[task_idx] for row in table_list)
        for row in table_list:
            if row[task_idx] == max_score:
                row[task_idx] = f"**{format_score(row[task_idx])}**"
            else:
                row[task_idx] = format_score(row[task_idx])

    # add header
    table_list.insert(0, ["Model", AVG_COLUMN_NAME, *dataset_keys])
    # Set alignment: left for model names, center for all numeric columns
    col_alignment = ["left"] + ["center"] * (len(dataset_keys) + 1)
    markdown_table = tabulate(table_list, headers="firstrow", tablefmt="pipe", colalign=col_alignment)
    markdown_tables[task_name] = markdown_table

"""
Dump the markdown tables to a file.
"""
with open("leaderboard.md", "w") as f:
    f.write("# Leaderboard\n")
    f.write(
        "This leaderboard shows the results stored under `docs/results`. The scores are all multiplied by 100.\n\n"
    )
    for task_name in [SUMMARY_KEY, *TASK_ORDER]:
        if task_name not in markdown_tables:
            continue
        markdown_table = markdown_tables[task_name]
        f.write(f"## {task_name}\n")

        if task_name == SUMMARY_KEY:
            f.write(
                "\nThe summary shows the average scores within each task. "
                "The average score is the average of scores by dataset.\n\n"
            )

        f.write(markdown_table)
        f.write("\n\n")
