import argparse
import os
from typing import Literal, List

import dotenv
import litellm
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm.auto import tqdm

DATA_URL = ("hf://datasets/vishnupriyavr/wiki-movie-plots-with-summaries"
            "/wiki_movie_plots_deduped_with_summaries.csv")

DEFAULT_MODEL = "ollama_chat/quant-theoros"
DEFAULT_SAMPLE_SIZE = 100

ACTION = "Action"
COMEDY = "Comedy"
DRAMA = "Drama"
FANTASY = "Fantasy"
HORROR = "Horror"
SCI_FI = "Sci-Fi"
GENRES = [ACTION, COMEDY, DRAMA, FANTASY, HORROR, SCI_FI]

GENRE_CLASSIFY_TEMPLATE = f"""
Classify the genres of the movie based on the title, director, and plot.
You should output a list made from the following genres: {", ".join(GENRES)}

Title: {{title}}
Director: {{director}}
Plot: {{plot}}
Genre(s):
"""

class MovieGenre(BaseModel):
    """A movie genre limited to the following genres: Action, Comedy,
    Drama, Fantasy, Horror, Sci-Fi"""
    genres: List[Literal["Action", "Comedy", "Drama", "Fantasy", "Horror", "Sci-Fi"]]

def import_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_URL)
    data.to_json("movies_w_plots.jsonl", lines=True, orient="records")
    return data

def read_data():
    if os.path.exists("movies_w_plots.jsonl"):
        return pd.read_json("movies_w_plots.jsonl", lines=True)
    else:
        return import_data()

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data[data["Genre"] != ""]
    data = data[data["Genre"] != "unknown"]
    data["genre_label"] = data["Genre"].apply(
        lambda gs: [g for g in GENRES if g.lower() in gs.lower()])
    data = data[data["genre_label"].apply(lambda gs: len(gs) > 0)]
    return data

def ai_classify(
        model: str, title: str, director: str, plot: str
) -> MovieGenre:
    prompt = GENRE_CLASSIFY_TEMPLATE.format(
        title=title, director=director, plot=plot)
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=MovieGenre,
        base_url="http://172.29.224.1:11434/"
    )
    response_content = response["choices"][0]["message"]["content"]
    return MovieGenre.model_validate_json(response_content)

def multilabel_metrics(
        y_true: list[list[str]], y_pred: list[list[str]]
) -> dict[str, float| dict[str, float]]:
    y_true = [set(gs) for gs in y_true]
    y_pred = [set(gs) for gs in y_pred]
    sum_precision = 0.0
    sum_recall = 0.0
    for true, pred in zip(y_true, y_pred):
        sum_precision += len(true.intersection(pred)) / max(0.1, len(pred))
        sum_recall += len(true.intersection(pred)) / max(0.1, len(true))
    by_genres = {}
    for genre in GENRES:
        true_genre = np.array([genre in gs for gs in y_true])
        pred_genre = np.array([genre in gs for gs in y_pred])
        by_genres[genre] = {
            "precision": (true_genre * pred_genre).sum() / len(y_true),
            "recall": (true_genre * pred_genre).sum() / len(y_true),
            "total_pred": pred_genre.sum(),
            "total_true": true_genre.sum()
        }
    return {
        "precision": sum_precision / len(y_true),
        "recall": sum_recall / len(y_true),
        "by_genres": by_genres,
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_MODEL,
    help="LLM model to use")
parser.add_argument(
    "--sample-size",
    type=int,
    default=DEFAULT_SAMPLE_SIZE,
    help="Number of movies to classify")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing labeled sample")


def main(arg_list: list[str] | None = None):
    if arg_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)
    dotenv.load_dotenv(".env")
    data = read_data()
    data = prepare_data(data)
    sample = pd.DataFrame()
    for genre in GENRES:
        genre_sample = data[data["genre_label"].apply(lambda gs: genre in gs)]
        genre_sample = genre_sample[~genre_sample.index.isin(sample.index)]
        genres_sample = genre_sample.sample(args.sample_size//len(GENRES), random_state=123)
        sample = pd.concat([sample, genres_sample])
    sample = sample.reset_index(drop=True)
    if not os.path.exists("labeled_sample.jsonl") or args.overwrite:
        sample["predicted_genre"] = None
        for ix, row in tqdm(sample.iterrows()):
            title = row["Title"]
            director = row["Director"]
            plot = row["Plot"]
            genre = ai_classify(args.model, title, director, plot)
            sample.at[ix, "predicted_genre"] = genre.genres
        sample.to_json("labeled_sample.jsonl", lines=True, orient="records")
    else:
        sample = pd.read_json("labeled_sample.jsonl", lines=True, orient="records")
    y_true = sample["genre_label"].tolist()
    y_pred = sample["predicted_genre"].tolist()
    metrics = multilabel_metrics(y_true, y_pred)
    metrics_report = f"""
Overall precision: {metrics["precision"]}
Overall recall: {metrics["recall"]}
"""
    for genre in GENRES:
        genre_metrics = metrics["by_genres"][genre]
        genre_report = f"""
{genre} precision: {genre_metrics["precision"]}
{genre} recall: {genre_metrics["recall"]}
{genre} total predicted: {genre_metrics["total_pred"]}
{genre} total true: {genre_metrics["total_true"]}
"""
        print(genre_report)
    print(metrics_report)

if __name__ == "__main__":
    main([
        # "--model", "gpt-5.4-mini",
        # "--model", "ollama_chat/llama3.2:3b",
        # "--model", "ollama_chat/deepseek-r1:8b",
        # "--model", "ollama_chat/phi4-mini:3.8b",
        # "--model", "ollama_chat/qwen3:8b",
        # "--model", "ollama_chat/deepseek-r1:1.5b",
        # "--model", "ollama_chat/llama3.2:1b",
        "--model", "ollama_chat/quant-theoros",
        "--overwrite"
    ])
