"""
The LLAMA of WallStreet — Batch Sentiment Analysis Pipeline
============================================================
This script is the production version of the notebook pipeline.
It is designed to be submitted as a SLURM job on Leonardo HPC.

Local usage:
    python student_job_LLM.py

Leonardo usage (via SLURM):
    sbatch config/LLM_start_job.job
"""

import os
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from transformers import pipeline as hf_pipeline

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# Local dev: point to OpenAI
API_KEY    = os.environ.get("OPENAI_API_KEY", "your-openai-api-key-here")
BASE_URL   = None
MODEL_NAME = "gpt-4o-mini"

# Leonardo: uncomment these and comment the three lines above
# API_KEY    = "bUon34Bu3o#2"
# BASE_URL   = "http://127.0.0.1:8000/v1"
# MODEL_NAME = "google/gemma-4-31B-it"

INPUT_FILE  = "reddit_comments.csv"
OUTPUT_FILE = "reddit_sentiment.csv"
LIMIT       = 2_000   # set to None to process all ~100k rows on Leonardo
MAX_WORKERS = 8       # parallel LLM calls

# ── LLM setup ──────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    **(dict(base_url=BASE_URL) if BASE_URL else {})
)

# ── Pydantic schema for structured output ──────────────────────────────────────
class TickerExtraction(BaseModel):
    tickers: list[str]  # list of ticker symbols, e.g. ["AAPL", "TSLA"] or ["NA"]
    relevant: bool      # True if comment mentions any publicly traded company

EXTRACTION_SYSTEM_PROMPT = """
You are a financial analyst assistant. Your job is to read Reddit comments and determine
if they mention any publicly traded companies on a stock exchange (NYSE, NASDAQ, etc.).

Rules:
- Return ALL ticker symbols mentioned in the comment.
- If multiple companies are mentioned, return all of them.
- If no company is mentioned, set tickers to ["NA"] and relevant to False.

Examples:
[INPUT]: I bought Apple and I am also bullish on Tesla!
[OUTPUT]: tickers=["AAPL", "TSLA"], relevant=True

[INPUT]: The weather today is amazing.
[OUTPUT]: tickers=["NA"], relevant=False

[INPUT]: Tesla is going to crash, Elon is insane.
[OUTPUT]: tickers=["TSLA"], relevant=True
"""

extraction_chain = llm.with_structured_output(TickerExtraction)

# ── Functions ──────────────────────────────────────────────────────────────────
def read_data(input_file: str, limit: int | None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    if limit:
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True)


def extract_ticker(comment: str) -> TickerExtraction:
    """Ask the LLM to extract all tickers from a comment."""
    return extraction_chain.invoke([
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",   "content": comment}
    ])


def run_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Parallelized ticker extraction across all comments."""
    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(extract_ticker, row["comments"]): i
            for i, row in df.iterrows()
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = {"tickers": result.tickers, "relevant": result.relevant}
            except Exception as e:
                log.warning(f"Extraction failed for row {idx}: {e}")
                results[idx] = {"tickers": ["ERROR"], "relevant": False}
            done += 1
            if done % 100 == 0:
                log.info(f"Extraction progress: {done}/{len(df)}")

    extraction_df = pd.DataFrame(results, index=df.index)
    df = df.join(extraction_df)

    # Explode: one row per ticker (handles multiple tickers per comment)
    df = df[df["relevant"] == True].explode("tickers").rename(columns={"tickers": "ticker"})
    df = df[df["ticker"] != "NA"].reset_index(drop=True)
    return df


def score_to_int(label: str, score: float) -> int:
    """Map sentiment label + confidence to 5-level integer (-2 to +2)."""
    if label == "positive":
        return 2 if score >= 0.8 else 1
    elif label == "negative":
        return -2 if score >= 0.8 else -1
    return 0


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run local sentiment model on all relevant comments."""
    sentiment_model = hf_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
    )
    df = df.copy()
    df["sentiment"] = df["comments"].apply(
        lambda c: score_to_int(**{k: v for k, v in zip(["label", "score"], [sentiment_model(c)[0]["label"], sentiment_model(c)[0]["score"]])})
    )
    return df


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily min/max/avg sentiment per ticker."""
    return (
        df.groupby(["date", "ticker"])["sentiment"]
        .agg(avg_sentiment="mean", min_sentiment="min", max_sentiment="max", n_comments="count")
        .reset_index()
    )


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = datetime.now()
    log.info(f"Pipeline started at {t0}")
    log.info(f"Model: {MODEL_NAME} | Endpoint: {BASE_URL or 'OpenAI'} | Limit: {LIMIT}")

    # Wait for vLLM to be ready on Leonardo (no-op locally)
    if BASE_URL:
        log.info("Waiting 30s for vLLM endpoint to be ready...")
        time.sleep(30)

    # Step 1-2: Read data and extract tickers
    log.info("Reading data...")
    df = read_data(INPUT_FILE, LIMIT)
    log.info(f"Loaded {len(df):,} comments")

    log.info("Extracting tickers...")
    df_extracted = run_extraction(df)
    log.info(f"Relevant rows after extraction: {len(df_extracted)}")

    # Step 3: Sentiment scoring
    log.info("Scoring sentiment...")
    df_final = run_sentiment(df_extracted)

    # Step 4: Save
    df_final.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Saved {len(df_final)} rows to {OUTPUT_FILE}")

    # Step 5: Daily metrics
    daily = compute_daily_metrics(df_final)
    daily.to_csv("daily_metrics.csv", index=False)
    log.info(f"Saved daily metrics to daily_metrics.csv")

    t1 = datetime.now()
    log.info(f"Pipeline finished in {t1 - t0}")
