"""
neurondbpy CLI: generate, explain, optimize, pipeline commands.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from neurondbpy.llm.client import LLMSQLClient
from neurondbpy.pipeline.download import download_datasets
from neurondbpy.pipeline.preprocess import preprocess
from neurondbpy.pipeline.split import create_splits
from neurondbpy.pipeline.tokenizer import SQLTokenizer, prepare_training_data
from neurondbpy.pipeline.train import run_train
from neurondbpy.pipeline.serve import run_serve
from neurondbpy.pipeline.evaluate import SQLEvaluator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(verbose: bool) -> None:
    """neurondbpy: NeuronDB Python SDK and LLM SQL pipeline."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument("prompt", type=str)
@click.option("--base-url", default="http://localhost:8080", help="LLM SQL API base URL")
@click.option("--api-key", default=None, help="Optional API key")
@click.option("--dialect", default="postgresql", type=click.Choice(["postgresql", "mysql"]))
@click.option("--schema", default=None, help="Optional schema JSON or path to file")
def generate(prompt: str, base_url: str, api_key: Optional[str], dialect: str, schema: Optional[str]) -> None:
    """Generate SQL from natural language."""
    client = LLMSQLClient(base_url=base_url, api_key=api_key)
    schema_val = None
    if schema:
        if schema.startswith("{") or schema.startswith("["):
            schema_val = json.loads(schema)
        else:
            with open(schema) as f:
                schema_val = json.load(f)
    result = client.generate_sql(prompt=prompt, dialect=dialect, schema=schema_val)
    click.echo(result.sql)
    if result.explanation:
        click.echo("\nExplanation: " + result.explanation)


@main.command()
@click.argument("sql", type=str)
@click.option("--base-url", default="http://localhost:8080", help="LLM SQL API base URL")
@click.option("--api-key", default=None, help="Optional API key")
@click.option("--detail-level", default="detailed", type=click.Choice(["brief", "detailed", "expert"]))
def explain(sql: str, base_url: str, api_key: Optional[str], detail_level: str) -> None:
    """Explain a SQL query."""
    client = LLMSQLClient(base_url=base_url, api_key=api_key)
    out = client.explain_sql(sql=sql, detail_level=detail_level)
    click.echo(out)


@main.command()
@click.argument("sql", type=str)
@click.option("--base-url", default="http://localhost:8080", help="LLM SQL API base URL")
@click.option("--api-key", default=None, help="Optional API key")
def optimize(sql: str, base_url: str, api_key: Optional[str]) -> None:
    """Get optimized SQL and suggestions."""
    client = LLMSQLClient(base_url=base_url, api_key=api_key)
    result = client.optimize_sql(sql=sql)
    click.echo(result.optimized_sql)
    if result.explanation:
        click.echo("\nExplanation: " + result.explanation)
    for s in result.suggestions:
        click.echo("  - " + s)


@main.group()
def pipeline() -> None:
    """Pipeline: download, preprocess, split, train, serve, evaluate."""
    pass


@pipeline.command("download")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--datasets", multiple=True, type=click.Choice(["llmsql", "spider", "spider2", "postgres_docs", "github", "all"]), default=["all"])
def pipeline_download(output_dir: Path, datasets: tuple) -> None:
    """Download SQL training datasets."""
    download_datasets(output_dir=output_dir, datasets=list(datasets or ["all"]))


@pipeline.command("preprocess")
@click.option("--input-dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/processed"))
def pipeline_preprocess(input_dir: Path, output_dir: Path) -> None:
    """Preprocess raw datasets to unified JSONL."""
    preprocess(input_dir=input_dir, output_dir=output_dir)


@pipeline.command("split")
@click.option("--input-file", type=click.Path(path_type=Path), default=Path("data/processed/all_data.jsonl"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/splits"))
@click.option("--train-ratio", type=float, default=0.8)
@click.option("--val-ratio", type=float, default=0.1)
@click.option("--test-ratio", type=float, default=0.1)
@click.option("--seed", type=int, default=42)
def pipeline_split(input_file: Path, output_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> None:
    """Create train/validation/test splits."""
    create_splits(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


@pipeline.command("train")
@click.option("--config", "model_config", required=True, type=click.Path(exists=True), help="Model config YAML")
@click.option("--data-path", required=True, type=click.Path(exists=True), help="Path to train/ and validation/")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path), help="Checkpoint output directory")
@click.option("--deepspeed-config", required=True, type=click.Path(exists=True), help="DeepSpeed config JSON")
@click.option("--run-name", default="sql-llm-70b")
@click.option("--seed", type=int, default=42)
def pipeline_train(model_config: str, data_path: str, output_dir: Path, deepspeed_config: str, run_name: str, seed: int) -> None:
    """Run SQL LLM training (DeepSpeed)."""
    run_train(
        model_config=model_config,
        data_path=data_path,
        output_dir=str(output_dir),
        deepspeed_config=deepspeed_config,
        run_name=run_name,
        seed=seed,
    )


@pipeline.command("serve")
@click.option("--model-path", type=str, default=None, help="Path to model (or set MODEL_PATH)")
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=8000)
def pipeline_serve(model_path: Optional[str], host: str, port: int) -> None:
    """Start the SQL LLM model server (vLLM)."""
    proc = run_serve(model_path=model_path, host=host, port=port)
    click.echo(f"Server starting on {host}:{port} (PID {proc.pid}). Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    sys.exit(proc.returncode or 0)


@pipeline.command("evaluate")
@click.option("--predictions", required=True, type=click.Path(exists=True), help="JSONL with generated_sql")
@click.option("--ground-truth", "ground_truth", required=True, type=click.Path(exists=True), help="JSONL with sql")
@click.option("--output-file", type=click.Path(path_type=Path), default=Path("evaluation_results.json"))
@click.option("--db-host", default="localhost")
@click.option("--db-port", type=int, default=5433)
@click.option("--db-name", default="neurondb")
@click.option("--db-user", default="neurondb")
@click.option("--db-password", default="neurondb")
def pipeline_evaluate(
    predictions: str,
    ground_truth: str,
    output_file: Path,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Evaluate SQL predictions vs ground truth."""
    db_config = {
        "host": db_host,
        "port": db_port,
        "database": db_name,
        "user": db_user,
        "password": db_password,
    }
    evaluator = SQLEvaluator(db_config)
    try:
        results = evaluator.evaluate_dataset(predictions, ground_truth)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {output_file}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
