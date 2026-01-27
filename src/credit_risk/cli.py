import typer
from credit_risk.train import train_pipeline
from credit_risk.score import score_pipeline

app = typer.Typer(no_args_is_help=True)

@app.command()
def train(input: str, artifacts: str = typer.Option(...)):
    """Train model + calibration bundle."""
    train_pipeline(input_path=input, artifacts_dir=artifacts)

@app.command()
def score(input: str, artifacts: str = typer.Option(...), output: str = typer.Option(...)):
    """Score a snapshot using an existing bundle."""
    score_pipeline(input_path=input, artifacts_dir=artifacts, output_path=output)

if __name__ == "__main__":
    app()
