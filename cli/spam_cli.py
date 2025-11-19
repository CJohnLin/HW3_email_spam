# cli/spam_cli.py
import click
import os
from src.train import train_all, train_single
from src.predict import predict_text
from src.evaluate import evaluate_on_csv
from src.utils import list_models

@click.group()
def cli():
    pass

@cli.command()
@click.option("--dataset", "-d", required=True, help="CSV dataset path (text,label)")
@click.option("--model", "-m", default="all", type=click.Choice(["all","lr","nb","svm"]))
@click.option("--models-dir", default="models")
def train(dataset, model, models_dir):
    """Train model(s)."""
    if model == "all":
        train_all(dataset, models_dir)
    else:
        df_path = dataset
        train_single(model, *__load_train_split(df_path), models_dir=models_dir)

def __load_train_split(dataset_path):
    # internal helper to reuse train_single signature
    from src.preprocessing import load_dataset, train_val_split
    df = load_dataset(dataset_path)
    return train_val_split(df)  # returns X_train_texts, X_val_texts, y_train, y_val

@cli.command()
@click.option("--models-dir", default="models")
def list(models_dir):
    """List saved model files."""
    for f in list_models(models_dir):
        click.echo(f)

@cli.command()
@click.option("--models-dir", default="models")
@click.option("--model", default="lr", type=click.Choice(["lr","nb","svm"]))
@click.option("--text", "-t", default="", help="Text to predict (if omitted, reads from stdin).")
def predict(models_dir, model, text):
    """Predict single text"""
    t = text or click.prompt("Enter text to classify")
    pred, score = predict_text(models_dir, model, t)
    click.echo(f"Model: {model} -> Prediction: {pred} (1=spam) score: {score}")

@cli.command()
@click.option("--models-dir", default="models")
@click.option("--model", default="lr", type=click.Choice(["lr","nb","svm"]))
@click.option("--csv", "csv_path", required=True, help="CSV with text,label for evaluation")
def evaluate(models_dir, model, csv_path):
    res = evaluate_on_csv(models_dir, model, csv_path)
    click.echo("Classification Report:\n")
    click.echo(res["report"])
    click.echo(f"\nConfusion Matrix:\n{res['confusion_matrix']}")
    click.echo(f"\nAUC: {res['auc']}")

if __name__ == "__main__":
    cli()
