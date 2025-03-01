import typer
from .workflow import main as workflow_main

app = typer.Typer()
app.command("workflow")(workflow_main)