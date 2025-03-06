import typer
from .workflow import main as workflow_main
from .aggregation import main as aggregation_main

app = typer.Typer()
app.command("workflow")(workflow_main)
app.command("aggregation")(aggregation_main)
