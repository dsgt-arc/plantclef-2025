import typer
from .workflow import main as workflow_main
from .aggregation import main as aggregation_main
from .naive_baseline import main as naive_baseline_main

app = typer.Typer()
app.command("workflow")(workflow_main)
app.command("aggregation")(aggregation_main)
app.command("naive_baseline")(naive_baseline_main)
