import typer
from .workflow import main as workflow_main
from .mask.workflow import filter_by_mask

app = typer.Typer()
app.command("workflow")(workflow_main)
app.command("filter-by-mask")(filter_by_mask)
