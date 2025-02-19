import typer
from .workflow import main as workflow_main
from .hello_world import main as hello_world_main
from .test_workflow import main as test_workflow_main

app = typer.Typer()
app.command("workflow")(workflow_main)
app.command("hello_world")(hello_world_main)
app.command("test_workflow")(test_workflow_main)
