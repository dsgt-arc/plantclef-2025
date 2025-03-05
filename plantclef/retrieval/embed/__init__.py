import typer
from .workflow import embed, embed_with_mask
from .mask.workflow import filter_by_mask

app = typer.Typer()
app.command("workflow")(embed)
app.command("filter-by-mask")(filter_by_mask)
app.command("workflow-with-mask")(embed_with_mask)
