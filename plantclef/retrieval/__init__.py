import typer
from .index import app as index_app
from .embed import app as embed_app

app = typer.Typer()
app.add_typer(index_app, name="index")
app.add_typer(embed_app, name="embed")