import typer
from .index import app as index_app
from .embed import app as embed_app
from .query import app as query_app
from .inference import app as inference_app

app = typer.Typer()
app.add_typer(index_app, name="index")
app.add_typer(embed_app, name="embed")
app.add_typer(query_app, name="query")
app.add_typer(inference_app, name="inference")
