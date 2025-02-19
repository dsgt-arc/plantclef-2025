from typer import Typer
from plantclef.embedding import app as embedding_app

app = Typer()
app.add_typer(embedding_app, name="embedding")
