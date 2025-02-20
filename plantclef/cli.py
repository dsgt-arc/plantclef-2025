from typer import Typer
from plantclef.embedding import app as embedding_app
from plantclef.preprocessing import app as preprocessing_app

app = Typer()
app.add_typer(embedding_app, name="embedding")
app.add_typer(preprocessing_app, name="preprocessing")
