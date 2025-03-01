from typer import Typer
from plantclef.embedding import app as embedding_app
from plantclef.classification import app as classification_app
from plantclef.preprocessing import app as preprocessing_app
from plantclef.masking import app as masking_app
from plantclef.morph.workflow import app as morph_app

app = Typer()
app.add_typer(embedding_app, name="embedding")
app.add_typer(classification_app, name="classification")
app.add_typer(preprocessing_app, name="preprocessing")
app.add_typer(masking_app, name="masking")
app.add_typer(morph_app, name="morph")
