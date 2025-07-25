from typer import Typer
from plantclef.ensemble import app as ensemble_app
from plantclef.classification import app as classification_app
from plantclef.preprocessing import app as preprocessing_app
from plantclef.retrieval import app as retrieval_app
from plantclef.masking import app as masking_app
from plantclef.morph.workflow import app as morph_app
from plantclef.workflow import app as workflow_app

app = Typer()
app.add_typer(ensemble_app, name="ensemble")
app.add_typer(classification_app, name="classification")
app.add_typer(preprocessing_app, name="preprocessing")
app.add_typer(retrieval_app, name="retrieval")
app.add_typer(masking_app, name="masking")
app.add_typer(morph_app, name="morph")
app.add_typer(workflow_app, name="workflow")
