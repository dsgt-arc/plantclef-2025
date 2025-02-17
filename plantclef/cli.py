from typer import Typer
from .etl import app as etl_app

app = Typer()
app.add_typer(etl_app, name="etl")
