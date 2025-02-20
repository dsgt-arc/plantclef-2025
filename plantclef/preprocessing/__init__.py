import typer
from .create_top_species_subset import main as create_top_species_subset_main

app = typer.Typer()
app.command("create_top_species_subset")(create_top_species_subset_main)
