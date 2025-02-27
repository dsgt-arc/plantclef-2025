import typer
from .create_test_subset import main as create_test_subset_main
from .create_top_species_subset import main as create_top_species_subset_main
from .test_to_parquet import main as test_to_parquet_main

app = typer.Typer()
app.command("create_test_subset")(create_test_subset_main)
app.command("create_top_species_subset")(create_top_species_subset_main)
app.command("test_to_parquet")(test_to_parquet_main)
