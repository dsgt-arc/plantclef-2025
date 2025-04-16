import pytest
from plantclef.spark import spark_resource


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("spark_data")
    with spark_resource(cores=2, local_dir=tmp_path.as_posix()) as spark:
        yield spark
