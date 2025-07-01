"""Provide fixtures for pytest-based unit tests."""

from click.testing import CliRunner
import pytest

from record import Record 


@pytest.fixture
def config_dir(tmpdir):
    return str(tmpdir.mkdir('config'))


@pytest.fixture
def record(config_dir):
    return Record(config_dir=config_dir)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def record_df(datafiles):
    """Creates a Record object with datafiles in config directory."""
    return Record(config_dir=str(datafiles))
