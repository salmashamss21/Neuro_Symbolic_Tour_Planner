import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tourplan import csp as C            # noqa: E402
from tourplan import data as D           # noqa: E402
from tourplan import nn                  # noqa: E402


@pytest.fixture(scope="session")
def city():
    return D.get_city("Edin")


@pytest.fixture(scope="session")
def models(city):
    return nn.train_all(city, 0)


@pytest.fixture(scope="session")
def bundle(models):
    return models["bundles"]["NN"]


@pytest.fixture(scope="session")
def prior(city, models):
    return C.category_prior(city, models["data"]["tr_t"])


@pytest.fixture(scope="session")
def parks(city):
    return [int(i) for i in city.pois.index[city.pois.category == "Park"]]
