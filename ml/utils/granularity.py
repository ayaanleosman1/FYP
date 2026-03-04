from enum import Enum
from typing import NamedTuple


class GranularityConfig(NamedTuple):
    code: str
    name: str
    pandas_freq: str
    default_horizon: int
    default_test_periods: int
    folder_name: str


class Granularity(Enum):
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    YEARLY = "Y"

    @classmethod
    def from_code(cls, code):
        code = code.upper()
        for g in cls:
            if g.value == code:
                return g
        raise ValueError(f"Unknown granularity code: {code}. Valid codes: {[g.value for g in cls]}")

    @property
    def config(self):
        return GRANULARITY_CONFIG[self]


GRANULARITY_CONFIG = {
    Granularity.HOURLY: GranularityConfig(
        code="H",
        name="hourly",
        pandas_freq="h",
        default_horizon=24,
        default_test_periods=168,  # 7 days
        folder_name="hourly",
    ),
    Granularity.DAILY: GranularityConfig(
        code="D",
        name="daily",
        pandas_freq="D",
        default_horizon=7,
        default_test_periods=30,
        folder_name="daily",
    ),
    Granularity.WEEKLY: GranularityConfig(
        code="W",
        name="weekly",
        pandas_freq="W-MON",
        default_horizon=4,
        default_test_periods=12,
        folder_name="weekly",
    ),
    Granularity.MONTHLY: GranularityConfig(
        code="M",
        name="monthly",
        pandas_freq="MS",
        default_horizon=3,
        default_test_periods=12,
        folder_name="monthly",
    ),
    Granularity.YEARLY: GranularityConfig(
        code="Y",
        name="yearly",
        pandas_freq="YS",
        default_horizon=1,
        default_test_periods=5,
        folder_name="yearly",
    ),
}


def get_all_granularities():
    return [
        {
            "code": g.value,
            "name": g.config.name,
            "default_horizon": g.config.default_horizon,
            "default_test_periods": g.config.default_test_periods,
        }
        for g in Granularity
    ]
