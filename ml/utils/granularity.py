"""
Granularity enum and configuration for multi-timeframe forecasting.
"""

from enum import Enum
from typing import NamedTuple


class GranularityConfig(NamedTuple):
    """Configuration for a specific granularity level."""
    code: str              # Short code (H, D, W, M, Y)
    name: str              # Human-readable name
    pandas_freq: str       # Pandas frequency string for resampling
    default_horizon: int   # Default forecast horizon in periods
    default_test_periods: int  # Default test set size in periods
    folder_name: str       # Output folder name


class Granularity(Enum):
    """Supported forecast granularities."""
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    YEARLY = "Y"

    @classmethod
    def from_code(cls, code: str) -> "Granularity":
        """Get Granularity enum from code string."""
        code = code.upper()
        for g in cls:
            if g.value == code:
                return g
        raise ValueError(f"Unknown granularity code: {code}. Valid codes: {[g.value for g in cls]}")

    @property
    def config(self) -> GranularityConfig:
        """Get configuration for this granularity."""
        return GRANULARITY_CONFIG[self]


# Configuration for each granularity level
GRANULARITY_CONFIG = {
    Granularity.HOURLY: GranularityConfig(
        code="H",
        name="hourly",
        pandas_freq="h",
        default_horizon=24,
        default_test_periods=168,  # 7 days * 24 hours
        folder_name="hourly",
    ),
    Granularity.DAILY: GranularityConfig(
        code="D",
        name="daily",
        pandas_freq="D",
        default_horizon=7,
        default_test_periods=7,  # 7 days
        folder_name="daily",
    ),
    Granularity.WEEKLY: GranularityConfig(
        code="W",
        name="weekly",
        pandas_freq="W-MON",  # Week starting Monday
        default_horizon=4,
        default_test_periods=4,  # 4 weeks
        folder_name="weekly",
    ),
    Granularity.MONTHLY: GranularityConfig(
        code="M",
        name="monthly",
        pandas_freq="MS",  # Month start
        default_horizon=3,
        default_test_periods=3,  # 3 months
        folder_name="monthly",
    ),
    Granularity.YEARLY: GranularityConfig(
        code="Y",
        name="yearly",
        pandas_freq="YS",  # Year start
        default_horizon=1,
        default_test_periods=1,  # 1 year
        folder_name="yearly",
    ),
}


def get_all_granularities() -> list[dict]:
    """Get list of all granularities with their configurations."""
    return [
        {
            "code": g.value,
            "name": g.config.name,
            "default_horizon": g.config.default_horizon,
            "default_test_periods": g.config.default_test_periods,
        }
        for g in Granularity
    ]
