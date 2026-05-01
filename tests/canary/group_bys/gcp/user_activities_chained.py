from joins.gcp import demo_parent

from ai.chronon.types import Aggregation, EnvironmentVariables, GroupBy, Operation, TimeUnit, Window

"""
Chained GroupBy that effectively enriches the last n listings the user interacted with to
include listing price information.
"""
chained_user_gb = GroupBy(
    sources=[demo_parent.upstream_join_source],
    keys=["user_id"],
    online=True,
    version=0,
    aggregations=[
        Aggregation(
            input_column="price_cents",
            operation=Operation.LAST_K(100),
            windows=[Window(length=7, time_unit=TimeUnit.DAYS)],
        ),
    ],
    env_vars=EnvironmentVariables(
        common={
            "CHRONON_ONLINE_ARGS": "-Ztasks=1",
        }
    ),
)
