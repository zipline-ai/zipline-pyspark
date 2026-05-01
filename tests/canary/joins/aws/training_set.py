from group_bys.aws import purchases

from ai.chronon.types import EventSource, Join, JoinPart, Query, selects

"""
This is the "left side" of the join that will comprise our training set. It is responsible for providing the primary keys
and timestamps for which features will be computed.
"""
source = EventSource(
    table="data.checkouts",
    query=Query(
        selects=selects("user_id"),  # The primary key used to join various GroupBys together
        time_column="ts",
    ),
)

v1_test = Join(
    left=source,
    row_ids="user_id",
    right_parts=[JoinPart(group_by=purchases.v1_test)],
    version=0,
)

v1_hub = Join(
    left=source,
    row_ids="user_id",
    right_parts=[JoinPart(group_by=purchases.v1_test)],
    version=0,
)

v1_dev = Join(
    left=source,
    row_ids="user_id",
    right_parts=[JoinPart(group_by=purchases.v1_dev)],
    version=0,
)
