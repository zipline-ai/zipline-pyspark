from group_bys.azure import user_activities_chained
from joins.azure import demo_parent

from ai.chronon.types import Join, JoinPart

"""
Final Join in a chaining scenario that uses as its left source the
demo_parent Join and enriches it with the chained_user_gb GroupBy.
"""
downstream_join = Join(
    left=demo_parent.source,
    row_ids=["event_id"],
    right_parts=[
        JoinPart(
            group_by=user_activities_chained.chained_user_gb,
        ),
    ],
    version=0,
    online=True,
    output_namespace="data",
)
