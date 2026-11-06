
from app.evaluation.metrics import SmartRAGMetrics


metrics = SmartRAGMetrics()

# Simulate three SmartRAG results
test_results = [
    {
        "cache_hit": False,
        "llm_called": True,
        "total_time": 2.5,
        "route": "lightweight",
    },
    {
        "cache_hit": True,
        "llm_called": False,
        "total_time": 0.03,
        "route": "cache",
    },
    {
        "cache_hit": False,
        "llm_called": True,
        "total_time": 3.2,
        "route": "advanced",
    },
]

for result in test_results:
    metrics.record_query(result)

metrics.display_metrics()
