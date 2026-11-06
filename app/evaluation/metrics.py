
class SmartRAGMetrics:
    def __init__(self):
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_calls = 0
        self.total_processing_time = 0.0

        self.routing_counts = {
            "lightweight": 0,
            "standard": 0,
            "advanced": 0,
            "cache": 0,
        }

    def record_query(self, result: dict):
        """Record the metrics for one completed query."""

        self.total_queries += 1

        if result.get("cache_hit", False):
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if result.get("llm_called", False):
            self.llm_calls += 1

        self.total_processing_time += result.get("total_time", 0.0)

        route = result.get("route")

        if route in self.routing_counts:
            self.routing_counts[route] += 1

    def display_metrics(self):
        """Display collected performance metrics."""

        print("\n========== SmartRAG Performance Metrics ==========")

        print(f"Total queries: {self.total_queries}")
        print(f"Cache hits: {self.cache_hits}")
        print(f"Cache misses: {self.cache_misses}")
        print(f"LLM calls: {self.llm_calls}")

        if self.total_queries > 0:
            cache_hit_rate = (
                self.cache_hits / self.total_queries
            ) * 100

            average_time = (
                self.total_processing_time / self.total_queries
            )

            print(f"Cache hit rate: {cache_hit_rate:.2f}%")
            print(
                f"Average processing time: "
                f"{average_time:.4f} seconds"
            )

        print("\n--- Routing Distribution ---")

        for route, count in self.routing_counts.items():
            print(f"{route}: {count}")

        print("==================================================")


# One shared metrics collector for this running application
metrics = SmartRAGMetrics()
