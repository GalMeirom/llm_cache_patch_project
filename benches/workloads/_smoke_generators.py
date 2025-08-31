# benches/workloads/_smoke_generators.py
from generators import interleave_stream, prefixed, uniform_stream, zipf_stream

# Preview using throwaway streams
short_preview = list(
    prefixed(zipf_stream(total_keys=20000, unique_ratio=0.05, theta=1.2, count=10, seed=1), "S")
)
long_preview = list(
    prefixed(uniform_stream(total_keys=100000, unique_ratio=0.8, count=10, seed=2), "N")
)
print("SHORT:", short_preview)
print("LONG :", long_preview)

# Fresh streams for mixing (not yet consumed)
short = prefixed(zipf_stream(total_keys=20000, unique_ratio=0.05, theta=1.2, count=10, seed=1), "S")
long = prefixed(uniform_stream(total_keys=100000, unique_ratio=0.8, count=10, seed=2), "N")
mix = interleave_stream(short, long, ratio_a=0.7, seed=3)

print("MIX  :", [next(mix) for _ in range(20)])  # 10+10 available
