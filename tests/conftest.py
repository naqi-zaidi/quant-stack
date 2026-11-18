from hypothesis import settings

# Derandomized so every CI run explores the same examples: test results reproduce exactly.
settings.register_profile("default", derandomize=True, deadline=None, max_examples=300)
settings.load_profile("default")
