from compressor.coverage import (
    extract_query_concepts,
    calculate_coverage,
    get_missing_concepts,
    coverage_passed,
)


query = (
    "What caused the increase in renewable "
    "energy production between 2020 and 2025?"
)


full_context = """
Renewable energy production increased significantly
between 2020 and 2025.

Solar generation increased by 42% during this period.

Government incentives and lower solar panel costs
were major contributors to this growth.

Wind energy production also increased during the period.
"""


partial_context = """
Solar generation increased by 42% during this period.
"""


print("\n===== COVERAGE GUARD TEST =====\n")


# ---------------------------------------------------------
# 1. Extract query concepts
# ---------------------------------------------------------

concepts = extract_query_concepts(
    query
)

print("QUERY CONCEPTS")
print("--------------")

print(
    sorted(concepts)
)


# ---------------------------------------------------------
# 2. Full context coverage
# ---------------------------------------------------------

full_coverage = calculate_coverage(
    query=query,
    compressed_context=full_context
)

print("\nFULL CONTEXT")
print("------------")

print(
    f"Coverage: "
    f"{full_coverage:.2%}"
)

print(
    f"Passed:   "
    f"{coverage_passed(full_coverage)}"
)


# ---------------------------------------------------------
# 3. Partial context coverage
# ---------------------------------------------------------

partial_coverage = calculate_coverage(
    query=query,
    compressed_context=partial_context
)

missing = get_missing_concepts(
    query=query,
    compressed_context=partial_context
)

print("\nPARTIAL CONTEXT")
print("---------------")

print(
    f"Coverage: "
    f"{partial_coverage:.2%}"
)

print(
    f"Passed:   "
    f"{coverage_passed(partial_coverage)}"
)

print(
    f"Missing:  "
    f"{missing}"
)


print("\n===============================\n")