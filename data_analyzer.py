import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv(override=True)


def get_numerical_array(min_count: int = 10) -> np.ndarray:
    """Prompt the user for at least `min_count` numbers, validate them,
    store them in a Python list, and return them as a NumPy array."""
    while True:
        raw_input = input(
            f"Enter at least {min_count} numbers (separated by spaces or commas): "
        ).strip()

        # Normalize commas to spaces so both "10, 20, ..." and "10 20 ..." work
        tokens = raw_input.replace(",", " ").split()

        if len(tokens) < min_count:
            print(
                f"Input error: Please enter at least {min_count} numbers "
                f"(you entered {len(tokens)}). Try again.\n"
            )
            continue

        number_list: list[float] = []
        invalid_tokens: list[str] = []

        for token in tokens:
            try:
                value = float(token)
                if np.isnan(value) or np.isinf(value):
                    invalid_tokens.append(token)
                else:
                    number_list.append(value)
            except ValueError:
                invalid_tokens.append(token)

        if invalid_tokens:
            print(
                f"Input error: Invalid number(s) detected: {', '.join(invalid_tokens)}. "
                "Please enter valid numerical values only.\n"
            )
            continue

        # Convert Python list into a NumPy array
        return np.array(number_list, dtype=float)


def analyze_data(data: np.ndarray) -> dict:
    """Calculate statistics and perform array operations using NumPy."""
    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    max_val = float(np.max(data))
    min_val = float(np.min(data))
    total_sum = float(np.sum(data))

    sorted_data = np.sort(data)
    above_mean = data[data > mean_val]

    return {
        "count": int(data.size),
        "mean": mean_val,
        "median": median_val,
        "max": max_val,
        "min": min_val,
        "sum": total_sum,
        "sorted_data": sorted_data,
        "above_mean": above_mean,
    }


def get_ai_insight(stats: dict) -> str:
    """Send calculated statistics to Groq API and request a short, plain-English summary."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "AI Insight skipped: GROQ_API_KEY was not found in your environment or .env file."
        )

    client = Groq(api_key=api_key)

    prompt = (
        "You are a helpful data tutor. Explain the following dataset statistics in 3 to 4 "
        "short sentences using simple, everyday English (avoid heavy jargon):\n"
        f"- Count of values: {stats['count']}\n"
        f"- Mean (average): {stats['mean']:.2f}\n"
        f"- Median (middle value): {stats['median']:.2f}\n"
        f"- Minimum: {stats['min']:.2f}\n"
        f"- Maximum: {stats['max']:.2f}\n"
        f"- Sum: {stats['sum']:.2f}\n"
        f"- Number of values above the mean: {len(stats['above_mean'])}"
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "Provide clear, brief, beginner-friendly statistical explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Could not retrieve AI insight due to an API error: {exc}"


def main() -> None:
    print("=== NumPy Data Analyzer with Groq AI Insight ===\n")

    data_array = get_numerical_array(min_count=10)
    results = analyze_data(data_array)

    print("\n--- Entered Data ---")
    print(data_array)

    print("\n--- Statistics ---")
    print(f"Mean:    {results['mean']:.2f}")
    print(f"Median:  {results['median']:.2f}")
    print(f"Maximum: {results['max']:.2f}")
    print(f"Minimum: {results['min']:.2f}")
    print(f"Sum:     {results['sum']:.2f}")

    print("\n--- Array Operations ---")
    print("Sorted (Ascending):", results["sorted_data"])
    if results["above_mean"].size > 0:
        print("Values Greater Than Mean:", results["above_mean"])
    else:
        print("Values Greater Than Mean: None (all values are equal)")

    print("\n--- AI Insight ---")
    insight = get_ai_insight(results)
    print(insight)


if __name__ == "__main__":
    main()
