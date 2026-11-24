# NumPy Data Analyzer with Groq AI Insight

A modular Python CLI program that collects and validates numerical input from the user, performs statistical analysis and array operations using **NumPy**, and generates a plain-English explanation of the results using the **Groq API**.

## Features

- **Validated Input**: Accepts at least 10 numbers (comma- or space-separated), rejects invalid/non-numeric entries, stores them in a Python `list`, and converts them into a `numpy.ndarray`.
- **NumPy Statistics**: Computes Mean, Median, Maximum, Minimum, and Sum.
- **Array Operations**: Sorts values in ascending order and filters values strictly greater than the mean.
- **AI Insight**: Sends computed statistics to Groq (`llama-3.3-70b-versatile`) for a beginner-friendly summary.
- **Environment Security**: Loads `GROQ_API_KEY` from a `.env` file via `python-dotenv`.

## Setup & Usage

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and replace `your_groq_api_key_here` with your actual Groq API key.

4. **Run the program:**
   ```bash
   python data_analyzer.py
   ```
