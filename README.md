# ToolPilot AI

ToolPilot AI is a Streamlit chatbot that uses Groq function calling to route requests to a small, explicit set of Python tools:

- Safe arithmetic calculator
- Current local time for an IANA timezone
- Current weather via Open-Meteo
- Stock quotes via `yfinance` (may be delayed; not financial advice)

The agent only executes registered tools. It does not execute shell commands, arbitrary Python, or arbitrary user-supplied URLs. Tool calls and results are shown in the app's execution trace.

## Requirements

- Python 3.10 or newer
- A Groq API key for general agent chat, calculator, time, and stock requests
- Network access for Groq, Open-Meteo weather, and Yahoo Finance data

Simple weather questions are routed directly to Open-Meteo and do not require a Groq API key.

## Install and run (Windows PowerShell)

From the folder containing `tool_agent.py`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements1.txt
```

Create a `.env` file beside `tool_agent.py`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

Keep `.env` private. Do not commit API keys. The repository does not include a real key.

Start the app:

```powershell
streamlit run tool_agent.py
```

## Available tool behavior

- Weather questions accept city names and common short forms such as `hyd` for Hyderabad. Comparisons such as “Compare the current weather in Hyderabad and Chennai” fetch each location separately.
- Calculator inputs are limited to basic arithmetic with bounded numbers and exponents.
- Timezone names use the IANA database, for example `Asia/Kolkata` or `UTC`.
- Stock tickers are validated before lookup. Market data can be delayed and is not financial advice.
- Groq model calls use the OpenAI-compatible Chat Completions API. The default model is `openai/gpt-oss-120b`; set `GROQ_MODEL` to another Groq model that supports tool use if needed.

The app includes local self-tests in the **Self-tests** tab. Weather and stock lookups depend on their external providers being reachable.
