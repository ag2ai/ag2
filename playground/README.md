# Planet-Satellite Playground

Working examples of the [planet-satellite architecture](../design/planet_satellite.md) across different domains.

## Quick Start

```bash
export OPENAI_API_KEY=sk-...

# No extra dependencies — just an OpenAI key
python playground/planet_satellite_lesson.py
python playground/planet_satellite_incident.py
python playground/planet_satellite_coding.py

# Requires pandas
pip install pandas
python playground/planet_satellite_data_analysis.py

# Requires Tavily API for real web search
pip install tavily-python beautifulsoup4 requests
export TAVILY_API_KEY=tvly-...
python playground/planet_satellite_competitive.py
python playground/planet_satellite_scientific.py
```

## Examples

| Example | Domain | Tools | Custom Satellite | Dependencies |
|---------|--------|-------|-----------------|--------------|
| [Coding Agent](planet_satellite_coding.py) | Software Engineering | `read_file`, `write_file`, `list_files`, `run_python` | — | — |
| [Data Analysis](planet_satellite_data_analysis.py) | Data Science | `inspect_data`, `run_analysis` (pandas) | — | pandas |
| [Incident Response](planet_satellite_incident.py) | DevOps / SRE | `query_logs`, `check_metrics`, `check_deployments`, `check_alerts` | — | — |
| [Lesson Plan](planet_satellite_lesson.py) | Education | — (pure LLM) | `DifficultyMonitor` | — |
| [Competitive Analysis](planet_satellite_competitive.py) | Business Strategy | `web_search` (Tavily) | `ClaimValidator` | tavily-python |
| [Document Review](planet_satellite_documents.py) | Legal / Compliance | `list_documents`, `read_document` | `CompletenessMonitor` | — |
| [Scientific Review](planet_satellite_scientific.py) | Research | `web_search`, `browse_url` | — | tavily-python, beautifulsoup4, requests |
| [Custom Satellite](planet_satellite_custom.py) | Extensibility | — | `KeywordWatchdog` | — |
| [Research Report](planet_satellite_research.py) | Research | — | — | — |
| [Code Review](planet_satellite_code_review.py) | Software Engineering | — | — | — |

### What each example demonstrates

**Coding Agent** — The planet plans a project (CLI expense tracker), delegates each module to a satellite, writes files to disk, and runs Python to verify. Pass an output directory as a CLI argument: `python playground/planet_satellite_coding.py ./my_project`.

**Data Analysis** — The planet explores a bundled [sales dataset](data/sales.csv) using real pandas, then delegates specialist analyses (statistics, trends, segments, anomalies) to satellites. Shows the pattern of *gather first, delegate second*.

**Incident Response** — The planet triages a simulated production outage using diagnostic tools that return realistic, interconnected data (a deployment triggers a query regression that cascades across services). Satellites investigate different root-cause hypotheses in parallel.

**Lesson Plan** — Pure LLM, no tools. The planet designs a recursion lesson for CS101 students, delegating content sections to satellites. A custom `DifficultyMonitor` satellite flags content that's too advanced or too basic for the target audience.

**Competitive Analysis** — Real web search via Tavily API. The planet researches the AI code assistant market, then delegates competitor deep-dives. A custom `ClaimValidator` satellite flags unsubstantiated statistics, nudging the planet to cite sources.

**Document Review** — The planet reads bundled [contract files](data/contracts/) and delegates extraction tasks (key terms, financials, risks, cross-document consistency). A `CompletenessMonitor` satellite flags if any document is omitted from the analysis.

**Scientific Review** — The most tool-intensive example. The planet runs multiple web searches, reads full articles with `browse_url`, then delegates thematic analysis to satellites. Demonstrates the search → browse → delegate pipeline.

**Custom Satellite** — Shows how to build a satellite plug-in. A `KeywordWatchdog` monitors output for watched terms and flags them for review.

## Bundled Data

```
data/
├── sales.csv                        # 60-row synthetic sales dataset
└── contracts/
    ├── saas_subscription.txt        # SaaS agreement
    ├── data_processing.txt          # DPA
    └── consulting_services.txt      # Consulting SOW
```

## Shared Tools

[`tools/web.py`](tools/web.py) provides `web_search` and `browse_url` — used by the competitive analysis and scientific review examples. Both are thin wrappers (~20 lines each) around Tavily and BeautifulSoup.
