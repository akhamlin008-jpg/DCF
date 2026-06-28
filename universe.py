"""universe.py — single source of truth for the ticker list, group weights, and
theme, so App.py, the Factor page, and the Risk page all share ONE universe (and
therefore ONE cache entry) instead of three divergent lists.

The previous list contained ~160 entries that were really ~45 unique tickers
(UNH/CVX/MA/ORCL/COST etc. repeated a dozen times each). load_universe fetched
every duplicate, multiplying Yahoo requests ~3.5x for zero benefit. The raw list
below is de-duplicated, order-preserving, at import time.
"""

# Edit this list. Duplicates are harmless (removed automatically). Use Yahoo's
# punctuation for class shares: BRK-B, not BRK.B. (EDGAR/Stooq want their own
# forms; the data layer maps them.)
_RAW = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "TSM", "AVGO", "TSLA", "META", "MU",
    "LLY", "BRK-B", "WMT", "JPM", "AMD",
    "ASML", "INTC", "V", "JNJ", "XOM",
    "ORCL", "LRCX", "CSCO", "ABBV", "MA",
    "COST", "BAC", "UNH", "GE", "ARM",
    "KO", "HD", "PG", "CVX", "MS",
    "KLAC", "HSBC", "MRK", "GS", "NFLX",
    # Semis/storage names from the old App.py list — uncomment to include:
    # "SNDK", "WDC", "STX", "PANW", "NOW",
]

# Order-preserving de-dupe. dict.fromkeys keeps first occurrence, drops repeats.
TICKERS = list(dict.fromkeys(t.strip().upper() for t in _RAW if t.strip()))

# Non-US filers (ADRs / foreign private issuers) that EDGAR does NOT cover with
# us-gaap companyfacts. The data layer keeps Yahoo as the fundamentals source
# for these and labels them so the honesty ledger stays accurate.
NON_US_EDGAR = {"TSM", "ASML", "HSBC", "ARM"}

NVIDIA_GREEN = "#76B900"

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background:
  radial-gradient(1200px 600px at 80% -10%, #14210a 0%, transparent 55%), #0a0a0a; }}
#MainMenu, footer {{ visibility: hidden; }}
.hero {{ padding: 8px 0 4px 0; border-bottom: 1px solid #1f1f1f; margin-bottom: 14px; }}
.hero h1 {{ font-weight: 800; letter-spacing: -0.5px; font-size: 2.0rem; margin: 0; color: #f4f4f4; }}
.hero h1 span {{ color: {NVIDIA_GREEN}; }}
.hero p {{ color: #9a9a9a; margin: 4px 0 0 0; font-size: 0.92rem; }}
div[data-testid="stMetricValue"] {{ font-weight: 800; }}
.stButton button {{ background:{NVIDIA_GREEN}; color:#0a0a0a; border:none; font-weight:700; border-radius:8px; }}
.cov-good {{ color:{NVIDIA_GREEN}; font-weight:700; }}
.cov-mid  {{ color:#e0b000; font-weight:700; }}
.cov-bad  {{ color:#ff6b6b; font-weight:700; }}
.note {{ color:#9a9a9a; font-size:0.84rem; }}
.fresh-ok   {{ color:{NVIDIA_GREEN}; font-weight:700; }}
.fresh-stale{{ color:#e0b000; font-weight:700; }}
.fresh-old  {{ color:#ff6b6b; font-weight:700; }}
</style>
"""


def hero(title_html, subtitle):
    return f'<div class="hero"><h1>{title_html}</h1><p>{subtitle}</p></div>'
