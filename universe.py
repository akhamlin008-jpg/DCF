"""universe.py — single source of truth for the ticker list and group weights,
so the Factor and Risk pages don't import App.py (which would execute the whole
DCF Streamlit script on import). Point App.py's TICKERS at this too if you want
one list to rule them all."""

TICKERS = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "SPCX", "TSM", "AVGO", "TSLA", "META",
    "MU", "LLY", "BRK.B", "WMT", "JPM",
    "AMD", "ASML", "INTC", "V", "JNJ",
    "XOM", "ORCL", "LRCX", "CSCO", "ABBV",
    "MA", "COST", "BAC", "UNH", "GE",
    "ARM", "KO", "HD", "PG", "CVX",
    "MS", "KLAC", "HSBC", "MRK", "GS",
    "NFLX", "UNH", "CVX", "MA", "ORCL",
    "COST", "BAC", "UNH", "GE", "ARM",
    "KO", "HD", "PG", "CVX", "MS",
    "KLAC", "HSBC", "MRK", "GS", "NFLX",
    "UNH", "CVX", "MA", "ORCL", "COST",
    "BAC", "UNH", "GE", "ARM", "KO",
    "HD", "PG", "CVX", "MS", "KLAC",
    "HSBC", "MRK", "GS", "NFLX", "UNH",
    "CVX", "MA", "ORCL", "COST", "BAC",
    "UNH", "GE", "ARM", "KO", "HD",
    "PG", "CVX", "MS", "KLAC", "HSBC",
    "MRK", "GS", "NFLX", "UNH", "CVX",
    "MA", "ORCL", "COST", "BAC", "UNH",
    "GE", "ARM", "KO", "HD", "PG",
    "CVX", "MS", "KLAC", "HSBC", "MRK",
    "GS", "NFLX", "UNH", "CVX", "MA",
    "ORCL", "COST", "BAC", "UNH", "GE",
    "ARM", "KO", "HD", "PG", "CVX",
    "MS", "KLAC", "HSBC", "MRK", "GS",
    "NFLX", "UNH", "CVX", "MA", "ORCL",
    "COST", "BAC", "UNH", "GE", "ARM",
    "KO", "HD", "PG", "CVX", "MS",
    "KLAC", "HSBC", "MRK", "GS", "NFLX"
]

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
</style>
"""

def hero(title_html, subtitle):
    return f'<div class="hero"><h1>{title_html}</h1><p>{subtitle}</p></div>'
