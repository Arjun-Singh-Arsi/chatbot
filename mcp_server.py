from datetime import datetime
import platform
import requests
import pandas as pd
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with a professional name
mcp = FastMCP("Resume Portfolio Agent")

@mcp.tool()
def get_crypto_price(coin_id: str = "bitcoin", currency: str = "usd") -> str:
    """
    Fetch the real-time price of a cryptocurrency.
    Args:
        coin_id: The ID of the coin (e.g., 'bitcoin', 'ethereum', 'dogecoin').
        currency: The target currency (e.g., 'usd', 'eur').
    """
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={currency}"
    try:
        # CoinGecko is a free public API requiring no key for simple requests
        response = requests.get(url, timeout=10)
        data = response.json()
        if coin_id in data:
            price = data[coin_id].get(currency)
            return f"💰 The price of {coin_id} is {price} {currency.upper()}."
        else:
            return f"❌ Could not find data for coin: {coin_id}"
    except Exception as e:
        return f"Error fetching crypto price: {str(e)}"

@mcp.tool()
def check_website_status(url: str) -> str:
    """
    Check if a website is accessible and measure its response latency.
    Args:
        url: The full URL to check (e.g., 'https://www.google.com').
    """
    if not url.startswith("http"):
        url = "https://" + url
        
    try:
        response = requests.get(url, timeout=5)
        status = response.status_code
        latency = round(response.elapsed.total_seconds() * 1000, 2)
        
        if 200 <= status < 300:
            return f"✅ {url} is UP (Status: {status}). Latency: {latency}ms."
        else:
            return f"⚠️ {url} returned status code {status}."
    except Exception as e:
        return f"❌ {url} is DOWN. Error: {str(e)}"

@mcp.tool()
def analyze_csv_dataset(file_path: str) -> str:
    """
    Perform a quick data analysis on a local CSV file.
    Returns the first 5 rows and statistical summary (mean, min, max).
    Args:
        file_path: Absolute or relative path to the .csv file.
    """
    try:
        df = pd.read_csv(file_path)
        rows = len(df)
        columns = list(df.columns)
        
        # Get basic statistics for numeric columns
        stats = df.describe().to_string()
        head = df.head(3).to_string()
        
        return (
            f"📊 **Dataset Analysis**\n"
            f"- Rows: {rows}\n"
            f"- Columns: {columns}\n\n"
            f"**Preview (First 3 rows):**\n{head}\n\n"
            f"**Statistical Summary:**\n{stats}"
        )
    except FileNotFoundError:
        return f"❌ File not found: {file_path}. Please check the path."
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"

@mcp.tool()
def get_system_info() -> str:
    """
    Retrieve details about the server's operating environment.
    Useful for debugging or checking deployment environment.
    """
    info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.machine(),
        "Python Version": platform.python_version(),
        "Processor": platform.processor()
    }
    return "\n".join([f"💻 {k}: {v}" for k, v in info.items()])

if __name__ == "__main__":
    mcp.run()
