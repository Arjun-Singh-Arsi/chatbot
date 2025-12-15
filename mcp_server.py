import random
import time
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("My Local Tools")

@mcp.tool()
def get_random_number(min: int, max: int) -> int:
    """Generate a random number between min and max."""
    return random.randint(min, max)

@mcp.tool()
def get_server_time() -> str:
    """Get the current server time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    mcp.run()
