"""Utility functions for handling async operations in the CoAct agent."""

import asyncio
import concurrent.futures


def run_async_in_sync(coro):
    """Run an async coroutine in a sync context, handling existing event loops.
    
    This helper function addresses the issue where asyncio.run() cannot be called
    from within an already running event loop. It detects if there's an existing
    loop and handles the coroutine execution appropriately.
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine execution
    """
    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
        # If we're in an async context, create a task in a new thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop, we can use asyncio.run directly
        return asyncio.run(coro)
