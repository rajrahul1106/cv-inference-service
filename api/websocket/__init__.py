"""
WebSocket package. Holds the process-wide ConnectionManager singleton.

The singleton lives here (a neutral module) rather than in api/main.py so that
both the WS route (api/routes/websocket.py) and the background subscriber
(api/main.py) can import the *same* instance without a circular import — main
imports the route to mount it, and the route imports the manager from here.
"""

from api.websocket.connection_manager import ConnectionManager

# Shared by the WS route (connect/disconnect) and the subscriber (broadcast).
connection_manager = ConnectionManager()

__all__ = ["ConnectionManager", "connection_manager"]
