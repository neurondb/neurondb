"""
WebSocket client for streaming responses
"""

import json
import logging
import threading
from typing import Callable, Optional, Dict, Any
import websocket

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    WebSocket client for streaming agent responses
    
    Usage:
        client = WebSocketClient(base_url, api_key)
        client.stream_message(
            session_id="...",
            content="Hello",
            on_message=lambda data: print(data['content'])
        )
    """
    
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize WebSocket client
        
        Args:
            base_url: Base URL (e.g., "http://localhost:8080")
            api_key: API key for authentication
        """
        # Convert http:// to ws://
        ws_base = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.base_url = ws_base.rstrip('/')
        self.api_key = api_key
        self._ws = None
        self._running = False
    
    def stream_message(
        self,
        session_id: str,
        content: str,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_complete: Optional[Callable[[], None]] = None
    ) -> None:
        """
        Stream a message and receive responses
        
        Args:
            session_id: Session ID
            content: Message content
            on_message: Callback for each message chunk (data: Dict)
            on_error: Callback for errors (error: Exception)
            on_complete: Callback when stream completes
        """
        url = f"{self.base_url}/ws?session_id={session_id}"
        
        def on_open(ws):
            logger.debug(f"WebSocket connected: {url}")
            # Send the message
            ws.send(json.dumps({'content': content}))
        
        def on_message_ws(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                
                if data.get('complete'):
                    if on_complete:
                        on_complete()
                    ws.close()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                if on_error:
                    on_error(e)
        
        def on_error_ws(ws, error):
            logger.error(f"WebSocket error: {error}")
            if on_error:
                on_error(error)
        
        def on_close(ws, close_status_code, close_msg):
            logger.debug("WebSocket closed")
            self._running = False
        
        self._running = True
        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message_ws,
            on_error=on_error_ws,
            on_close=on_close,
            header=[f"Authorization: Bearer {self.api_key}"]
        )
        
        # Run in blocking mode
        self._ws.run_forever()
    
    def close(self) -> None:
        """Close WebSocket connection"""
        if self._ws:
            self._ws.close()
        self._running = False

