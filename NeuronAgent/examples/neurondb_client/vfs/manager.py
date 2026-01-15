"""
Virtual Filesystem module

Provides VFS operations:
- Create file
- Read file
- Write file
- Delete file
- List files
- Copy file
- Move file
"""

import logging
from typing import Dict, List, Optional
import base64

from ..core.client import NeuronAgentClient
from ..core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class VFSManager:
    """
    High-level virtual filesystem management
    
    Usage:
        client = NeuronAgentClient()
        manager = VFSManager(client)
        
        file = manager.create_file(
            agent_id="...",
            path="/documents/readme.txt",
            content="Hello World"
        )
    """
    
    def __init__(self, client: NeuronAgentClient):
        """
        Initialize VFS manager
        
        Args:
            client: NeuronAgentClient instance
        """
        self.client = client
    
    def create_file(
        self,
        agent_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        mime_type: str = "text/plain",
        encoding: str = "text"
    ) -> Dict:
        """
        Create a new file
        
        Args:
            agent_id: Agent UUID
            path: File path
            content: File content (text or base64 encoded)
            session_id: Optional session UUID
            mime_type: MIME type (default: text/plain)
            encoding: Content encoding ("text" or "base64")
        
        Returns:
            Created file dictionary
        """
        if not path:
            raise ValueError("path is required")
        if not content:
            raise ValueError("content is required")
        
        logger.info(f"Creating file: {path} for agent: {agent_id}")
        
        payload = {
            'path': path,
            'content': content,
            'mime_type': mime_type,
            'encoding': encoding
        }
        
        if session_id:
            payload['session_id'] = session_id
        
        file = self.client.post(
            f'/api/v1/agents/{agent_id}/files',
            json_data=payload
        )
        logger.info(f"File created: {file['id']}")
        return file
    
    def list_files(
        self,
        agent_id: str,
        path: str = "/",
        session_id: Optional[str] = None
    ) -> Dict:
        """
        List files in a directory
        
        Args:
            agent_id: Agent UUID
            path: Directory path (default: "/")
            session_id: Optional session UUID
        
        Returns:
            Directory listing dictionary with files
        """
        params = {'path': path}
        if session_id:
            params['session_id'] = session_id
        
        return self.client.get(
            f'/api/v1/agents/{agent_id}/files',
            params=params
        )
    
    def read_file(
        self,
        agent_id: str,
        path: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Read a file
        
        Args:
            agent_id: Agent UUID
            path: File path
            session_id: Optional session UUID
        
        Returns:
            File dictionary with content
        
        Raises:
            NotFoundError: If file not found
        """
        try:
            params = {}
            if session_id:
                params['session_id'] = session_id
            
            # URL encode the path
            encoded_path = path.replace('/', '%2F')
            return self.client.get(
                f'/api/v1/agents/{agent_id}/files/{encoded_path}',
                params=params
            )
        except NotFoundError:
            logger.error(f"File not found: {path}")
            raise
    
    def write_file(
        self,
        agent_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        mime_type: Optional[str] = None,
        encoding: str = "text"
    ) -> Dict:
        """
        Write/update a file
        
        Args:
            agent_id: Agent UUID
            path: File path
            content: File content
            session_id: Optional session UUID
            mime_type: Optional MIME type
            encoding: Content encoding ("text" or "base64")
        
        Returns:
            Updated file dictionary
        """
        logger.info(f"Writing file: {path} for agent: {agent_id}")
        
        payload = {
            'content': content,
            'encoding': encoding
        }
        
        if session_id:
            payload['session_id'] = session_id
        if mime_type:
            payload['mime_type'] = mime_type
        
        # URL encode the path
        encoded_path = path.replace('/', '%2F')
        file = self.client.put(
            f'/api/v1/agents/{agent_id}/files/{encoded_path}',
            json_data=payload
        )
        logger.info(f"File written: {path}")
        return file
    
    def delete_file(
        self,
        agent_id: str,
        path: str,
        session_id: Optional[str] = None
    ) -> None:
        """
        Delete a file
        
        Args:
            agent_id: Agent UUID
            path: File path
            session_id: Optional session UUID
        """
        logger.info(f"Deleting file: {path} for agent: {agent_id}")
        
        params = {}
        if session_id:
            params['session_id'] = session_id
        
        # URL encode the path
        encoded_path = path.replace('/', '%2F')
        self.client.delete(
            f'/api/v1/agents/{agent_id}/files/{encoded_path}',
            params=params
        )
        logger.info(f"File deleted: {path}")
    
    def copy_file(
        self,
        agent_id: str,
        source_path: str,
        dest_path: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Copy a file
        
        Args:
            agent_id: Agent UUID
            source_path: Source file path
            dest_path: Destination file path
            session_id: Optional session UUID
        
        Returns:
            Copied file dictionary
        """
        logger.info(f"Copying file: {source_path} -> {dest_path}")
        
        payload = {'dest_path': dest_path}
        if session_id:
            payload['session_id'] = session_id
        
        encoded_path = source_path.replace('/', '%2F')
        file = self.client.post(
            f'/api/v1/agents/{agent_id}/files/{encoded_path}/copy',
            json_data=payload
        )
        logger.info(f"File copied: {source_path} -> {dest_path}")
        return file
    
    def move_file(
        self,
        agent_id: str,
        source_path: str,
        dest_path: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Move a file
        
        Args:
            agent_id: Agent UUID
            source_path: Source file path
            dest_path: Destination file path
            session_id: Optional session UUID
        
        Returns:
            Moved file dictionary
        """
        logger.info(f"Moving file: {source_path} -> {dest_path}")
        
        payload = {'dest_path': dest_path}
        if session_id:
            payload['session_id'] = session_id
        
        encoded_path = source_path.replace('/', '%2F')
        file = self.client.post(
            f'/api/v1/agents/{agent_id}/files/{encoded_path}/move',
            json_data=payload
        )
        logger.info(f"File moved: {source_path} -> {dest_path}")
        return file



