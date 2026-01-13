"""
Integration tests for SDK (require running NeuronAgent server)
"""

import pytest
import os
from neurondb_client import (
    NeuronAgentClient,
    AgentManager,
    SessionManager,
    ConversationManager,
    ToolManager,
    WorkflowManager,
    BudgetManager,
    EvaluationManager,
    ReplayManager,
    MemoryManager,
    WebhookManager,
    VFSManager,
    CollaborationManager
)


@pytest.fixture
def client():
    """Create test client"""
    base_url = os.getenv('NEURONAGENT_BASE_URL', 'http://localhost:8080')
    api_key = os.getenv('NEURONAGENT_API_KEY')
    
    if not api_key:
        pytest.skip("NEURONAGENT_API_KEY not set")
    
    return NeuronAgentClient(base_url=base_url, api_key=api_key)


@pytest.fixture
def agent_manager(client):
    """Create agent manager"""
    return AgentManager(client)


@pytest.fixture
def test_agent(agent_manager):
    """Create test agent"""
    agent = agent_manager.create(
        name=f"test-agent-{os.urandom(4).hex()}",
        system_prompt="You are a test agent.",
        model_name="gpt-4",
        enabled_tools=['sql']
    )
    yield agent
    # Cleanup
    try:
        agent_manager.delete(agent['id'])
    except:
        pass


class TestIntegration:
    """Integration tests"""
    
    def test_health_check(self, client):
        """Test health check"""
        assert client.health_check() is True
    
    def test_create_and_get_agent(self, agent_manager, test_agent):
        """Test agent creation and retrieval"""
        agent = agent_manager.get(test_agent['id'])
        assert agent['id'] == test_agent['id']
        assert agent['name'] == test_agent['name']
    
    def test_list_agents(self, agent_manager):
        """Test listing agents"""
        agents = agent_manager.list()
        assert isinstance(agents, list)
    
    def test_create_session(self, client, test_agent):
        """Test session creation"""
        session_mgr = SessionManager(client)
        session = session_mgr.create(agent_id=test_agent['id'])
        assert 'id' in session
        assert session['agent_id'] == test_agent['id']
    
    def test_send_message(self, client, test_agent):
        """Test sending message"""
        session_mgr = SessionManager(client)
        session = session_mgr.create(agent_id=test_agent['id'])
        
        response = session_mgr.send_message(
            session_id=session['id'],
            content="Hello, test!"
        )
        assert 'response' in response or 'content' in response
    
    def test_conversation_manager(self, client, test_agent):
        """Test conversation manager"""
        conversation = ConversationManager(client, agent_id=test_agent['id'])
        conversation.start()
        
        response = conversation.send("Hello!")
        assert isinstance(response, str)
        
        history = conversation.get_history()
        assert len(history) > 0
        
        conversation.close()
    
    def test_tool_manager(self, client):
        """Test tool manager"""
        tool_mgr = ToolManager(client)
        tools = tool_mgr.list()
        assert isinstance(tools, list)
    
    def test_budget_manager(self, client, test_agent):
        """Test budget manager"""
        budget_mgr = BudgetManager(client)
        
        # Set budget
        budget = budget_mgr.set_budget(
            agent_id=test_agent['id'],
            budget_amount=1000.0,
            period_type="monthly"
        )
        assert 'id' in budget
        
        # Get budget
        budget_status = budget_mgr.get_budget(
            agent_id=test_agent['id'],
            period_type="monthly"
        )
        assert budget_status.get('budget_set') is True
    
    def test_memory_manager(self, client, test_agent):
        """Test memory manager"""
        memory_mgr = MemoryManager(client)
        
        # List chunks
        chunks = memory_mgr.list_chunks(agent_id=test_agent['id'])
        assert isinstance(chunks, list)
        
        # Search (if chunks exist)
        if len(chunks) > 0:
            results = memory_mgr.search(
                agent_id=test_agent['id'],
                query="test query",
                top_k=5
            )
            assert isinstance(results, list)
    
    def test_workflow_manager(self, client):
        """Test workflow manager"""
        workflow_mgr = WorkflowManager(client)
        schedules = workflow_mgr.list_schedules()
        assert isinstance(schedules, list)
    
    def test_evaluation_manager(self, client):
        """Test evaluation manager"""
        eval_mgr = EvaluationManager(client)
        
        # List tasks
        tasks = eval_mgr.list_tasks()
        assert isinstance(tasks, list)
        
        # List runs
        runs = eval_mgr.list_runs()
        assert isinstance(runs, list)
    
    def test_replay_manager(self, client, test_agent):
        """Test replay manager"""
        replay_mgr = ReplayManager(client)
        
        # Create session for snapshot
        session_mgr = SessionManager(client)
        session = session_mgr.create(agent_id=test_agent['id'])
        
        # List snapshots
        snapshots = replay_mgr.list_by_session(session['id'])
        assert isinstance(snapshots, list)
    
    def test_webhook_manager(self, client):
        """Test webhook manager"""
        webhook_mgr = WebhookManager(client)
        webhooks = webhook_mgr.list()
        assert isinstance(webhooks, list)
    
    def test_vfs_manager(self, client, test_agent):
        """Test VFS manager"""
        vfs_mgr = VFSManager(client)
        
        # List files
        listing = vfs_mgr.list_files(agent_id=test_agent['id'])
        assert 'files' in listing or isinstance(listing, list)
    
    def test_collaboration_manager(self, client):
        """Test collaboration manager"""
        collab_mgr = CollaborationManager(client)
        # Note: May require workspace setup
        # This test verifies the manager can be instantiated
        assert collab_mgr is not None


