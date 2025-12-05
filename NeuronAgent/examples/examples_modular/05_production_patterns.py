#!/usr/bin/env python3
"""
Example 5: Production Patterns

Demonstrates production-ready patterns including:
- Error handling
- Retry logic
- Metrics collection
- Logging
- Resource cleanup
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurondb_client import (
    NeuronAgentClient,
    AgentManager,
    ConversationManager,
    AuthenticationError,
    NotFoundError,
    ServerError
)
from neurondb_client.utils.logging import setup_logging
from neurondb_client.utils.metrics import MetricsCollector

# Setup detailed logging
setup_logging(level="INFO")

def main():
    # Initialize with production settings
    client = NeuronAgentClient(
        max_retries=3,
        timeout=30,
        enable_logging=True
    )
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    try:
        # Health check with retry
        print(" Checking server health...")
        if not client.health_check():
            print("✗ Server is not healthy")
            return
        
        print("✓ Server is healthy")
        metrics.increment("health_checks", 1)
        
        # Create agent manager
        agent_mgr = AgentManager(client)
        
        # Create agent with error handling
        print("\n Creating agent...")
        try:
            agent = agent_mgr.create(
                name="production-example-agent",
                system_prompt="You are a production-ready assistant.",
                model_name="gpt-4",
                enabled_tools=['sql', 'http'],
                config={
                    'temperature': 0.7,
                    'max_tokens': 2000
                }
            )
            metrics.increment("agents_created", 1)
            print(f"✓ Agent created: {agent['id']}")
        except Exception as e:
            print(f"✗ Failed to create agent: {e}")
            metrics.increment("agent_errors", 1)
            return
        
        # Create conversation with error handling
        print("\n Starting conversation...")
        conversation = ConversationManager(
            client=client,
            agent_id=agent['id'],
            external_user_id="production-user-001"
        )
        
        try:
            conversation.start()
            print(f"✓ Conversation started")
            metrics.increment("conversations_started", 1)
            
            # Send messages with error handling
            messages = [
                "Hello! Can you help me?",
                "What are your capabilities?",
            ]
            
            for i, msg in enumerate(messages, 1):
                start_time = time.time()
                
                try:
                    print(f"\n Sending message {i}...")
                    response = conversation.send(msg)
                    
                    duration = time.time() - start_time
                    metrics.timer("message_duration", duration)
                    metrics.increment("messages_sent", 1)
                    
                    print(f"✓ Response received ({duration:.2f}s)")
                    print(f"   {response[:150]}...")
                    
                    # Track tokens if available
                    history = conversation.get_history()
                    if history:
                        last_msg = history[-1]
                        tokens = last_msg.get('tokens', 0)
                        if tokens > 0:
                            metrics.record("tokens_used", tokens)
                    
                except ServerError as e:
                    print(f"✗ Server error: {e}")
                    metrics.increment("message_errors", 1)
                    # Could implement retry logic here
                    break
                except Exception as e:
                    print(f"✗ Error: {e}")
                    metrics.increment("message_errors", 1)
                    break
                
                time.sleep(0.5)  # Rate limiting
            
            # Show conversation summary
            print("\n" + "=" * 60)
            print("Conversation Summary")
            print("=" * 60)
            history = conversation.get_history()
            print(f"Total exchanges: {len(history)}")
            print(f"Total tokens: {conversation.get_total_tokens()}")
            
        finally:
            # Always cleanup
            conversation.close()
            print("\n Conversation closed")
        
    except AuthenticationError as e:
        print(f"✗ Authentication failed: {e}")
        metrics.increment("auth_errors", 1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        metrics.increment("unexpected_errors", 1)
    finally:
        # Show metrics
        print("\n" + "=" * 60)
        print("Metrics Summary")
        print("=" * 60)
        summary = metrics.get_summary()
        
        print("\nCounters:")
        for name, value in summary['counters'].items():
            print(f"  {name}: {value}")
        
        print("\nTimers:")
        for name, stats in summary['timers'].items():
            print(f"  {name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Average: {stats['average']:.3f}s")
            print(f"    Min: {stats['min']:.3f}s")
            print(f"    Max: {stats['max']:.3f}s")
        
        # Show client metrics
        client_metrics = client.get_metrics()
        print("\nClient Metrics:")
        print(f"  Requests: {client_metrics['requests']}")
        print(f"  Errors: {client_metrics['errors']}")
        print(f"  Error rate: {client_metrics['error_rate']:.2%}")
        print(f"  Avg request time: {client_metrics['average_request_time']:.3f}s")
        
        # Cleanup
        client.close()
        print("\n✓ Example completed!")

if __name__ == "__main__":
    main()

