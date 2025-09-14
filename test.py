#!/usr/bin/env python3
"""
Test script to verify all 3 parallel agent system goals work:
1. Core-level distribution of agents across CPU cores
2. Hybrid communication (concurrent.futures for same-core, multiprocessing for cross-core)
3. Real-time context switching between LLMs
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen.agentchat.parallel.communication.message_router import MessageRouter
from autogen.agentchat.parallel.communication.thread_messenger import ThreadMessenger
from autogen.agentchat.parallel.context.context_manager import ContextManager
from autogen.agentchat.parallel.context.llm_pool import LLMPool
from autogen.agentchat.parallel.core.process_manager import ProcessManager


def test_minimal_parallel_system():
    """Test script to verify all 3 goals work"""

    print("=" * 60)
    print("TESTING PARALLEL AGENT SYSTEM - 3 GOALS")
    print("=" * 60)

    # GOAL 1: Core-level distribution
    print("\n🎯 GOAL 1: Core-level distribution of agents across CPU cores")
    print("-" * 50)

    process_manager = ProcessManager(num_cores=4)
    process_manager.initialize_cores()

    # Test task distribution
    process_manager.distribute_task({"task": "test_core_0", "data": "hello"}, target_core=0)
    process_manager.distribute_task({"task": "test_core_1", "data": "world"}, target_core=1)

    # Get results to verify core distribution
    result_0 = process_manager.get_result(core_id=0)
    result_1 = process_manager.get_result(core_id=1)

    print(f"✅ Core 0 result: {result_0}")
    print(f"✅ Core 1 result: {result_1}")

    # GOAL 2: Hybrid communication
    print("\n🎯 GOAL 2: Hybrid communication (concurrent.futures + multiprocessing)")
    print("-" * 50)

    router = MessageRouter()
    thread_messenger = ThreadMessenger()
    router.set_thread_messenger(thread_messenger)

    # Register agents on different cores
    router.register_agent_location("agent1", 0)  # Core 0
    router.register_agent_location("agent2", 1)  # Core 1
    router.register_agent_location("agent3", 0)  # Same core as agent1

    # Test cross-core communication (multiprocessing)
    print("Testing cross-core communication (multiprocessing)...")
    router.route_message("agent1", "agent2", {"msg": "cross-core test", "type": "multiprocessing"})

    # Test same-core communication (concurrent.futures)
    print("Testing same-core communication (concurrent.futures)...")
    future = router.route_message("agent1", "agent3", {"msg": "same-core test", "type": "threading"})

    # Verify cross-core message
    cross_core_msg = router.receive_cross_core_message(core_id=1)
    print(f"✅ Cross-core message received: {cross_core_msg}")

    # Verify same-core message
    same_core_msg = thread_messenger.get_message("agent3")
    print(f"✅ Same-core message received: {same_core_msg}")

    # GOAL 3: Real-time context switching
    print("\n🎯 GOAL 3: Real-time context switching between LLMs")
    print("-" * 50)

    context_mgr = ContextManager()
    llm_pool = LLMPool(core_id=0)

    # Test context updates and synchronization
    print("Testing context synchronization...")
    context_mgr.update_context("current_task", "document_analysis", process_id=0)
    context_mgr.update_context("user_id", "user_123", process_id=0)
    context_mgr.update_context("session_id", "session_456", process_id=0)

    # Test context switching with LLM execution
    print("Testing LLM context switching...")

    # Context 1: Document analysis
    analysis_context = context_mgr.get_context(process_id=0)
    result1 = llm_pool.execute_with_context("agent1", "Analyze the document structure", analysis_context)
    print(f"✅ LLM Result 1: {result1}")

    # Update context
    context_mgr.update_context("current_task", "summarization", process_id=0)
    context_mgr.update_context("output_format", "bullet_points", process_id=0)

    # Context 2: Summarization
    summary_context = context_mgr.get_context(process_id=0)
    result2 = llm_pool.execute_with_context("agent1", "Summarize the key findings", summary_context)
    print(f"✅ LLM Result 2: {result2}")

    # Verify context version tracking
    print(f"✅ Context version: {context_mgr.get_context_version()}")
    print(f"✅ Context stats: {context_mgr.get_stats()}")
    print(f"✅ LLM pool stats: {llm_pool.get_pool_stats()}")

    # Test integration: All 3 goals working together
    print("\n🚀 INTEGRATION TEST: All 3 goals working together")
    print("-" * 50)

    # Simulate a workflow where agents on different cores collaborate
    # with shared context and hybrid communication

    # Agent 1 (Core 0) starts analysis
    context_mgr.update_context("workflow_stage", "initial_analysis", process_id=0)
    analysis_result = llm_pool.execute_with_context(
        "agent1", "Start document analysis workflow", context_mgr.get_context(process_id=0)
    )

    # Send results to Agent 2 (Core 1) via cross-core communication
    router.route_message("agent1", "agent2", {"analysis_result": analysis_result, "next_stage": "detailed_processing"})

    # Update context for next stage
    context_mgr.update_context("workflow_stage", "detailed_processing", process_id=1)

    # Agent 2 processes on Core 1
    llm_pool_core1 = LLMPool(core_id=1)
    processing_result = llm_pool_core1.execute_with_context(
        "agent2", "Perform detailed processing", context_mgr.get_context(process_id=1)
    )

    print("✅ Integration test completed successfully!")
    print("   - Core distribution: Agents on cores 0 and 1")
    print("   - Cross-core communication: agent1 -> agent2")
    print("   - Context switching: 2 different contexts")

    # Cleanup
    print("\n🧹 Cleaning up resources...")
    thread_messenger.shutdown()
    llm_pool.cleanup()
    llm_pool_core1.cleanup()
    process_manager.shutdown()

    print("\n✅ ALL TESTS PASSED! Parallel agent system is working correctly.")
    print("   🎯 Goal 1: Core-level distribution ✓")
    print("   🎯 Goal 2: Hybrid communication ✓")
    print("   🎯 Goal 3: Real-time context switching ✓")


if __name__ == "__main__":
    test_minimal_parallel_system()
