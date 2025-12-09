from adaptivegraph import LearnableEdge
import random

# Mock "LangGraph" style usage
# In a real app, this would be part of a StateGraph

def main():
    print("Welcome to AdaptiveGraph Demo")
    
    # Define an edge that chooses between 'slow_accurate' and 'fast_cheap'
    # Scenario: Users with 'vip' status need accuracy. others need speed.
    
    edge = LearnableEdge(
        options=["slow_accurate", "fast_cheap"],
        policy="linucb",
        exploration_alpha=0.5
    )
    
    # Simulation Loop
    for i in range(50):
        # 1. User Input comes in
        user_type = "vip" if i % 2 == 0 else "guest"
        query = f"User {user_type} request {i}"
        
        # 2. Edge decides routing
        # Note: We pass the query or state object to the edge
        # The edge embeds it automatically
        route = edge(query)
        
        # 3. Simulate execution & outcome
        # Let's say:
        # - VIPs need "slow_accurate" -> Reward 1.0, otherwise 0.0
        # - Guests need "fast_cheap" -> Reward 1.0, otherwise 0.0
        
        reward = 0.0
        if user_type == "vip" and route == "slow_accurate":
            reward = 1.0
        elif user_type == "guest" and route == "fast_cheap":
            reward = 1.0
            
        # 4. Feedback
        edge.record_feedback(result={}, reward=reward)
        
        print(f"Request: {user_type} -> Route: {route} -> Reward: {reward}")

    print("\nTraining complete. Testing VIP again:")
    print(f"VIP -> {edge('vip request')}")
    print(f"Guest -> {edge('guest request')}")

if __name__ == "__main__":
    main()
