from adaptivegraph import LearnableEdge
import random

def main():
    print("Welcome to AdaptiveGraph Advanced Demo")
    
    # Use the new create() method for "batteries included"
    # This automatically sets up SentenceTransformerEmbedding and FaissExperienceStore
    try:
        edge = LearnableEdge.create(
            options=["support_ticket", "sales_lead", "spam"],
            embedding="sentence-transformers",
            memory="faiss",
            feature_dim=384, # common for miniLM
            exploration_alpha=0.5
        )
        print("Successfully created Edge with Sentence Transformers and FAISS!")
    except ImportError as e:
        print(f"Skipping execution: {e}")
        return

    # Simulation Loop
    queries = [
        ("I need help with my broken device", "support_ticket"),
        ("I want to buy 500 units", "sales_lead"),
        ("Cheap watches for sale", "spam")
    ]
    
    for i in range(10):
        query_text, expected = random.choice(queries)
        
        # 1. Routing
        route = edge(query_text)
        
        # 2. Reward
        reward = 1.0 if route == expected else -0.1
        
        # 3. Learn
        edge.record_feedback(result={}, reward=reward)
        
        print(f"Query: '{query_text}' -> Route: {route} (Expected: {expected}) -> Reward: {reward}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
