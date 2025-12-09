import sys
import os
sys.path.append(os.path.abspath("src"))
from adaptivegraph import LearnableEdge
import numpy as np

def debug_run():
    # Setup exactly like the notebook
    router = LearnableEdge(
        options=["premium", "fast"],
        policy="linucb",
        feature_dim=16,
        exploration_alpha=0.5
    )
    
    # Trace the internal scores
    print("--- Start Debug ---")
    
    # 1. VIP Request (Expect 'premium' -> idx 0)
    # 2. Guest Request (Expect 'fast' -> idx 1)
    
    # Force first input: VIP
    state_vip = "vip"
    ctx_vip = router.encoder.encode(state_vip)
    
    print(f"\n[Step 1] Input: VIP")
    # Peek at scores
    # We have to access internal policy
    p_vip = []
    for a in range(router.policy.n_actions):
        A_a = router.policy.A[a]
        b_a = router.policy.b[a]
        A_inv_x = np.linalg.solve(A_a, ctx_vip)
        theta_a = np.linalg.solve(A_a, b_a)
        ucb = np.dot(theta_a, ctx_vip) + router.policy.alpha * np.sqrt(np.dot(ctx_vip, A_inv_x))
        p_vip.append(ucb)
        print(f"Arm {a} score: {ucb:.4f} (Theta*x: {np.dot(theta_a, ctx_vip):.4f}, Unc: {router.policy.alpha * np.sqrt(np.dot(ctx_vip, A_inv_x)):.4f})")
        
    choice = router(state_vip)
    print(f"Choice: {choice}")
    
    # Reward it
    reward = 1.0 if choice == "premium" else 0.0
    print(f"Reward: {reward}")
    router.record_feedback(result={}, reward=reward)
    
    # Step 2: Guest
    state_guest = "guest"
    ctx_guest = router.encoder.encode(state_guest)
    print(f"\n[Step 2] Input: Guest")
    
    p_guest = []
    for a in range(router.policy.n_actions):
        A_a = router.policy.A[a]
        b_a = router.policy.b[a]
        A_inv_x = np.linalg.solve(A_a, ctx_guest)
        theta_a = np.linalg.solve(A_a, b_a)
        ucb = np.dot(theta_a, ctx_guest) + router.policy.alpha * np.sqrt(np.dot(ctx_guest, A_inv_x))
        p_guest.append(ucb)
        print(f"Arm {a} score: {ucb:.4f} (Theta*x: {np.dot(theta_a, ctx_guest):.4f}, Unc: {router.policy.alpha * np.sqrt(np.dot(ctx_guest, A_inv_x)):.4f})")

    choice_2 = router(state_guest)
    print(f"Choice: {choice_2}")
    
    if choice_2 == "premium":
        print("Model chose 'premium' for guest -> WRONG (but expected if it generalized poorly)")
    else:
        print("Model chose 'fast' for guest -> CORRECT")

if __name__ == "__main__":
    debug_run()
