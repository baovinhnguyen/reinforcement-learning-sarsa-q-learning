# reinforcement-learning

- Use two different reinforcement learning algorithms, SARSA and Q-learning, to learn the optimal policy for an agent in a grid-world environment consisting of:

   • A starting state, marked by an S.

   • A goal state, marked by a G.
   
   • A set of states consisting of open space, marked by an O. • A set of states consisting of icy surface, marked by an I. • A set of holes, marked by an H.
    
 - We have the following dynamics:
 
   • An agent can move in one of four directions—up, down, left, right—by a single grid-square.
    
   • If the agent is in either the start state or in open space, it can move in any of the four directions deterministically, except that it cannot leave the grid. 
    
   •Any move that attempts to move off the grid will cause the agent to stay exactly where it is. The goal location functions as an absorbing state, so that once the agent has entered that state, any movement has no effect, and they remain at the goal.
   
   • If the agent is on an icy surface, any attempt to move in a given direction will succeed with probability 0.8; in other cases, the agent will move diagonally to the left or right in the given direction, with probability 0.1 of each possibility. 
    
   •Any move into a hole will cause the agent to return to the location from which they just moved (after climbing out of the hole), and will incur a cost of −50; all other movements into open space or on icy surfaces incur a cost of −1. For any action that results in entering the absorbing goal state, the reward received is +100.

Parameters:

   • An episode for learning is defined as min(M,N), where M is any number of moves that take the agent from the start state to the goal, and N = (10×w×h), where w and h are the width and height of the grid-world, respectively. That is, each episode ends as soon as the agent reaches the goal; if that doesn’t happen after N time-steps (actions taken), then the episode terminates without reaching the goal. After each episode, the agent will return to the start state.

   • The future discount factor is set to γ = 0.9; it never changes.
    
   • The policy randomness parameter is initially set to ε = 0.9; every 10 episodes it is updated. In particular, if E is the number of episodes already past, then for all values E >= 10, we set ε = 0.9/⌊E/10⌋. This means that after 1000 episodes, ε = 0.009; at this point, we should set it to 0, and no longer act randomly.
    
   • The step-size parameter for learning updates is set to α = 1.0, and can be omitted from calculations; for this application, we do not need to reduce it, as we are not particularly interested in the values to which it converges, only the policy that is produced.
