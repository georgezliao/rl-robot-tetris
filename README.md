Plan for the project:

Integrate a self-made tetris game into an RL loop. I want test theory and compare the Deep RL Approach (Raw Board Grid) and also the Heuristic, matrix-based, approach.

High Level Design:
- Follows a state, reward, done, info = env.step(action) system.
- Deep RL Approach will just take the tetris game through a CNN and train based on the image itself, philosophically, one could say this is more like what a human would see. 
- To maximize the effectiveness of the robot, there are a couple features that we can train our little robot to do: 1. Calculate the number of lines one action clears (and maybe even the second line). 2. The number of holes one action results in 3. And more?
(Regarding the above, we won't be telling or giving an algorithm to the agent, the agent will only be aware that they should focus on those things, but they won't know HOW/in what proportion to do those actions.)
- I guess, on that note, it would be interesting to compare the performance to already established algorithms like MisaMino which uses a Beam Search algorithm, and Cold Clear which uses a Monte Carlo Tree Search.
- The next choice is simply an experimental design aesthetic choice, we could either allow the agent to just choose the orientation and where to place it and it'll place it. Or we can challenge it and only give it the controls. The latter would be more complicated but in my opinion fits more in the spirit of good RL.

