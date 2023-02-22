# Autonomous Vehicle Trajectory Planning for On-Ramp Merging
## About
This project is being worked on as a final year project for BEng Mechanical Engineering at Queen's University, Belfast. It seeks to use reinforcement learning to handle acceleration and lane change decision making for autonomous on-ramp merging. The .xml files are used to generate the road network in the SUMO simualtor. train.py is used to train a DQN agent in the OpenAI gym environment defined in env.py. The resultant model can be evaluated using evaluate.py. This project is under active development.

### Environment
#### Overview
A SUMO subprocess is started using the TraCI API and communicated with to obtain states and implement actions. An ego vehicle is released initially and once it has merged, the episode is done and a new ego vehicle is released. The SUMO configuration is reloaded upon a collision or timeout.

#### State
A box state space is being used. The state space consists of gaps between vehicles, vehicle speeds and four other variables.

The gaps between the ego vehicle and vehicles immediately leading and trailing it in the right highway lane are included, along with their speeds. In addition to this, the gap between the leading vehicle in the right lane and its leading vehicle are also recorded, along with this vehicle’s speed. The gap between the following vehicle in the right lane and its following vehicle are included along with this vehicles speed.

In addition, the state space includes the ego vehicle’s speed, its x-distance to the merging point, the number of lanes in the section of road the ego vehicle is travelling on, the ego vehicle’s current lane, the ego vehicle's lateral position relative to the centre of the lane and the velocity of a directly adjacent vehicle if applicable.

#### Actions
A discrete action space is being used consisting of accelerations in the range +-3 m/s^2 in 0.5 m/s^2 increments along with a lane change option.

#### Reward
Generally, when the vehicle has not yet merged it receives no reward. A penalty for crashing is applied. Otherwise, it receives a reward based the concept of Social Value Orientation (SVO) from the field of social psychology. In this project, an SVO of 0 degrees indicates purely selfish behaviour and 90 degrees indicates purely altruistic behaviour. Sin(SVO) and Cos(SVO) are used as multipliers to adjust the value the ego vehicle places its own satisfaction and the satisfaction of the immediately surrounding vehicles.
