from gym import Env
from gym.spaces import Discrete, Box
import time
import numpy as np
import os, sys
import traci
import traci.constants as tc
from traci.exceptions import TraCIException
import pathlib
import csv

# defines env
class SumoEnv(Env):
    def __init__(self, gui):
        # defining misc variables
        self.step_length = 0.1
        self.rl_counter = 1
        self.rl_id = "rl_1"
        self.speed_limit = 0
        self.network_length = 0
        self.merge_counter = 0

        # initalise reset variables
        self.timeout = False
        self.merged = False
        self.crash = False
        self.triggered = False

        # setting vehicle observation variables and initalising state
        self.leading_obs = 2
        self.trailing_obs = 2
        self.other_obs = 6
        self.num_observations = 2 * (self.leading_obs + self.trailing_obs) + self.other_obs
        self.state = [0] * self.num_observations

        # produces enough actions for +-3 at 0.5 intervals
        self.max_accel = 3
        self.num_acc_actions = 4 * self.max_accel + 1

        # defining state and observation spaces
        self.action_space = Discrete(self.num_acc_actions + 1)
        self.observation_space = Box(low=-1, high=3, shape=(self.num_observations,), dtype=np.float32)

        # start the SUMO simulation
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)

        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")

        if gui == True:
            self.sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
        else:
            self.sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"

        self.path = os.path.join(pathlib.Path(__file__).parent.resolve(), "eval_sim.sumocfg")

        # sumo start cmd. Sets step length, checks only real collisions
        self.sumoCmd = [self.sumoBinary, "-c", self.path, "--step-length", f"{self.step_length}", "--collision.check-junctions", "--lateral-resolution", "1.6", "--collision.mingap-factor", "0", "--random"]
        self.loadCmd = self.sumoCmd[1:]

        f = open(r"C:\Users\AdminL\Desktop\TraCI\82 Evaluation\Evaluations\velocity.csv", "w", newline="")
        self.velocity_writer = csv.writer(f)

    # this is how a step is taken
    def step(self, action):
        info = {}
        done = False

        # apply the actions and move a step forward
        self.apply_action(action, self.rl_id)
        traci.simulationStep()
        self.velocity_writer.writerow([self.rl_id, traci.vehicle.getSpeed(self.rl_id)])
        if ":merge" in traci.vehicle.getRoadID(self.rl_id):
            traci.vehicle.changeSublane(self.rl_id, -traci.vehicle.getLateralLanePosition(self.rl_id))

        # check for collision, merge or timeout here
        if traci.simulation.getCollisions():
            self.crash = True
            info = {"is_success": False}

        if traci.simulation.getTime() > 150:
            self.timeout = True
            info = {"is_success": True}

        if not self.crash and not self.timeout:
            if traci.vehicle.getLaneID(self.rl_id) in ["merging_1", "outgoing_0", "outgoing_1"]:
                self.merged = True
                traci.vehicle.setColor(self.rl_id, (255,255,255))     
                info = {"is_success": True}
        
        # if some conditions are met set done to True. Otherwise done remains false.
        if self.crash or self.merged or self.timeout:
            done = True
        
        # gets the new state
        self.state = self.get_state(self.rl_id, crash=self.crash)

        # calculates the new reward
        reward = self.get_reward(self.rl_id, crash=self.crash, merged=self.merged)
    
        return self.state, reward, done, info


    def reset(self):

        if self.crash or self.timeout:
            # reset for crash or sim timeout
            traci.load(self.loadCmd)

            self.rl_counter = 1
            self.rl_id = "rl_1"
            self.release_rl()

        elif self.merged:
            # return contorl to sim
            traci.vehicle.setSpeedMode(self.rl_id, 31)
            traci.vehicle.setLaneChangeMode(self.rl_id, 1621)
            traci.vehicle.setSpeed(self.rl_id, -1)

            # releases a new vehicle and sets appropriate speed and lane change modes
            self.release_rl()

        else:
            # initial run setup, starts sim, gets speed limit, resets rl_id, gets network length, changes settings for first RL vehicle
            try:
                traci.start(self.sumoCmd)
            except TraCIException:
                traci.close()
                time.sleep(10)
                traci.start(self.sumoCmd)

            self.rl_counter = 1
            self.release_rl()
            self.rl_id = "rl_1"
            self.speed_limit = max(
            traci.lane.getMaxSpeed(lane) for lane in ["on-ramp_0", "incoming_0", "merging_0", "outgoing_0"]
            )

            self.network_length = sum(
                traci.lane.getLength(lane) for lane in ["on-ramp_0", "incoming_0", "merging_0", "outgoing_0"]
            ) - 100

        self.crash = False
        self.merged = False
        self.timeout = False
        self.triggered = False

        self.state = self.get_state(self.rl_id, crash=self.crash)
        
        return self.state
###########################################################################################################################
# primary functions

    def apply_action(self, action, rl_id):
        if action < self.num_acc_actions:
            traci.vehicle.setAcceleration(rl_id, action/2 - 3, self.step_length)
        else:
            if traci.vehicle.getRoadID(rl_id) == "merging" and traci.vehicle.getPosition(rl_id)[0] < 240 and self.triggered == False:
                traci.vehicle.changeSublane(rl_id, 3.2)
                self.triggered = True


    def get_state(self, rl_id, **kwargs):
        observation = [0] * self.num_observations
        total_observed = self.leading_obs + self.trailing_obs

        if kwargs["crash"]:
            return observation

        ego_speed = traci.vehicle.getSpeed(rl_id)
        merge_dist = (350 - traci.vehicle.getPosition(rl_id)[0])
        current_lane = traci.vehicle.getLaneIndex(rl_id)
        edge = traci.vehicle.getRoadID(rl_id)
        
        if edge == "":
            edge = "merging"
        
        num_lanes = traci.edge.getLaneNumber(edge)

        blocker_speed = 0

        if edge != "on-ramp":
            leaders = traci.vehicle.getLeftLeaders(rl_id, blockingOnly=True)
            followers = traci.vehicle.getLeftFollowers(rl_id, blockingOnly=True)
            blockers = followers + leaders

            for veh in blockers:
                if veh not in [None, ('',-1), ()]:
                    id = veh[0]
                    pos = traci.vehicle.getPosition(id)[0]
                    if pos == traci.vehicle.getPosition(rl_id)[0]:
                        blocker_speed = traci.vehicle.getSpeed(id)
                        break

        trailing_vehicles = self.add_trailing_vehicles(rl_id, self.trailing_obs, self.speed_limit, self.network_length, self.merged)
        leading_vehicles = self.add_leading_vehicles(rl_id, self.leading_obs, self.speed_limit, self.network_length, self.merged)

        for i, vehicle in enumerate(trailing_vehicles):
            observation[2 * i] = vehicle["speed"]
            observation[2 * i + 1] = vehicle["gap"]

        for i, vehicle in enumerate(leading_vehicles):
            observation[2 * i + 2 * self.trailing_obs] = vehicle["speed"]
            observation[2 * i + 1 + 2 * self.trailing_obs] = vehicle["gap"]

        observation[total_observed * 2] = ego_speed/self.speed_limit
        observation[total_observed * 2 + 1] = current_lane
        observation[total_observed * 2 + 2] = merge_dist/self.network_length
        observation[total_observed * 2 + 3] = num_lanes
        observation[total_observed * 2 + 4] = blocker_speed/self.speed_limit
        observation[total_observed * 2 + 5] = traci.vehicle.getLateralLanePosition(rl_id)/3.2
        
        self.state = observation
  
        return self.state


    def get_reward(self, rl_id, **kwargs):

        # if a crash has occurred return or on on-ramp
        if kwargs["crash"]:
            return -20

        ego_speed = traci.vehicle.getSpeed(rl_id)

        if not kwargs["merged"]:
            return 0

        # set default head and tailways 
        tailway = traci.vehicle.getPosition(rl_id)[0]
        headway = (500 - traci.vehicle.getPosition(rl_id)[0])

        leader_speed = ego_speed
        trailing_speed = ego_speed

        normalising_gap = traci.lane.getLength("incoming_0") + traci.lane.getLength("merging_0") + traci.lane.getLength("outgoing_0") - 100

        normalising_dis_from_centre = normalising_gap/2
    
        trailing = traci.vehicle.getFollower(rl_id, 500)
        if trailing not in [None, ('',-1), ()]:
            tailway = trailing[1] + traci.vehicle.getMinGap(trailing[0])
            trailing_speed = traci.vehicle.getSpeed(trailing[0])

        leading = traci.vehicle.getLeader(rl_id, 500) 
        if leading not in [None, ('',-1), ()]:
            headway = leading[1] + traci.vehicle.getMinGap(rl_id)
            leader_speed = traci.vehicle.getSpeed(leading[0])
        
        gap = headway + tailway + traci.vehicle.getLength(rl_id)
        dis_from_centre = abs(headway - tailway)

        leader_speed_dif = min(leader_speed - ego_speed, 0)
        trailing_speed_dif = min(ego_speed - trailing_speed, 0)
        
        if headway > 40 and tailway > 40:
            dis_from_centre = 0

        # weights 

        w1, w2, w3, w4, w5 = 0.5/self.speed_limit, 15/normalising_gap, 10/normalising_dis_from_centre, 12/self.speed_limit, 16/self.speed_limit
        # general form of usual reward function
        ego = w1 * ego_speed + w4 * leader_speed_dif
        oth = w2 * gap - w3 * dis_from_centre + w5 * trailing_speed_dif
        svo = np.radians(45) # svo value in radians

        reward = ego * np.cos(svo) + oth * np.sin(svo)

        return reward

########################################################################################################################
# helper functions
    def release_rl(self):
        
        # release a new vehicles, two steps forward, set rl_id variable
        traci.vehicle.add(vehID=f"rl_{self.rl_counter}",routeID="r_0", departPos=0, departLane="free", departSpeed=13, typeID="rl")
        self.rl_id = f"rl_{self.rl_counter}"
        traci.simulationStep()

        # change rl vehicle characteristics and get the first state
        traci.vehicle.setSpeedMode(self.rl_id, 32)
        traci.vehicle.setLaneChangeMode(self.rl_id, 0)

        # increases the ID counter
        self.rl_counter += 1
    
    def add_trailing_vehicles(self, start_id, num_followers, speed_limit, network_length, merged):
        vehicles = []
        if traci.vehicle.getRoadID(start_id) == "on-ramp":
            return vehicles

        for _ in range(num_followers):
            vehicle_data = {}

            # if the vehicle being checked is not in the merging lane
            if merged == True or "f" in start_id:
                trailing = traci.vehicle.getFollower(start_id, 500)
                if trailing in [None, ('',-1), ()]:
                    break
                else:
                    start_id = trailing[0]

            else:
                trailing = traci.vehicle.getLeftFollowers(start_id)
                if trailing in [None, ('',-1), ()]:
                    break
                else:
                    trailing = trailing[0]
                    start_id = trailing[0]
            
            gap = (trailing[1] + traci.vehicle.getMinGap(trailing[0]))
            speed = traci.vehicle.getSpeed(start_id)

            vehicle_data["id"] = start_id
            vehicle_data["gap"] = gap/network_length
            vehicle_data["speed"] = speed/speed_limit

            vehicles.append(vehicle_data)

        return vehicles

    def add_leading_vehicles(self, start_id, num_leaders, speed_limit, network_length, merged):
        vehicles = []
        if traci.vehicle.getRoadID(start_id) == "on-ramp":
            return vehicles

        for _ in range(num_leaders):
            vehicle_data= {}
            min_gap = traci.vehicle.getMinGap(start_id)
            # if the vehicle being checked is not in the merging lane
            if merged == True or "f"  in start_id:
                leading = traci.vehicle.getLeader(start_id, 500)
                if leading in [None, ('',-1), ()]:
                    break
                else:
                    start_id = leading[0]
            else:
                leading = traci.vehicle.getLeftLeaders(start_id)
                if leading in [None, ('',-1), ()]:
                    break
                else:
                    leading = leading[0]
                    start_id = leading[0]

            gap = leading[1] + min_gap
            speed = traci.vehicle.getSpeed(start_id)

            vehicle_data["id"] = start_id
            vehicle_data["gap"] = gap/network_length
            vehicle_data["speed"] = speed/speed_limit

            vehicles.append(vehicle_data)

        return vehicles