import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
from simulation import BusFactory, Workstation
from data_loader import DataLoader
from config import STATIONS, SHIFT_DURATION, REWARD_COMPLETION, REWARD_STEP, REWARD_OVERTIME, REWARD_IDLE, SIM_TIME

class BusFactoryEnv(gym.Env):
    def __init__(self, enable_logging=False):
        super().__init__()
        self.enable_logging = enable_logging
        self.event_log = []
        
        # Load Data
        self.dl = DataLoader()
        self.dl.load_data()
        self.num_workers = len(self.dl.workers)
        self.num_stations = len(STATIONS)
        
        # Action Space: WorkerID * StationID
        # Flattened: Action 0 = Worker 0 -> Station 0, ...
        self.action_space = spaces.Discrete(self.num_workers * self.num_stations)
        
        # Observation Space:
        # - Worker Status (Busy/Idle) [num_workers]
        # - Station Status (Queue Size, Occupied) [num_stations * 2]
        # - Assembly Buffers (R10 inputs + R9 inputs) [6 + 3 = 9]
        # - Shift Status [1]
        self.obs_dim = self.num_workers + (self.num_stations * 2) + 9 + 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.sim_env = None
        self.factory = None
        
        # Mapping for Action decoding
        self.station_map = {i: name for i, name in enumerate(STATIONS)}
        self.station_rev_map = {name: i for i, name in enumerate(STATIONS)}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.enable_logging:
            self.event_log = []
        
        self.sim_env = simpy.Environment()
        self.factory = BusFactory(self.sim_env, self.dl)
        
        # CRITICAL: Reset Worker States (is_busy, minutes_worked)
        for w in self.dl.workers:
             w['is_busy'] = False
             w['minutes_worked_this_shift'] = 0
        
        # WARMUP: Inject Full Capacity Inventory
        # Ensures stations start FULL (High Utilization from Minute 0)
        # and helps achieving the steady-state rate of ~18-20 buses.
        for sid, s in self.factory.stations.items():
            cap = s.machine.capacity
            for pred in s.predecessors:
                s.input_buffer[pred] = cap
            
            # Create Jobs for the injected inventory
            for _ in range(cap):
                s._check_assembly_condition()
        
        # Episode Metrics
        self.metrics = {
            "buses_produced": 0,
            "total_busy_minutes": 0,
            "total_overtime_minutes": 0,
            "total_labor_cost": 0
        }

        self.last_shift = 0 # Initialize shift tracker
        self.is_on_break = False # Initialize break tracker
        
        # Start initial processes
        self.sim_env.process(self._production_flow())
        
        self.sim_env.run(until=1) 
        
        return self._get_observation(), {}
        
    def step(self, action):
        # Decode Action
        worker_idx = action // self.num_stations
        station_idx = action % self.num_stations
        
        worker_id = self.dl.workers[worker_idx]['id'] 
        worker_dict = self.dl.workers[worker_idx]
        station_name = self.station_map[station_idx]
        station = self.factory.stations[station_name]
        
        reward = REWARD_STEP 
        terminated = False
        truncated = False
        
        # Execute Action
        duration = self._calculate_duration(worker_dict, station)
        
        # Decrement queue immediately (SimPy resource does this on request usually, but we manage queue_size manually for Observation)
        if station.queue_size > 0:
             station.queue_size -= 1
             
        # Mark Worker as Busy
        worker_dict['is_busy'] = True
             
        self.sim_env.process(self._execute_task(station, worker_dict, duration))
        
        if self.enable_logging:
            self.event_log.append({
                "event": "ASSIGN",
                "time": self.sim_env.now,
                "worker": worker_id,
                "station": station_name,
                "duration": duration
            })
        
        # Track Usage
        self.metrics["total_busy_minutes"] += duration
        worker_dict['minutes_worked_this_shift'] = worker_dict.get('minutes_worked_this_shift', 0) + duration
        
        # Advance Simulation
        # Track initial G2 (Final Station) completion count
        final_station = self.factory.stations["G2"]
        initial_finished_count = final_station.parts_processed
        
        # Advance Simulation
        try:
            self.sim_env.run(until=self.sim_env.now + 1.0) 
        except simpy.Interrupt:
            pass
            
        current_shift = self.factory.worker_pool.current_shift
        
        # Log Shift Change
        if current_shift != self.last_shift:
            if self.enable_logging:
                self.event_log.append({
                    "event": "SHIFT_CHANGE",
                    "time": self.sim_env.now,
                    "new_shift": current_shift
                })
            self.last_shift = current_shift
            
        # Log Break Status
        current_break_status = self.factory.worker_pool.is_break
        if current_break_status != self.is_on_break:
            event_type = "BREAK_START" if current_break_status else "BREAK_END"
            if self.enable_logging:
                self.event_log.append({
                    "event": event_type,
                    "time": self.sim_env.now
                })
            self.is_on_break = current_break_status
            
        # Check Delta for G2
        final_finished_count = final_station.parts_processed
        new_buses = final_finished_count - initial_finished_count
        
        if new_buses > 0:
             # Standard Bus Count (No Subtasks)
             real_buses = float(new_buses)
             
             reward += (REWARD_COMPLETION * real_buses)
             self.metrics["buses_produced"] += real_buses
             
             if self.enable_logging:
                 self.event_log.append({
                     "event": "BUS_COMPLETE",
                     "time": self.sim_env.now,
                     "count": real_buses
                 })

             
        if self.sim_env.now >= SIM_TIME: 
            truncated = True
        
        # Calculate Cost (Approximation)
        # Regular: 184 workers * 8 hours/shift * 3 shifts * 30 days? No, simpler:
        # Cost = Accumulated Busy Time * Rate ? 
        # Better: Cost is fixed per shift (Salaries) + Efficiency Bonus?
        # Let's define Cost = (Time Elapsed / 60) * Total Workers * HourlyRate + Overtime
        # Hourly Rate = 10 units
        hours_elapsed = self.sim_env.now / 60.0
        # This is cumulative cost over the episode
        self.metrics["total_labor_cost"] = hours_elapsed * self.num_workers * 10
             
        obs = self._get_observation()
        
        # Pass metrics in info
        info = self.metrics.copy()
        
        return obs, reward, terminated, truncated, info
        
    def _execute_task(self, station, worker, duration):
        """SimPy process for the actual work."""
        try:
            # Occupy station
            with station.machine.request() as req:
                yield req
                
                if self.enable_logging:
                    self.event_log.append({
                        "event": "START",
                        "time": self.sim_env.now,
                        "worker": worker['id'],
                        "level": worker.get('level', 1),
                        "station": station.station_id
                    })
                
                # 0. Check for Supply Chain Disruption / Part Shortage (Line Feeding)
                from config import PART_SHORTAGE_RATE
                if random.random() < PART_SHORTAGE_RATE:
                    delay = random.randint(15, 60)
                    yield self.sim_env.timeout(delay)
                
                # Work
                station.is_busy = True
                yield self.sim_env.timeout(duration)
                station.is_busy = False
                station.parts_processed += 1
                
                if self.enable_logging:
                    self.event_log.append({
                        "event": "FINISH",
                        "time": self.sim_env.now,
                        "worker": worker['id'],
                        "station": station.station_id
                    })
                
                # PUSH TO NEXT STATION (DAG LOGIC)
                next_stations = self.dl.station_dag.get(station.station_id, [])
                
                # CRITICAL: Sink Logic
                if not next_stations:
                    self.factory.total_finished += 1
                    
                for next_sid in next_stations:
                    if next_sid in self.factory.stations:
                        target_station = self.factory.stations[next_sid]
                        
                        # ONE PIECE FLOW (Strict Buffer Limit = 1 BUS)
                        while target_station.input_buffer.get(station.station_id, 0) >= 1:
                            yield self.sim_env.timeout(1.0)
                            
                        target_station.add_component(station.station_id)
        finally:
            worker['is_busy'] = False

    def _calculate_duration(self, worker, station):
        """Skill-based duration calculation."""
        from config import SIM_TIME
        
        # UNIVERSAL CALIBRATION:
        # User Formula: Workload = (Total Time * Capacity) / Target Buses
        # Balanced Line: Duration scales with Capacity to ensure full utilization with Standard Feed.
        # This fixes "Rear Body 1 person" issue by making Rear Body slower (needs parallel work).
        # Theoretical Target: 1800 Buses/Month (Aggressive).
        base_time = (SIM_TIME * station.machine.capacity) / 1800.0
 
        # Simple skill factor logic
        # Average skill? Or specific?
        # Let's say welding for R stations...
        skill_level = worker['skills']['welding'] # Default to welding for now
        factor = 1.0
        if skill_level > 3.0: factor = 0.8 # Faster
        elif skill_level < 2.5: factor = 1.2 # Slower
        
        # Apply Fatigue
        fatigue = self._calculate_fatigue_factor(worker) # Internal helper uses generic name 'worker', passing worker dict is fine if arg matches
        # Wait, inside _calculate_duration, the argument is named 'worker'.
        # Let's check _calculate_duration signature.
        # def _calculate_duration(self, worker, station):
        # Ah! The argument IS named 'worker' in the function definition!
        
        # Checking traceback again:
        # File "C:\Users\MSI\projects\dqnproductlinev1\factory_env.py", line 84, in step
        # worker['minutes_worked_this_shift'] = ...
        
        # The error was in 'step' method, not inside '_calculate_duration'.
        # In 'step' method, the variable is 'worker_dict'.
        
        # Inside '_calculate_duration', the variable IS 'worker'.
        factor *= fatigue
        
        return base_time * factor
        
    def _calculate_fatigue_factor(self, worker):
        """
        Calculates efficiency drop due to fatigue.
        - Assuming fatigue kicks in after 4 hours (240 mins) of work in a shift.
        - Dropping efficiency by 10% for every hour after that.
        - We need to track 'minutes_worked_this_shift' for each worker.
        """
        # This requires tracking accumulated work per shift per worker.
        # Since we don't have a granular worker object state in Env, we rely on 'worker_dict'.
        # We need to add a 'fatigue' tracker to the worker dict in DataLoader/Env.
        
        minutes_worked = worker.get('minutes_worked_this_shift', 0)
        
        if minutes_worked < 240:
            return 1.0 # No fatigue
        else:
            # Overtime / Late shift fatigue
            # Example: 300 mins -> 60 mins over -> 10% slower -> factor 1.1
            excess_hours = (minutes_worked - 240) / 60.0
            fatigue_penalty = 1.0 + (excess_hours * 0.1) 
            return fatigue_penalty

    def _get_observation(self):
        # 1. Worker Status (1 if busy/active, 0 if not)
        # Note: We need to track who is busy in SimPy.
        worker_status = np.zeros(self.num_workers)
        # TODO: Link SimPy active status to this
        
        # 2. Station Status
        station_status = np.zeros(self.num_stations * 2)
        for i, name in enumerate(STATIONS):
            s = self.factory.stations[name]
            station_status[i*2] = len(s.machine.queue) # Queue
            station_status[i*2+1] = 1 if s.is_busy else 0 # Busy
            
        # 3. Component Buffers (NEW: Essential for Assembly Logic)
        prev_shift = getattr(self, 'last_shift_index', -1)
        current_shift_idx = self.factory.worker_pool.current_shift
        
             
        if current_shift_idx != prev_shift:
              # Log Shift Change
              if self.enable_logging:
                  self.event_log.append({
                      "event": "SHIFT_CHANGE",
                      "time": self.sim_env.now,
                      "shift": current_shift_idx + 1 # 1-indexed for display
                  })
                  
              # Reset fatigue for all workers when shift rotates
              # Ideally only for the *new* shift workers, but resetting all is safe 
              # as off-shift workers won't work.
              for w in self.dl.workers:
                  w['minutes_worked_this_shift'] = 0
              self.last_shift_index = current_shift_idx
             
        # Track how many parts R10, R9 have in their input buffers
        # We flatten this: For each station, for each of its predecessors, count buffered items.
        # To keep vector fixed size, we allocate slots for all stations' potential buffers.
        # Simplified: Just track "Ready Ratio" for merge stations? 
        # Better: Fixed tracking for Critical Paths (R10 inputs: R1..R5, R12)
        
        # New approach: Add a vector of length 'Total Predecessors in DAG'
        # But DAG is dynamic? No, static.
        # Let's add specific features for R10 (Main Merge) and R9 (Sub Merge) status.
        r10 = self.factory.stations["R10"]
        r9 = self.factory.stations["R9"]
        
        # R10 Inputs: R1, R2, R3, R4, R5, R12 (6 slots)
        r10_buffer = [r10.input_buffer.get(k, 0) for k in ["R1", "R2", "R3", "R4", "R5", "R12"]]
        
        # R9 Inputs: R6, R7, R8 (3 slots)
        r9_buffer = [r9.input_buffer.get(k, 0) for k in ["R6", "R7", "R8"]]
        
        buffer_status = np.array(r10_buffer + r9_buffer, dtype=np.float32)
        
        # 4. Shift
        shift = [self.factory.worker_pool.current_shift / 3.0]
        
        return np.concatenate([worker_status, station_status, buffer_status, shift], dtype=np.float32)

    def _get_action_mask(self):
        """Returns boolean mask where 1 = Valid."""
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        
        # Iterate all potential actions
        for w_idx, worker in enumerate(self.dl.workers):
            for s_idx, s_name in enumerate(STATIONS):
                action_idx = w_idx * self.num_stations + s_idx
                
                # Check Validity
                # 1. Is Worker in current shift?
                current_shift = self.factory.worker_pool.current_shift
                if worker['shift_group'] != current_shift and not worker.get('is_reserve', False):
                    continue
                    
                # 2. Is Worker Busy? (Need tracking)
                if worker.get('is_busy', False):
                    continue
                
                # 3. Is Station Queue Empty? (No work to do)
                s = self.factory.stations[s_name]
                if s.queue_size <= 0: continue
                
                # 4. Is Station at Full Capacity? (Prevent assigning workers to wait)
                # Count current users + pending requests
                current_assigned = s.machine.count + len(s.machine.queue)
                if current_assigned >= s.machine.capacity:
                    continue
                
                mask[action_idx] = 1
                
        return mask

    def _production_flow(self):
        """Generates parts into station queues (Standard Feed)."""
        while True:
            # Yield parts for R1-R5 every X mins
            # Standard Feed: 1 part per station per cycle (TML Logic)
            for start_node in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R11", "R12"]:
                self.factory.stations[start_node].queue_size += 1
                
            yield self.sim_env.timeout(15) # Feed every 15 mins (Aggressive)
            
import random 
