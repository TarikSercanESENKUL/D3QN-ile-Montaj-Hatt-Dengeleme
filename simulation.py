import simpy
import random
import logging
from collections import deque
from config import SIM_TIME, SHIFT_DURATION, WORKERS_PER_SHIFT, ABSENTEEISM_RATE, NUM_SHIFTS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class WorkerPool:
    def __init__(self, env, workers_data):
        self.env = env
        self.workers_data = workers_data  # List of all worker dicts
        self.active_workers = simpy.FilterStore(env) # Store for currently available workers
        self.current_shift = 0
        self.total_shifts = 0
        
        # Performance metrics
        self.total_overtime_minutes = 0
        self.reserve_usage_count = 0
        self.is_break = False # Track break status
        
        # Start the shift management process
        self.env.process(self.manage_shifts())

    def manage_shifts(self):
        """Cyclical process to manage shifts every 8 hours with 30 min break."""
        while True:
            self._start_shift(self.current_shift)
            
            # Work Part 1: 3.75 hours (225 mins)
            yield self.env.timeout(225)
            
            # BREAK START
            self.is_break = True
            
            # Break Duration: 30 mins
            yield self.env.timeout(30)
            
            # BREAK END
            self.is_break = False
            
            # Work Part 2: 3.75 hours (225 mins)
            yield self.env.timeout(225)
            
            self._end_shift()
            
            # Rotate shift
            self.current_shift = (self.current_shift + 1) % NUM_SHIFTS
            self.total_shifts += 1

    def _start_shift(self, shift_id):
        """Loads workers for the current shift into the store, handling absenteeism."""
        logger.info(f"--- Starting Shift {shift_id} at {self.env.now} ---")
        
        # Identify workers belonging to this shift
        shift_workers = [w for w in self.workers_data if w['shift_group'] == shift_id]
        
        # Simulate Absenteeism
        present_workers = []
        absent_skills = []
        
        for w in shift_workers:
            if random.random() < ABSENTEEISM_RATE:
                # Worker is absent
                absent_skills.append(w['skills'])
            else:
                present_workers.append(w)
                
        # Fill gaps with Reserve Pool (Group 3) if needed
        reserve_workers = [w for w in self.workers_data if w['shift_group'] == 3]
        random.shuffle(reserve_workers) # Randomize reserve selection order
        
        # Simple reserve allocation logic: Match count first, theoretically skill matching should be here
        needed_count = len(shift_workers) - len(present_workers)
        
        for i in range(min(needed_count, len(reserve_workers))):
            reserve_w = reserve_workers[i]
            # Rename/Mark as active reserve
            reserve_w_copy = reserve_w.copy()
            reserve_w_copy['is_reserve'] = True
            present_workers.append(reserve_w_copy)
            self.reserve_usage_count += 1
            
        # Put workers into the store
        for w in present_workers:
            # We wrap the worker dict in an object or just use the dict
            # FilterStore needs items.
            self.active_workers.put(w)
            
        logger.info(f"Shift {shift_id}: {len(present_workers)} workers active ({needed_count} from reserve).")

    def _end_shift(self):
        """Clears the worker store for the next shift."""
        # In a real continuous sim, we might wait for tasks to finish.
        # Here we assume 'hot swap' or tasks pause.
        # To simplify, we clear the store. Workers currently busy will eventually release to a cleared store.
        # We need to handle this carefully: busy workers are NOT in the store. 
        # They will put THEMSELVES back when done. We need to prevent that if they are off shift.
        # Implementation Detail: We won't purge busy workers, but we drain the idle ones.
        
        # Drain all idle workers
        items = []
        while self.active_workers.items:
            items.append(self.active_workers.get().value)
        
        # Calculate overtime for any worker still busy? 
        # For now, simplistic clear.
        pass

class Workstation:
    def __init__(self, env, station_id, avg_duration, worker_pool, capacity=1):
        self.env = env
        self.station_id = station_id
        self.avg_duration = avg_duration
        self.worker_pool = worker_pool
        
        self.machine = simpy.Resource(env, capacity=capacity) # Capacity defined by config
        self.queue_size = 0
        self.parts_processed = 0
        
        # For RL to observe
        self.current_worker = None
        self.is_busy = False
        
        # Assembly Logic: Input Buffer
        # Tracks how many parts we have from each predecessor
        self.input_buffer = {} 
        self.predecessors = [] # Will be populated by Factory

    def add_component(self, source_station_id):
        """Called when a predecessor finishes a part."""
        if source_station_id not in self.input_buffer:
            self.input_buffer[source_station_id] = 0
        self.input_buffer[source_station_id] += 1
        
        # Check if we have a full set to create a job
        self._check_assembly_condition()
        
    def _check_assembly_condition(self):
        """If we have at least 1 part from ALL predecessors, consume them and add to main queue."""
        if not self.predecessors:
            # Source station (R1..R5), logic driven by external generator
            return 
            
        # Check if all predecessors have > 0 in buffer
        can_assemble = True
        for pred in self.predecessors:
            if self.input_buffer.get(pred, 0) < 1:
                can_assemble = False
                break
                
        if can_assemble:
            # Consume 1 from each
            for pred in self.predecessors:
                self.input_buffer[pred] -= 1
            
            # Add to main processing queue
            self.queue_size += 1
            # Recurse? No, 1 set = 1 job. Simulation flow handles the rest.

    def process_part(self, part_id):
        """A process that requests a machine and a worker."""
        
        # 1. Request Machine (Space)
        with self.machine.request() as req_machine:
            yield req_machine
            
            # Machine acquired. Now we need a worker.
            # This is where the RL Agent comes in usually.
            # BUT: For SimPy standalone, we verify with FIFO. 
            # For RL integration, this 'request worker' part is replaced by an external decision or
            # the process yields until an 'assigned_worker' event is triggered.
            
            # HYBRID APPROACH:
            # The process yields on a 'worker_assigned' event.
            # The RL agent sets this event when it picks an action.
            
            pass 
            # Note: The logic for "process_part" depends on if we are running 
            # pure simulation or RL training.
            # We will refactor this in BusFactoryEnv. 

class BusFactory:
    def __init__(self, env, data_loader):
        self.env = env
        self.dl = data_loader
        self.worker_pool = WorkerPool(env, data_loader.workers)
        self.stations = {}
        self.total_finished = 0  # <--- Added Tracker
        
        from config import STATION_CAPACITIES
        
        # Initialize stations
        for sid in data_loader.station_dag.keys():
            workload = data_loader.get_station_avg_workload(sid)
            capacity = STATION_CAPACITIES.get(sid, 1)
            self.stations[sid] = Workstation(env, sid, workload, self.worker_pool, capacity)
            
        # Populate Predecessors for Assembly Logic
        # Invert DAG: Value -> Key dependency
        for src, dests in data_loader.station_dag.items():
            for dest in dests:
                if dest in self.stations:
                    self.stations[dest].predecessors.append(src)
                    # Initialize buffer key
                    self.stations[dest].input_buffer[src] = 0

