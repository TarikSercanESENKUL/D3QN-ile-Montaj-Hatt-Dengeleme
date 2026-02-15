import pandas as pd
import numpy as np
from config import ROTA_FILE, BOM_FILE, SKILLS_FILE, TOTAL_WORKERS, NUM_SHIFTS, WORKERS_PER_SHIFT

class DataLoader:
    def __init__(self):
        self.rota_df = None
        self.bom_df = None
        self.skills_df = None
        self.station_dag = {}  # Adjacency list for station dependencies
        self.station_workloads = {} # Base workload per station (minutes)
        self.workers = [] # List of worker dicts

    def load_data(self):
        """Loads all CSV data and processes it."""
        print("Loading data...")
        self._load_rota()
        self._load_bom()
        self._load_and_generate_workers()
        print("Data loaded successfully.")

    def _load_rota(self):
        # Load factory route to build the DAG
        # Encoding might be cp1254 for Turkish characters
        try:
            self.rota_df = pd.read_csv(ROTA_FILE, encoding='cp1254')
        except:
            self.rota_df = pd.read_csv(ROTA_FILE, encoding='latin1')
        
        # Build DAG based on SortOrder or logical sequence
        # Assuming Rota defines the sequence. 
        # Simplified DAG construction based on report description:
        # R1-R5 (Start) -> R10 (Merge)
        # G1/G2 -> R10 (Merge)
        # R10 -> R9 -> R11 -> R12 (Main Line)
        
        # We will hardcode the DAG structure for now based on the report 
        # because the CSV SortOrder is linear but the description says "Parallel".
        # The CSV shows: 
        # 9->R1, 10->R2, 11->G1, 12->G2, 25->R3, 26->R4, 27->R5 ...
        # 31->R9, 32->R10, 33->R11, 34->R12
        # Note: Report says R10 is merge (Union).
        
        self.station_dag = {
            # Sub-assembly Line (Chassis/Bottom)
            # R6, R7, R8, R11, R12 all feed into R9 (Chassis Completion)
            "R6": ["R9"], "R7": ["R9"], "R8": ["R9"], 
            "R11": ["R9"], "R12": ["R9"], 
            
            "R9": ["R10"], # Chassis goes to Marriage (R10)

            # Body Parts Line
            "R1": ["R10"], "R2": ["R10"], "R3": ["R10"], "R4": ["R10"], "R5": ["R10"], # Body inputs feed R10

            # Main Assembly & Finishing
            "R10": ["G1"],  # Body + Chassis merge at R10
            "G1": ["G2"],  
            "G2": []        # End
        }
        
    def _load_bom(self):
        # Load Bill of Materials to calculate station workloads
        try:
            self.bom_df = pd.read_csv(BOM_FILE, encoding='utf-8-sig')
        except:
            self.bom_df = pd.read_csv(BOM_FILE, encoding='latin1')

        # Clean column names
        self.bom_df.columns = [c.strip().replace('"', '').replace("'", "") for c in self.bom_df.columns]
        # Remove BOM if present in the first column
        if self.bom_df.columns[0].startswith('\ufeff'):
             new_cols = list(self.bom_df.columns)
             new_cols[0] = new_cols[0].replace('\ufeff', '')
             self.bom_df.columns = new_cols

        print(f"DEBUG: BOM Columns: {self.bom_df.columns.tolist()}")
        
        # Calculate base workload per station
        # Formula: Sum(Quantity * Alpha + Beta)
        # For simplicity, we use the 'Workload' column if it exists, sum it up.
        
        if 'Workload' in self.bom_df.columns and 'StationID' in self.bom_df.columns:
            workload_series = self.bom_df.groupby('StationID')['Workload'].sum()
            self.station_workloads = workload_series.to_dict()
        elif 'StationID' in self.bom_df.columns:
             # Fallback if no explicit time column
            quantity_series = self.bom_df.groupby('StationID')['Quantity'].sum()
            self.station_workloads = (quantity_series * 5.0).to_dict() # 5 mins per part default
        else:
            print("ERROR: 'StationID' column missing in BOM file!")
            self.station_workloads = {}

        # Ensure all stations in DAG have a workload Entry
        # USER_REQUEST: Stations usually have same durations.
        # Setting standardized workload ~60 mins for all.
        for station in self.station_dag.keys():
            self.station_workloads[station] = 60.0
            
        # R12 override check (removed, since we want uniform)
        # self.station_workloads["R12"] = 60.0

    def _load_and_generate_workers(self):
        # Load base skills
        try:
            raw_skills = pd.read_csv(SKILLS_FILE, encoding='cp1254')
        except:
            raw_skills = pd.read_csv(SKILLS_FILE, encoding='latin1')
            
        # Normalize columns
        # Map messy Turkish headers to clean English keys
        # Column 1: Welding, 2: Grinding, 3: Crane, 4: Blueprint
        # 0 is WorkerID
        
        base_workers = []
        for _, row in raw_skills.iterrows():
            skills = {
                "welding": float(row.iloc[1]) if not pd.isna(row.iloc[1]) else 2.0,
                "grinding": float(row.iloc[2]) if not pd.isna(row.iloc[2]) else 2.0,
                "crane": float(row.iloc[3]) if not pd.isna(row.iloc[3]) else 2.0,
                "blueprint": float(row.iloc[4]) if not pd.isna(row.iloc[4]) else 2.0
            }
            base_workers.append(skills)
            
        # Generate 184 workers (4 groups of 46)
        # We recycle the base profiles to maintain distribution
        self.workers = []
        worker_id_counter = 1
        
        # We need 4 groups (3 active shifts + 1 reserve)
        # Assuming base_workers has ~46 rows. If less/more, we loop.
        
        num_base = len(base_workers)
        
        for group_id in range(4): # 0, 1, 2 (Active), 3 (Reserve)
            for i in range(WORKERS_PER_SHIFT):
                # Pick a profile from base (cyclic)
                base_profile = base_workers[i % num_base]
                
                # Create worker dict
                worker = {
                    "id": worker_id_counter,
                    "shift_group": group_id, # 0,1,2 are active shifts in rotation, 3 is static reserve
                    "skills": base_profile.copy()
                }
                
                # REQ: Group 3 is "Acemi" (Novice) -> Reduce skills
                if group_id == 3:
                    for k in worker['skills']:
                         worker['skills'][k] *= 0.7 # 30% less skilled
                
                # Calculate Level (Avg Skill)
                avg_skill = sum(worker['skills'].values()) / len(worker['skills'])
                worker['level'] = int(round(avg_skill))
                
                self.workers.append(worker)
                worker_id_counter += 1
                
        # Validate Shift Power (REQ: > 460 total skill points)
        # Assuming sum of all skills * workers
        # Just a check, real enforcement is harder without discarding data.
        # We print a warning if constraint isn't met.
        for g in range(3):
             group_power = sum([sum(w['skills'].values()) for w in self.workers if w['shift_group'] == g])
             if group_power < 460:
                 print(f"WARNING: Shift {g} Power {group_power:.1f} < 460 Constraint!")
                
        # Limit to TOTAL_WORKERS just in case
        self.workers = self.workers[:TOTAL_WORKERS] 

    def get_station_avg_workload(self, station_id):
        return self.station_workloads.get(station_id, 10.0)

if __name__ == "__main__":
    dl = DataLoader()
    dl.load_data()
    print(f"Loaded {len(dl.workers)} workers.")
    print(f"Station Workloads: {dl.station_workloads}")
