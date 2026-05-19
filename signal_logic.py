from config import (
    MIN_GREEN_TIME, MAX_GREEN_TIME, YELLOW_TIME, RED_ALL_TIME, 
    MAX_WAITING_THRESHOLD, PHASES
)
from logger_config import logger

class TrafficSignalOptimizer:
    def __init__(self):
        """
        Initializes the signal optimizer and state manager.
        """
        self.phases = PHASES
        self.num_phases = len(PHASES)
        self.active_phase_idx = 0
        
        # State can be: 'GREEN', 'YELLOW', 'RED_ALL'
        self.current_state = 'GREEN'
        self.timer = MIN_GREEN_TIME  # Remaining seconds in current state
        self.green_duration = MIN_GREEN_TIME  # Total green duration computed for active phase
        
        # Tracking wait time to prevent starvation
        # Keys are lane names, value is seconds waiting since last green
        self.lane_wait_times = {
            lane: 0.0 for phase in PHASES for lane in phase["lanes"]
        }
        
        # Track which phase is next (for sequential or priority-based switches)
        self.next_phase_idx = 1 if self.num_phases > 1 else 0
        
        # Emergency override flags
        self.emergency_override_active = False
        self.emergency_lane = None

    def calculate_green_duration(self, lane_densities):
        """
        Computes dynamic green time for the active phase based on lane density ratios.
        
        Formula:
            GreenTime = G_min + (G_max - G_min) * (ActivePhaseDensity / TotalDensity)
        """
        active_lanes = self.phases[self.active_phase_idx]["lanes"]
        active_density = sum(lane_densities.get(lane, 0.0) for lane in active_lanes)
        total_density = sum(lane_densities.values())
        
        if total_density == 0:
            return MIN_GREEN_TIME
            
        ratio = active_density / total_density
        duration = MIN_GREEN_TIME + (MAX_GREEN_TIME - MIN_GREEN_TIME) * ratio
        return int(clip(duration, MIN_GREEN_TIME, MAX_GREEN_TIME))

    def update_wait_times(self, elapsed_seconds):
        """
        Increases wait time for lanes that do not currently have a green light.
        Resets wait times for lanes that currently have a green light.
        """
        active_lanes = self.phases[self.active_phase_idx]["lanes"]
        
        for lane in self.lane_wait_times.keys():
            if self.current_state == 'GREEN' and lane in active_lanes:
                self.lane_wait_times[lane] = 0.0
            else:
                self.lane_wait_times[lane] += elapsed_seconds

    def check_starvation(self):
        """
        Checks if any lane has exceeded the maximum waiting threshold.
        Returns the lane name if starved, otherwise None.
        """
        for lane, wait_time in self.lane_wait_times.items():
            if wait_time >= MAX_WAITING_THRESHOLD:
                logger.warning(f"Starvation detected! Lane '{lane}' has been waiting for {wait_time:.1f}s.")
                return lane
        return None

    def handle_state_transitions(self, lane_densities, emergency_detected_lane, dt=1.0):
        """
        Executes the signal state machine transitions.
        Runs once per simulated second (dt = 1.0).
        """
        self.timer -= dt
        self.update_wait_times(dt)
        
        # Update emergency tracking
        if emergency_detected_lane:
            if not self.emergency_override_active:
                logger.warning(f"Emergency vehicle override triggered on lane: {emergency_detected_lane}")
                self.emergency_override_active = True
                self.emergency_lane = emergency_detected_lane
                # Transition to Yellow immediately if current state is GREEN and it is not the emergency phase
                if self.current_state == 'GREEN' and self.emergency_lane not in self.phases[self.active_phase_idx]["lanes"]:
                    self.current_state = 'YELLOW'
                    self.timer = YELLOW_TIME
        else:
            if self.emergency_override_active:
                logger.info("Emergency vehicle cleared. Restoring standard cycles.")
                self.emergency_override_active = False
                self.emergency_lane = None
                
        # State machine transition logic when timer expires
        if self.timer <= 0:
            if self.current_state == 'GREEN':
                # Transition to Yellow
                self.current_state = 'YELLOW'
                self.timer = YELLOW_TIME
                logger.info(f"Phase {self.active_phase_idx} ({self.phases[self.active_phase_idx]['name']}) green expired. Transitioning to Yellow.")
                
            elif self.current_state == 'YELLOW':
                # Transition to Red All
                self.current_state = 'RED_ALL'
                self.timer = RED_ALL_TIME
                logger.info("Transitioning to RED-ALL buffer.")
                
            elif self.current_state == 'RED_ALL':
                # Transition to next Green phase
                self.current_state = 'GREEN'
                
                # Determine next phase
                starved_lane = self.check_starvation()
                
                if self.emergency_override_active and self.emergency_lane:
                    # Find phase corresponding to emergency lane
                    next_idx = self.active_phase_idx
                    for idx, phase in enumerate(self.phases):
                        if self.emergency_lane in phase["lanes"]:
                            next_idx = idx
                            break
                    self.active_phase_idx = next_idx
                    logger.warning(f"Override: Selecting Phase {self.active_phase_idx} for emergency lane '{self.emergency_lane}'.")
                elif starved_lane:
                    # Find phase corresponding to starved lane
                    next_idx = self.active_phase_idx
                    for idx, phase in enumerate(self.phases):
                        if starved_lane in phase["lanes"]:
                            next_idx = idx
                            break
                    self.active_phase_idx = next_idx
                    logger.info(f"Starvation override: Selecting Phase {self.active_phase_idx} for starved lane '{starved_lane}'.")
                else:
                    # Standard sequential cycle or density-based priority cycle
                    self.active_phase_idx = self.next_phase_idx
                    self.next_phase_idx = (self.active_phase_idx + 1) % self.num_phases
                    
                # Compute dynamic green duration for new phase
                self.green_duration = self.calculate_green_duration(lane_densities)
                self.timer = self.green_duration
                logger.info(f"Phase {self.active_phase_idx} ({self.phases[self.active_phase_idx]['name']}) active with dynamic Green: {self.timer}s.")

    def get_signal_colors(self):
        """
        Returns a dictionary mapping each lane to its current traffic light color ('RED', 'YELLOW', 'GREEN').
        """
        colors = {}
        active_lanes = self.phases[self.active_phase_idx]["lanes"]
        
        # Populate colors for all configured lanes
        for phase in self.phases:
            for lane in phase["lanes"]:
                if lane in active_lanes:
                    if self.current_state == 'GREEN':
                        colors[lane] = 'GREEN'
                    elif self.current_state == 'YELLOW':
                        colors[lane] = 'YELLOW'
                    else:
                        colors[lane] = 'RED'
                else:
                    colors[lane] = 'RED'
        return colors

def clip(val, min_val, max_val):
    return max(min_val, min(val, max_val))
