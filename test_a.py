import csv
import math
import heapq
import os
import random

GRID_SIZE = 129
REAL_SIZE = 50.0
MAX_HEIGHT = 4.0

START = (random.randrange(0, 90), random.randrange(0, 90))
GOAL = (115, 115)
print(f"Start: {START}, Goal: {GOAL}")

MIN_FRICTION = 0.013
GRAVITY_PENALTY = 10.0
MAX_SLOPE = 0.5
FRICTION_FILE = "friction_map.csv"
HEIGHT_FILE = "height_map.csv"
INPUT_WORLD = "thesis_3d.world"
OUTPUT_WORLD = "thesis_3d_path.world"

def get_3d_step_cost(curr, next_node, friction_map, height_map):
    cx, cy = curr
    nx, ny = next_node
    dist_2d = math.hypot(nx - cx, ny - cy)
    friction = friction_map[nx][ny]
    h_current = height_map[cx][cy]
    h_next = height_map[nx][ny]
    delta_h = h_next - h_current
    
    # Check if too steep
    if dist_2d == 0: return float('inf')
    slope = abs(delta_h) / dist_2d
    if slope > MAX_SLOPE:
        return float('inf')

    base_energy = dist_2d * friction
    
    # Add gravity penalty only if going uphill
    if delta_h > 0:
        return base_energy + (delta_h * GRAVITY_PENALTY)
    return base_energy

def heuristic(curr, goal, height_map):
    cx, cy = curr
    gx, gy = goal
    
    cz = height_map[cx][cy]
    gz = height_map[gx][gy]
    
    dist_3d = math.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2)
    return dist_3d * MIN_FRICTION
def run_astar(friction_map, height_map):
    print(f"Starting A* search from {START} to {GOAL}...")
    
    open_list = []
    counter = 0 
    heapq.heappush(open_list, (0, counter, START))
    
    came_from = {}
    
    g_score = {START: 0}
    f_score = {START: heuristic(START, GOAL, height_map)}
    
    closed_set = set()

    while open_list:
        current = heapq.heappop(open_list)[2]
        
        if current == GOAL:
            print("Path Found!")
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(START)
            path.reverse()
            return path
            
        closed_set.add(current)
        
        cx, cy = current
        neighbors = [(cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy),
                     (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)]
                     
        for nx, ny in neighbors:
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if (nx, ny) in closed_set:
                continue
                
            step_cost = get_3d_step_cost(current, (nx, ny), friction_map, height_map)
            if step_cost == float('inf'):
                continue
                
            tentative_g = g_score[current] + step_cost
            
            if tentative_g < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), GOAL, height_map)
                
                counter += 1
                heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))
                
    print("NO PATH FOUND. The goal might be surrounded by steep hills.")
    return []
def grid_to_gazebo(grid_x, grid_y, grid_z):
    gazebo_x = (grid_y / (GRID_SIZE - 1)) * REAL_SIZE - (REAL_SIZE / 2.0)
    gazebo_y = -(grid_x / (GRID_SIZE - 1)) * REAL_SIZE + (REAL_SIZE / 2.0)
    return gazebo_x, gazebo_y, grid_z

def export_to_gazebo(path, height_map):
    print("Injecting route and robot into Gazebo world...")
    
    with open(INPUT_WORLD, 'r') as file:
        world_data = file.read()
        
    # Remove the closing </world></sdf> tags to append our markers
    world_data = world_data.replace("  </world>\n</sdf>", "")
    
    # 1. Spawn a "Robot" (A blue box) at the Start
    start_x, start_y, start_z = grid_to_gazebo(path[0][0], path[0][1], height_map[path[0][0]][path[0][1]])
    
    world_data += f"""
    <model name="my_robot">
      <pose>{start_x} {start_y} {start_z + 0.3} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>0.6 0.4 0.2</size></box></geometry>
          <material><ambient>0 0 1 1</ambient></material> </visual>
      </link>
    </model>
    """

    # 2. Spawn glowing spheres along the path
    # We step by 3 so we don't spawn 1000 spheres and crash Gazebo
    for i in range(0, len(path), 3):
        gx, gy = path[i]
        gz = height_map[gx][gy]
        pos_x, pos_y, pos_z = grid_to_gazebo(gx, gy, gz)
        
        world_data += f"""
    <model name="path_marker_{i}">
      <static>true</static>
      <pose>{pos_x} {pos_y} {pos_z + 3} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><sphere><radius>0.15</radius></sphere></geometry>
          <material>
            <ambient>0 1 0 1</ambient> <emissive>0 1 0 1</emissive> </material>
        </visual>
      </link>
    </model>
    """
    
    world_data += "  </world>\n</sdf>"
    
    with open(OUTPUT_WORLD, 'w') as file:
        file.write(world_data)
        
    print(f"Route visualization saved to: {OUTPUT_WORLD}")
if __name__ == "__main__":
    # Load CSV Data
    print("Loading map data...")
    with open(FRICTION_FILE, 'r') as f:
        friction_map = [[float(val) for val in row] for row in csv.reader(f)]
        
    with open(HEIGHT_FILE, 'r') as f:
        height_map = [[float(val) for val in row] for row in csv.reader(f)]
        
    # Run Pathfinding
    path = run_astar(friction_map, height_map)
    
    if path:
        print(f"Path length: {len(path)} steps.")
        export_to_gazebo(path, height_map)
