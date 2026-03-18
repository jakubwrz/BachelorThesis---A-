import csv
import math
import heapq
import os
import random

# --- CONFIGURATION ---
GRID_SIZE = 129
REAL_SIZE = 50.0
MAX_HEIGHT = 4.0

# Start/Goal in Grid Coordinates
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
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE): continue
            if (nx, ny) in closed_set: continue
            step_cost = get_3d_step_cost(current, (nx, ny), friction_map, height_map)
            if step_cost == float('inf'): continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), GOAL, height_map)
                counter += 1
                heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))
                
    print("NO PATH FOUND.")
    return []

def grid_to_gazebo(grid_x, grid_y, grid_z):
    # Perfect alignment for a CCW 90-degree rotated Gazebo Heightmap
    gazebo_x = -(grid_y / (GRID_SIZE - 1)) * REAL_SIZE + (REAL_SIZE / 2.0)
    gazebo_y = -(grid_x / (GRID_SIZE - 1)) * REAL_SIZE + (REAL_SIZE / 2.0)
    return gazebo_x, gazebo_y, grid_z

def export_to_gazebo(path, height_map):
    print("Injecting route and robot into Gazebo world...")
    HOVER_HEIGHT = 0.4  # Sufficient height to stay above terrain mesh curves

    with open(INPUT_WORLD, 'r') as file:
        world_data = file.read()

    # Appending markers to the world file
    world_data = world_data.replace("  </world>\n</sdf>", "")

    # 1. Start marker
    start_x, start_y, start_z = grid_to_gazebo(path[0][0], path[0][1], height_map[path[0][0]][path[0][1]])
    world_data += f"""
    <model name="start_marker">
      <static>true</static>
      <pose>{start_x} {start_y} {start_z + HOVER_HEIGHT + 0.25} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.4</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0 1 1 1</ambient><diffuse>0 1 1 1</diffuse><emissive>0 0.9 0.9 1</emissive></material>
        </visual>
      </link>
    </model>
    """

    # 2. Goal marker
    goal_x, goal_y, goal_z = grid_to_gazebo(GOAL[0], GOAL[1], height_map[GOAL[0]][GOAL[1]])
    world_data += f"""
    <model name="goal_marker">
      <static>true</static>
      <pose>{goal_x} {goal_y} {goal_z + HOVER_HEIGHT + 0.25} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.4</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse><emissive>1 0 0 1</emissive></material>
        </visual>
      </link>
    </model>
    """

    # 3. Road segments
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        z1, z2 = height_map[x1][y1], height_map[x2][y2]
        x1g, y1g, z1g = grid_to_gazebo(x1, y1, z1)
        x2g, y2g, z2g = grid_to_gazebo(x2, y2, z2)
        mid_x, mid_y, mid_z = (x1g + x2g) / 2.0, (y1g + y2g) / 2.0, (z1g + z2g) / 2.0 + HOVER_HEIGHT
        dx, dy, dz = x2g - x1g, y2g - y1g, z2g - z1g
        dist_2d = math.hypot(dx, dy)
        length = math.hypot(dist_2d, dz) 
        yaw, pitch = math.atan2(dy, dx), -math.atan2(dz, dist_2d) 

        world_data += f"""
    <model name="road_segment_{i}">
      <static>true</static>
      <pose>{mid_x} {mid_y} {mid_z} 0 {pitch} {yaw}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{length} 0.25 0.05</size></box></geometry>
          <material><ambient>0.8 0.8 0 1</ambient><diffuse>0.9 0.9 0 1</diffuse><emissive>0.9 0.9 0 1</emissive></material>
        </visual>
      </link>
    </model>
    """

    # 4. Robot Trajectory
    world_data += f"""
    <actor name="my_robot">
      <pose>{start_x} {start_y} {start_z + HOVER_HEIGHT + 0.1} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>0.6 0.4 0.2</size></box></geometry>
          <material><ambient>0 0 1 1</ambient><diffuse>0 0 1 1</diffuse></material>
        </visual>
      </link>
      <script>
        <loop>true</loop><delay_start>0.0</delay_start><auto_start>true</auto_start>
        <trajectory id="0" type="driving">
"""
    total_time = len(path) * 0.5
    time_step = total_time / (len(path) - 1) if len(path) > 1 else 0
    curr_time = 0.0

    for idx, (gx, gy) in enumerate(path):
        gz = height_map[gx][gy]
        wx, wy, wz = grid_to_gazebo(gx, gy, gz)
        wz += HOVER_HEIGHT + 0.1  # Box height offset
        if idx < len(path) - 1:
            next_gx, next_gy = path[idx + 1]
            next_gz = height_map[next_gx][next_gy]
            next_wx, next_wy, _ = grid_to_gazebo(next_gx, next_gy, 0)
            dx, dy, dz = next_wx - wx, next_wy - wy, next_gz - gz
            yaw, pitch = math.atan2(dy, dx), -math.atan2(dz, math.hypot(dx, dy))
        else: yaw, pitch = 0.0, 0.0
        world_data += f"""          <waypoint><time>{curr_time:.2f}</time><pose>{wx} {wy} {wz} 0 {pitch} {yaw}</pose></waypoint>\n"""
        curr_time += time_step

    world_data += """        </trajectory>\n      </script>\n    </actor>\n  </world>\n</sdf>"""
    with open(OUTPUT_WORLD, 'w') as file: file.write(world_data)
    print(f"Route visualization saved to: {OUTPUT_WORLD}")

if __name__ == "__main__":
    with open(FRICTION_FILE, 'r') as f: friction_map = [[float(v) for v in r] for r in csv.reader(f)]
    with open(HEIGHT_FILE, 'r') as f: height_map = [[float(v) for v in r] for r in csv.reader(f)]
    path = run_astar(friction_map, height_map)
    if path: export_to_gazebo(path, height_map)