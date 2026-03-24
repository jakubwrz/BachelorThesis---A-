# -*- coding: utf-8 -*-
import os
import csv
import math
import random

from PIL import Image

# Import your custom terrain generator (Ensure separate file exists with color fixes)
try:
    from generate_terrain import generate_terrain  
except ImportError:
    print("CRITICAL ERROR: Could not find 'generate_terrain.py'.")
    print("Ensure both files are in the same folder.")
    exit(1)

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
IMG_SIZE = 129
GRID_SIZE = IMG_SIZE

REAL_SIZE  = 50.0           # meters across X/Y (Gazebo size of the terrain)
MAX_HEIGHT = 4.0            # meters vertical span ("size.z" in SDF)

# World building options
HEIGHTMAP_BOTTOM_ALIGNED = False   # False: <pos>0 0 0</pos> (centered)
                                   # True : <pos>0 0 MAX_HEIGHT/2</pos> (bottom aligned)
REMOVE_GROUND_PLANE      = True

# Visuals / road rendering
HOVER_HEIGHT     = 0.05     # small gap above surface to prevent z-fighting
ROAD_THICKNESS   = 0.12
ROAD_WIDTH       = 0.30
RESAMPLE_STEP_M  = 0.30     # road piece length (~0.2..0.5 looks good)
ADD_DEBUG_PINS   = False
PIN_EVERY_N      = 8

# Route (Truly randomized)
START = (random.randrange(0, 90), random.randrange(0, 90))
GOAL  = (115, 115)

# A* cost parameters
MIN_FRICTION    = 0.013
GRAVITY_PENALTY = 10.0
MAX_SLOPE       = 0.5

# Files - SYNCED with generate_terrain.py output names
CURRENT_DIR = os.getcwd()
OUTPUT_TEXTURE      = os.path.join(CURRENT_DIR, "terrain_texture.png") 
OUTPUT_HEIGHT_IMG   = os.path.join(CURRENT_DIR, "heightmap.png")  
OUTPUT_WORLD        = os.path.join(CURRENT_DIR, "thesis_3d_path.world")


# -----------------------------
# 2) PATH VERIFICATION & SMOOTHING TOOLS
# -----------------------------
def get_bresenham_line(x0, y0, x1, y1):
    """Generates a mathematically straight line on a grid."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

def get_dense_grid_cost(path_sparse, friction_map, height_img):
    """Calculates true cost of a path by evaluating every single cell along the lines."""
    total_cost = 0.0
    is_valid = True
    for i in range(len(path_sparse) - 1):
        line = get_bresenham_line(path_sparse[i][0], path_sparse[i][1], path_sparse[i+1][0], path_sparse[i+1][1])
        for j in range(len(line) - 1):
            cost = get_3d_step_cost(line[j], line[j+1], friction_map, height_img)
            if cost == float('inf'):
                is_valid = False
            total_cost += cost
    return total_cost, is_valid

def smooth_path_los(path, friction_map, height_img):
    """
    Cost-Aware String-Pulling.
    Only takes a shortcut if the true terrain cost of the straight line is cheaper 
    than staying on the jagged grid path.
    """
    if len(path) <= 2:
        return path
        
    print("Smoothing path (Cost-Aware String-Pulling)...")
    smoothed_path = [path[0]]
    current_index = 0
    
    while current_index < len(path) - 1:
        furthest_visible = current_index + 1
        
        for j in range(current_index + 2, len(path)):
            # 1. Evaluate the cost of the straight line shortcut
            shortcut_line = [path[current_index], path[j]]
            straight_cost, valid = get_dense_grid_cost(shortcut_line, friction_map, height_img)
            
            # 2. Evaluate the cost of the original A* route for this segment
            astar_segment = path[current_index:j+1]
            astar_cost, _ = get_dense_grid_cost(astar_segment, friction_map, height_img)
            
            # 3. Only pull the string if the shortcut avoids hills/mud!
            # We allow a tiny 2% margin because straight diagonal lines on a grid 
            # naturally measure slightly longer mathematically than human lines.
            if valid and straight_cost <= (astar_cost * 1.02):
                furthest_visible = j
            else:
                # The shortcut is too steep or muddy! Stop pulling the string.
                break
                
        smoothed_path.append(path[furthest_visible])
        current_index = furthest_visible
        
    print(f"Original nodes: {len(path)} -> Smoothed nodes: {len(smoothed_path)}")
    return smoothed_path

def verify_path_costs(astar_path, straight_path_grid, friction_map, height_img):
    """Compares the smoothed A* path against a mathematical straight line."""
    print("\n--- PATH VERIFICATION ---")
    
    # Use our new helper to accurately step through every cell of the smoothed path
    astar_cost, _ = get_dense_grid_cost(astar_path, friction_map, height_img)
    print(f"Smoothed A* Path Cost:  {astar_cost:.4f}")

    # The straight path is already dense, but we can use the same helper
    straight_cost, is_valid = get_dense_grid_cost([straight_path_grid[0], straight_path_grid[-1]], friction_map, height_img)
        
    if not is_valid:
        print("Straight Line Cost:     INVALID (Exceeds MAX_SLOPE)")
        print("Conclusion: Path had to bend because a straight line hit an unscalable cliff.")
    else:
        print(f"Straight Line Cost:     {straight_cost:.4f}")
        diff = straight_cost - astar_cost
        print(f"Difference:             {diff:.4f}")
        if diff > 0:
            print("Conclusion: A* successfully routed around a hill or high-friction area!")
    print("-------------------------\n")


# -----------------------------
# 3) IMAGE-BASED HEIGHT SAMPLING & MAPPING
# -----------------------------
def bilinear_sample_u8(img: Image.Image, u: float, v: float) -> float:
    W, H = img.size
    x = max(0.0, min(1.0, u)) * (W - 1)
    y = max(0.0, min(1.0, v)) * (H - 1)

    x0 = int(math.floor(x)); x1 = min(x0 + 1, W - 1)
    y0 = int(math.floor(y)); y1 = min(y0 + 1, H - 1)
    dx = x - x0; dy = y - y0

    p00 = img.getpixel((x0, y0)) / 255.0
    p10 = img.getpixel((x1, y0)) / 255.0
    p01 = img.getpixel((x0, y1)) / 255.0
    p11 = img.getpixel((x1, y1)) / 255.0

    p0 = p00*(1-dx) + p10*dx
    p1 = p01*(1-dx) + p11*dx
    return p0*(1-dy) + p1*dy

def csv_index_to_uv(grid_x: int, grid_y: int) -> tuple:
    u = grid_x / (GRID_SIZE - 1)
    v = grid_y / (GRID_SIZE - 1)
    return u, v

def world_xy_from_grid(grid_x: int, grid_y: int) -> tuple:
    u, v = csv_index_to_uv(grid_x, grid_y)
    world_x = (u - 0.5) * REAL_SIZE
    world_y = (0.5 - v) * REAL_SIZE
    return world_x, world_y

def world_to_uv(wx: float, wy: float, pos_x=0.0, pos_y=0.0) -> tuple:
    u = (wx - pos_x) / REAL_SIZE + 0.5
    v = 0.5 - (wy - pos_y) / REAL_SIZE
    return u, v

def world_xy_to_grid(wx: float, wy: float) -> tuple:
    """World (x,y) -> Grid indices (gx, gy) to look up friction."""
    u, v = world_to_uv(wx, wy, pos_x=0.0, pos_y=0.0)
    gx = int(max(0, min(GRID_SIZE - 1, round(u * (GRID_SIZE - 1)))))
    gy = int(max(0, min(GRID_SIZE - 1, round(v * (GRID_SIZE - 1)))))
    return gx, gy

def terrain_z_from_world_xy(wx: float, wy: float, height_img: Image.Image,
                            size_z: float, pos_z: float, bottom_aligned: bool) -> float:
    u, v = world_to_uv(wx, wy, pos_x=0.0, pos_y=0.0)
    p = bilinear_sample_u8(height_img, u, v)  
    return pos_z + (p * size_z)


# -----------------------------
# 4) PATH PLANNING (A*) on image heights
# -----------------------------
def get_height_m_from_img(grid_x: int, grid_y: int, height_img: Image.Image) -> float:
    u, v = csv_index_to_uv(grid_x, grid_y)
    p = bilinear_sample_u8(height_img, u, v)
    return p * MAX_HEIGHT

def get_3d_step_cost(curr, nxt, friction_map, height_img: Image.Image):
    cx, cy = curr
    nx, ny = nxt
    dist_2d = math.hypot(nx - cx, ny - cy)
    if dist_2d == 0:
        return float('inf')

    friction = friction_map[nx][ny]
    h_curr = get_height_m_from_img(cx, cy, height_img)
    h_next = get_height_m_from_img(nx, ny, height_img)
    dh = h_next - h_curr
    slope = abs(dh) / dist_2d
    if slope > MAX_SLOPE:
        return float('inf')

    base_energy = dist_2d * friction
    if dh > 0:
        return base_energy + (dh * GRAVITY_PENALTY)
    return base_energy

def heuristic(curr, goal, start, height_img: Image.Image):
    cx, cy = curr
    gx, gy = goal
    sx, sy = start
    
    cz = get_height_m_from_img(cx, cy, height_img)
    gz = get_height_m_from_img(gx, gy, height_img)
    dist_3d = math.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2)
    
    # Base heuristic cost
    h = dist_3d * MIN_FRICTION
    
    # Cross-product tie-breaker to favor straight lines during raw A*
    dx1 = cx - gx
    dy1 = cy - gy
    dx2 = sx - gx
    dy2 = sy - gy
    cross_product = abs(dx1 * dy2 - dx2 * dy1)
    
    return h + (cross_product * 0.0001)

def run_astar(friction_map, height_img: Image.Image):
    import heapq
    open_list = []
    counter = 0
    heapq.heappush(open_list, (0, counter, START))

    came_from = {}
    g_score = {START: 0.0}
    f_score = {START: heuristic(START, GOAL, START, height_img)}
    closed = set()

    while open_list:
        current = heapq.heappop(open_list)[2]
        if current == GOAL:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(START)
            path.reverse()
            return path

        closed.add(current)
        cx, cy = current
        neighbors = [
            (cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy),
            (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)
        ]
        for nx, ny in neighbors:
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if (nx, ny) in closed:
                continue
            step = get_3d_step_cost(current, (nx, ny), friction_map, height_img)
            if step == float('inf'):
                continue
            tentative_g = g_score[current] + step
            if tentative_g < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), GOAL, START, height_img)
                counter += 1
                heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))

    return []  # no path


# -----------------------------
# 5) PATH RESAMPLING -> DENSE WORLD POLYLINE
# -----------------------------
def resample_polyline_world(path_grid, height_img: Image.Image,
                            step_m: float, size_z: float, pos_z: float, bottom_aligned: bool):
    dense = []
    for i in range(len(path_grid) - 1):
        x1, y1 = path_grid[i]
        x2, y2 = path_grid[i + 1]
        w1x, w1y = world_xy_from_grid(x1, y1)
        w2x, w2y = world_xy_from_grid(x2, y2)
        dx = w2x - w1x
        dy = w2y - w1y
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        n_steps = max(1, int(math.ceil(seg_len / step_m)))
        for k in range(n_steps):
            t = k / n_steps
            wx = w1x + t * dx
            wy = w1y + t * dy
            wz = terrain_z_from_world_xy(wx, wy, height_img, size_z, pos_z, bottom_aligned)
            dense.append((wx, wy, wz))
    w2x, w2y = world_xy_from_grid(path_grid[-1][0], path_grid[-1][1])
    wz_end   = terrain_z_from_world_xy(w2x, w2y, height_img, size_z, pos_z, bottom_aligned)
    dense.append((w2x, w2y, wz_end))
    return dense


# -----------------------------
# 6) WORLD EXPORT (SDF)
# -----------------------------
def build_world_sdf(astar_path, straight_path_grid, height_img: Image.Image, friction_map):
    HEIGHT_URI = "file://" + os.path.abspath(OUTPUT_HEIGHT_IMG)
    TEX_URI    = "file://" + os.path.abspath(OUTPUT_TEXTURE)

    pos_z_for_sdf = (MAX_HEIGHT / 2.0) if HEIGHTMAP_BOTTOM_ALIGNED else 0.0

    world_header = """<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <scene><ambient>0.6 0.6 0.6 1</ambient><shadows>true</shadows><grid>false</grid></scene>
"""
    if not REMOVE_GROUND_PLANE:
        world_header += '    <include><uri>model://ground_plane</uri></include>\n'

    terrain_model = f"""
    <model name="thesis_terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>{HEIGHT_URI}</uri>
              <size>{REAL_SIZE} {REAL_SIZE} {MAX_HEIGHT}</size>
              <pos>0 0 {pos_z_for_sdf}</pos>
            </heightmap>
          </geometry>
          <surface>
            <friction>
              <ode><mu>100</mu><mu2>50</mu2></ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <use_terrain_paging>false</use_terrain_paging>
              <texture>
                <diffuse>{TEX_URI}</diffuse>
                <normal>file://media/materials/textures/flat_normal.png</normal>
                <size>{REAL_SIZE}</size>
              </texture>
              <uri>{HEIGHT_URI}</uri>
              <size>{REAL_SIZE} {REAL_SIZE} {MAX_HEIGHT}</size>
              <pos>0 0 {pos_z_for_sdf}</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
"""

    # Start / Goal markers
    sx, sy = astar_path[0]
    gx, gy = astar_path[-1]
    sxw, syw = world_xy_from_grid(sx, sy)
    gxw, gyw = world_xy_from_grid(gx, gy)

    sz = terrain_z_from_world_xy(sxw, syw, height_img, MAX_HEIGHT, pos_z_for_sdf, HEIGHTMAP_BOTTOM_ALIGNED)
    gz = terrain_z_from_world_xy(gxw, gyw, height_img, MAX_HEIGHT, pos_z_for_sdf, HEIGHTMAP_BOTTOM_ALIGNED)

    start_model = f"""
    <model name="start_marker">
      <static>true</static>
      <pose>{sxw} {syw} {sz + HOVER_HEIGHT + 0.25} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.4</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0 1 1 1</ambient><diffuse>0 1 1 1</diffuse><emissive>0 1 1 1</emissive></material>
        </visual>
      </link>
    </model>
"""
    goal_model = f"""
    <model name="goal_marker">
      <static>true</static>
      <pose>{gxw} {gyw} {gz + HOVER_HEIGHT + 0.25} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.4</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse><emissive>1 0 0 1</emissive></material>
        </visual>
      </link>
    </model>
"""

    # --- PART 1: A* Road (Hardcoded Yellow) ---
    dense_pts = resample_polyline_world(astar_path, height_img, RESAMPLE_STEP_M,
                                        MAX_HEIGHT, pos_z_for_sdf, HEIGHTMAP_BOTTOM_ALIGNED)
    road_models = []

    for i in range(len(dense_pts) - 1):
        x1g, y1g, z1g = dense_pts[i]
        x2g, y2g, z2g = dense_pts[i + 1]

        mid_x = (x1g + x2g) / 2.0
        mid_y = (y1g + y2g) / 2.0
        mid_z = (z1g + z2g) / 2.0 + HOVER_HEIGHT + (ROAD_THICKNESS / 2.0)

        # Physics (Friction) Lookup
        gx_grid, gy_grid = world_xy_to_grid(mid_x, mid_y)
        mu = friction_map[gx_grid][gy_grid]
        
        dx = x2g - x1g
        dy = y2g - y1g
        dz = z2g - z1g

        dist_2d = math.hypot(dx, dy)
        length = math.hypot(dist_2d, dz)
        yaw   = math.atan2(dy, dx)
        pitch = -math.atan2(dz, dist_2d) if dist_2d > 1e-6 else 0.0

        road_models.append(f"""
    <model name="road_segment_{i}">
      <static>true</static>
      <pose>{mid_x} {mid_y} {mid_z} 0 {pitch} {yaw}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{length} {ROAD_WIDTH} {ROAD_THICKNESS}</size></box></geometry>
          <surface>
            <friction>
              <ode><mu>{mu}</mu><mu2>{mu}</mu2></ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{length} {ROAD_WIDTH} {ROAD_THICKNESS}</size></box></geometry>
          <material>
            <ambient>1 1 0 1</ambient> <diffuse>1 1 0 1</diffuse>
            <emissive>1 1 0 1</emissive>
          </material>
        </visual>
      </link>
    </model>
""")

    # --- PART 2: Direct Straight Line (Magenta spheres) ---
    dense_straight_pts = resample_polyline_world(straight_path_grid, height_img, 0.5, 
                                                MAX_HEIGHT, pos_z_for_sdf, HEIGHTMAP_BOTTOM_ALIGNED)
    straight_line_models = []
    
    for j, (wx, wy, wz) in enumerate(dense_straight_pts):
        straight_line_models.append(f"""
    <model name="straight_marker_{j}">
      <static>true</static>
      <pose>{wx} {wy} {wz + 0.1} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><sphere><radius>0.05</radius></sphere></geometry>
          <material>
            <ambient>1 0 1 1</ambient> <diffuse>1 0 1 1</diffuse>
            <emissive>1 0 1 1</emissive>
          </material>
        </visual>
      </link>
    </model>
""")

    world_footer = """
  </world>
</sdf>
"""
    return (world_header + terrain_model + start_model + goal_model + 
            "".join(road_models) + "".join(straight_line_models) + world_footer)


# -----------------------------
# 7) MAIN
# -----------------------------
def main():
    print(f"Truly Randomized Start: {START}, Goal: {GOAL}")

    map_type = input("Enter scenario type (mix/friction/hill): ").strip().lower()

    # 1) Generate terrain maps using separate module
    friction_map, height_map, height_img = generate_terrain(map_type)
    
    # 2) Plan RAW A* path on image heights (meters)
    raw_path = run_astar(friction_map, height_img)
    if not raw_path:
        print("NO PATH FOUND.")
        return
    print(f"Path length (raw grid nodes): {len(raw_path)}")

    # 3) SMOOTH THE PATH (Cost-Aware String-Pulling)
    smoothed_path = smooth_path_los(raw_path, friction_map, height_img)

    # 4) Generate mathematically direct line grid coordinates early for use in Verify and Build
    straight_line_grid = get_bresenham_line(smoothed_path[0][0], smoothed_path[0][1], smoothed_path[-1][0], smoothed_path[-1][1])

    # 5) VERIFY the Smoothed A* cost vs a straight line in terminal output
    verify_path_costs(smoothed_path, straight_line_grid, friction_map, height_img)

    # 6) Build SDF world with visual markers for both paths
    sdf = build_world_sdf(smoothed_path, straight_line_grid, height_img, friction_map)
    with open(OUTPUT_WORLD, "w") as f:
        f.write(sdf)

    print(f"World saved to: {OUTPUT_WORLD}")
    print(f"Open with: gazebo {OUTPUT_WORLD}")

if __name__ == "__main__":
    main()