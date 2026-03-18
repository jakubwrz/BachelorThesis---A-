# -*- coding: utf-8 -*-
"""
One-pass terrain -> path planning -> world export (Gazebo).

This merges your generate_terrain2.py and test_a.py:
- Generates texture/heightmap images + CSVs (for inspection)
- Plans A* path directly on in-memory arrays
- Writes a single .world with road + actor aligned to the heightmap surface

Key fix:
- grid_to_gazebo() adds +MAX_HEIGHT/2 to Z so the path/robot sit on the surface.

Jakub’s original details preserved:
- Perlin noise params, color/friction mapping, gamma for height *visual*
- A* cost model (slope cap, gravity penalty), yaw/pitch along the path
"""

import os
import csv
import math
import random

from PIL import Image
import noise

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
IMG_SIZE = 129
GRID_SIZE = IMG_SIZE

REAL_SIZE = 50.0           # meters across X/Y (Gazebo size of the terrain)
MAX_HEIGHT = 4.0           # meters peak-to-peak height of the terrain mesh

NOISE_SCALE = 20.0
SEED = 42

# File outputs (also kept for debugging/inspection)
CURRENT_DIR = os.getcwd()
OUTPUT_TEXTURE      = os.path.join(CURRENT_DIR, "terrain_texture.png")
OUTPUT_HEIGHT_IMG   = os.path.join(CURRENT_DIR, "heightmap.png")
OUTPUT_WORLD        = os.path.join(CURRENT_DIR, "thesis_3d_path.world")
OUTPUT_FRICTION_CSV = os.path.join(CURRENT_DIR, "friction_map.csv")
OUTPUT_HEIGHT_CSV   = os.path.join(CURRENT_DIR, "height_map.csv")

# Visual materials
COLOR_ASPHALT = (50, 50, 50)
COLOR_GRASS   = (34, 139, 34)
COLOR_MUD     = (101, 67, 33)

# Route & robot
# (You can hard-code START if you want a stable start; here it's reproducible via SEED)
random.seed(1234)
START = (random.randrange(0, 90), random.randrange(0, 90))
GOAL  = (115, 115)

# A* cost parameters (from your test_a.py)
MIN_FRICTION    = 0.013
GRAVITY_PENALTY = 10.0
MAX_SLOPE       = 0.5

# Geometry placement
HOVER_HEIGHT = 0.10        # small “lift” above surface for path and actor
REMOVE_GROUND_PLANE = True # recommended with heightmap terrain

# -----------------------------
# 2) TERRAIN GENERATION
# -----------------------------
def get_terrain_data(val):
    """
    Return (color, friction) for a given Perlin value.
    Matches your original thresholds.
    """
    if val < -0.05:
        return COLOR_ASPHALT, 0.013
    elif val < 0.3:
        return COLOR_GRASS, 0.05
    else:
        return COLOR_MUD, 0.35

def generate_terrain():
    """
    Produces:
      - terrain_texture.png (for the visual)
      - heightmap.png       (for the heightmap geometry)
      - friction_map.csv, height_map.csv (for inspection/consistency)
    Returns in-memory arrays: friction_map, height_map (meters in [0..MAX_HEIGHT]).
    """
    img_texture = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
    img_height  = Image.new('L',   (IMG_SIZE, IMG_SIZE))
    p_tex = img_texture.load()
    p_hgt = img_height.load()

    friction_map = []
    height_map_m = []  # physical height in meters (0 .. MAX_HEIGHT)

    for x in range(IMG_SIZE):
        row_f = []
        row_h = []
        for y in range(IMG_SIZE):
            # Texture & friction noise
            val_tex = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE,
                                    octaves=6, persistence=0.5, lacunarity=2.0, base=SEED)
            color, friction = get_terrain_data(val_tex)
            p_tex[x, y] = color
            row_f.append(friction)

            # Height noise
            val_hgt = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE,
                                    octaves=4, persistence=0.5, lacunarity=2.0, base=SEED+100)
            normalized_0_1 = (val_hgt + 1.0) / 2.0                # 0..1
            physical_h_m   = normalized_0_1 * MAX_HEIGHT          # 0..MAX_HEIGHT

            # Gamma for *visual* height image only (geometry still comes from this image,
            # but our path z-values are from 'physical_h_m', i.e., pre-gamma)
            gamma_corrected = math.pow(normalized_0_1, 1.0/2.2)
            px = int(max(0, min(255, gamma_corrected * 255)))
            p_hgt[x, y] = px

            row_h.append(round(physical_h_m, 3))
        friction_map.append(row_f)
        height_map_m.append(row_h)

    # Save reference images & CSVs for inspection
    img_texture.save(OUTPUT_TEXTURE)
    img_height.save(OUTPUT_HEIGHT_IMG)
    with open(OUTPUT_FRICTION_CSV, "w", newline='') as f:
        csv.writer(f).writerows(friction_map)
    with open(OUTPUT_HEIGHT_CSV, "w", newline='') as f:
        csv.writer(f).writerows(height_map_m)

    return friction_map, height_map_m

# -----------------------------
# 3) PATH PLANNING (A*)
# -----------------------------
def get_3d_step_cost(curr, nxt, friction_map, height_map):
    cx, cy = curr
    nx, ny = nxt
    dist_2d = math.hypot(nx - cx, ny - cy)
    if dist_2d == 0:
        return float('inf')

    friction = friction_map[nx][ny]
    h_curr = height_map[cx][cy]
    h_next = height_map[nx][ny]
    dh = h_next - h_curr
    slope = abs(dh) / dist_2d
    if slope > MAX_SLOPE:
        return float('inf')

    base_energy = dist_2d * friction
    if dh > 0:
        return base_energy + (dh * GRAVITY_PENALTY)
    return base_energy

def heuristic(curr, goal, height_map):
    cx, cy = curr
    gx, gy = goal
    cz = height_map[cx][cy]
    gz = height_map[gx][gy]
    dist_3d = math.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2)
    return dist_3d * MIN_FRICTION

def run_astar(friction_map, height_map):
    import heapq
    open_list = []
    counter = 0
    heapq.heappush(open_list, (0, counter, START))

    came_from = {}
    g_score = {START: 0.0}
    f_score = {START: heuristic(START, GOAL, height_map)}
    closed = set()

    while open_list:
        current = heapq.heappop(open_list)[2]
        if current == GOAL:
            # reconstruct
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
            step = get_3d_step_cost(current, (nx, ny), friction_map, height_map)
            if step == float('inf'):
                continue
            tentative_g = g_score[current] + step
            if tentative_g < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), GOAL, height_map)
                counter += 1
                heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))

    return []  # no path

# -----------------------------
# 4) WORLD EXPORT
# -----------------------------
def grid_to_gazebo(grid_x, grid_y, grid_z):
    """
    Map grid indices -> Gazebo world (meters).
    X/Y mapping matches your existing orientation.

    **Critical Z fix (Option A):**
    The heightmap mesh in Gazebo is centered around the link's origin,
    so we add +MAX_HEIGHT/2 to pose Z so objects sit on the visible surface.
    """
    gazebo_x = (grid_y / (GRID_SIZE - 1)) * REAL_SIZE - (REAL_SIZE / 2.0)
    gazebo_y = -(grid_x / (GRID_SIZE - 1)) * REAL_SIZE + (REAL_SIZE / 2.0)

    z_offset = +MAX_HEIGHT / 2.0     # <<< IMPORTANT: positive offset
    gazebo_z = grid_z + z_offset

    return gazebo_x, gazebo_y, gazebo_z

def build_world_sdf(path, height_map):
    """
    Compose a full world SDF string with:
    - Heightmap (visual + collision) at <pos> 0 0 0
    - No ground_plane
    - Start/goal markers
    - Road segments following the surface
    - Actor following the same waypoints
    """
    world_header = """<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <scene><ambient>0.6 0.6 0.6 1</ambient><shadows>true</shadows></scene>
"""
    if not REMOVE_GROUND_PLANE:
        world_header += '    <include><uri>model://ground_plane</uri></include>\n'

    # Heightmap SDF at z=0 (we do all Z offsetting in poses)
    terrain_model = f"""
    <model name="thesis_terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>file://{OUTPUT_HEIGHT_IMG}</uri>
              <size>{REAL_SIZE} {REAL_SIZE} {MAX_HEIGHT}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu><mu2>50</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <use_terrain_paging>false</use_terrain_paging>
              <texture>
                <diffuse>file://{OUTPUT_TEXTURE}</diffuse>
                <normal>file://media/materials/textures/flat_normal.png</normal>
                <size>{REAL_SIZE}</size>
              </texture>
              <uri>file://{OUTPUT_HEIGHT_IMG}</uri>
              <size>{REAL_SIZE} {REAL_SIZE} {MAX_HEIGHT}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
"""

    # Markers
    sx, sy = path[0]
    gx, gy = path[-1]
    sz = height_map[sx][sy]
    gz = height_map[gx][gy]

    start_x, start_y, start_z = grid_to_gazebo(sx, sy, sz)
    goal_x,  goal_y,  goal_z  = grid_to_gazebo(gx, gy, gz)

    start_model = f"""
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
    goal_model = f"""
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

    # Road segments following the surface
    road_models = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        z1 = height_map[x1][y1]
        z2 = height_map[x2][y2]

        x1g, y1g, z1g = grid_to_gazebo(x1, y1, z1)
        x2g, y2g, z2g = grid_to_gazebo(x2, y2, z2)

        mid_x = (x1g + x2g) / 2.0
        mid_y = (y1g + y2g) / 2.0
        mid_z = (z1g + z2g) / 2.0 + HOVER_HEIGHT

        dx = x2g - x1g
        dy = y2g - y1g
        dz = z2g - z1g
        dist_2d = math.hypot(dx, dy)
        length = math.hypot(dist_2d, dz)
        yaw = math.atan2(dy, dx)
        pitch = -math.atan2(dz, dist_2d) if dist_2d > 1e-6 else 0.0

        road_models.append(f"""
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
""")

    # Actor trajectory (waypoints along the path)
    total_time = len(path) * 0.5
    time_step = total_time / (len(path) - 1) if len(path) > 1 else 0.0

    actor_block = f"""
    <actor name="my_robot">
      <pose>{start_x} {start_y} {start_z + HOVER_HEIGHT + 0.1} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>0.6 0.4 0.2</size></box></geometry>
          <material><ambient>0 0 1 1</ambient><diffuse>0 0 1 1</diffuse></material>
        </visual>
      </link>
      <script>
        <loop>true</loop>
        <delay_start>0.0</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="driving">
"""

    current_time = 0.0
    for idx, (gx, gy) in enumerate(path):
        gz = height_map[gx][gy]
        wx, wy, wz = grid_to_gazebo(gx, gy, gz)
        wz += HOVER_HEIGHT + 0.1

        # Orientation towards next point
        if idx < len(path) - 1:
            ngx, ngy = path[idx + 1]
            ngz = height_map[ngx][ngy]
            nwx, nwy, _ = grid_to_gazebo(ngx, ngy, 0.0)
            dx = nwx - wx
            dy = nwy - wy
            dz = (ngz - gz)  # world z difference uses same +offset -> cancels
            dist_2d = math.hypot(dx, dy)
            yaw = math.atan2(dy, dx)
            pitch = -math.atan2(dz, dist_2d) if dist_2d > 1e-6 else 0.0
        else:
            yaw = 0.0
            pitch = 0.0

        actor_block += f"""          <waypoint>
            <time>{current_time:.2f}</time>
            <pose>{wx} {wy} {wz} 0 {pitch} {yaw}</pose>
          </waypoint>
"""
        current_time += time_step

    actor_block += """        </trajectory>
      </script>
    </actor>
"""

    world_footer = """
  </world>
</sdf>
"""

    world = world_header + terrain_model + start_model + goal_model + "".join(road_models) + actor_block + world_footer
    return world

def main():
    print(f"Start: {START}, Goal: {GOAL}")

    # 1) Generate terrain
    friction_map, height_map = generate_terrain()

    # 2) Plan path
    path = run_astar(friction_map, height_map)
    if not path:
        print("NO PATH FOUND.")
        return
    print(f"Path length: {len(path)}")

    # 3) Build SDF & write world
    sdf = build_world_sdf(path, height_map)
    with open(OUTPUT_WORLD, "w") as f:
        f.write(sdf)

    print(f"World saved to: {OUTPUT_WORLD}")

if __name__ == "__main__":
    main()