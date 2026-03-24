import os
import csv
import math
import random

from PIL import Image
import noise

from generate_terrain import generate_terrain  

IMG_SIZE = 129
GRID_SIZE = IMG_SIZE

REAL_SIZE  = 50.0           # meters across X/Y (Gazebo size of the terrain)
MAX_HEIGHT = 4.0            # meters vertical span ("size.z" in SDF)

NOISE_SCALE = 20.0
SEED = 42                   # Perlin base seed

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

# Route (start randomized but reproducible)
random.seed(12)
START = (random.randrange(0, 90), random.randrange(0, 90))
GOAL  = (115, 115)

# A* cost parameters
MIN_FRICTION    = 0.013
GRAVITY_PENALTY = 10.0
MAX_SLOPE       = 0.5

# Files
CURRENT_DIR = os.getcwd()
OUTPUT_TEXTURE      = os.path.join(CURRENT_DIR, "terrain_texture.png")
OUTPUT_HEIGHT_IMG   = os.path.join(CURRENT_DIR, "heightmap.png")
OUTPUT_WORLD        = os.path.join(CURRENT_DIR, "thesis_3d_path.world")
OUTPUT_FRICTION_CSV = os.path.join(CURRENT_DIR, "friction_map.csv")
OUTPUT_HEIGHT_CSV   = os.path.join(CURRENT_DIR, "height_map.csv")


# -----------------------------
# 2) TERRAIN GENERATION
# -----------------------------
def get_terrain_data(val):
    """Return (color, friction) for a given Perlin value."""
    if val < -0.05:
        return (50, 50, 50), 0.013      # asphalt
    elif val < 0.3:
        return (34, 139, 34), 0.05      # grass
    else:
        return (101, 67, 33), 0.35      # mud



# -----------------------------
# 3) IMAGE-BASED HEIGHT SAMPLING
# -----------------------------
def bilinear_sample_u8(img: Image.Image, u: float, v: float) -> float:
    """Bilinear sample grayscale image at normalized coords u,v in [0,1]; returns p∈[0,1]."""
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
    """Grid indices -> normalized image coords (u,v)."""
    u = grid_x / (GRID_SIZE - 1)
    v = grid_y / (GRID_SIZE - 1)
    return u, v

def world_xy_from_grid(grid_x: int, grid_y: int) -> tuple:
    """Grid indices -> world (x,y), image centered over REAL_SIZE with Y flipped."""
    u, v = csv_index_to_uv(grid_x, grid_y)
    world_x = (u - 0.5) * REAL_SIZE
    world_y = (0.5 - v) * REAL_SIZE
    return world_x, world_y

def world_to_uv(wx: float, wy: float, pos_x=0.0, pos_y=0.0) -> tuple:
    """World (x,y) -> normalized (u,v) given <size>REAL_SIZE REAL_SIZE and <pos>.x/y (here zero)."""
    u = (wx - pos_x) / REAL_SIZE + 0.5
    v = 0.5 - (wy - pos_y) / REAL_SIZE
    return u, v

def terrain_z_from_world_xy(wx: float, wy: float, height_img: Image.Image,
                            size_z: float, pos_z: float, bottom_aligned: bool) -> float:
    """
    Sample the PNG at (wx,wy) and map to world Z using the exact SDF formula.
    Gazebo heightmaps extrude UPWARDS from their local Z origin, they do not center.
    """
    u, v = world_to_uv(wx, wy, pos_x=0.0, pos_y=0.0)
    p = bilinear_sample_u8(height_img, u, v)  # [0..1]
    
    # Regardless of orientation flags, Z is simply the base Z + (pixel * total_height)
    return pos_z + (p * size_z)


# -----------------------------
# 4) PATH PLANNING (A*) on image heights
# -----------------------------
def get_height_m_from_img(grid_x: int, grid_y: int, height_img: Image.Image) -> float:
    """Return physical height in meters (0..MAX_HEIGHT) from image."""
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
    h_curr = get_height_m_from_img(cx, cy, height_img)  # meters [0..MAX_HEIGHT]
    h_next = get_height_m_from_img(nx, ny, height_img)
    dh = h_next - h_curr
    slope = abs(dh) / dist_2d
    if slope > MAX_SLOPE:
        return float('inf')

    base_energy = dist_2d * friction
    if dh > 0:
        return base_energy + (dh * GRAVITY_PENALTY)
    return base_energy

def heuristic(curr, goal, height_img: Image.Image):
    cx, cy = curr
    gx, gy = goal
    cz = get_height_m_from_img(cx, cy, height_img)
    gz = get_height_m_from_img(gx, gy, height_img)
    dist_3d = math.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2)
    return dist_3d * MIN_FRICTION

def run_astar(friction_map, height_img: Image.Image):
    import heapq
    open_list = []
    counter = 0
    heapq.heappush(open_list, (0, counter, START))

    came_from = {}
    g_score = {START: 0.0}
    f_score = {START: heuristic(START, GOAL, height_img)}
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
            step = get_3d_step_cost(current, (nx, ny), friction_map, height_img)
            if step == float('inf'):
                continue
            tentative_g = g_score[current] + step
            if tentative_g < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), GOAL, height_img)
                counter += 1
                heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))

    return []  # no path


# -----------------------------
# 5) PATH RESAMPLING -> DENSE WORLD POLYLINE
# -----------------------------
def resample_polyline_world(path_grid, height_img: Image.Image,
                            step_m: float, size_z: float, pos_z: float, bottom_aligned: bool):
    """
    Convert grid path [(gx,gy), ...] to dense world polyline [(wx,wy,wz), ...],
    sampling Z from the height image every ~step_m meters with the SAME formula SDF uses.
    """
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
    # include the final endpoint
    w2x, w2y = world_xy_from_grid(path_grid[-1][0], path_grid[-1][1])
    wz_end   = terrain_z_from_world_xy(w2x, w2y, height_img, size_z, pos_z, bottom_aligned)
    dense.append((w2x, w2y, wz_end))
    return dense


# -----------------------------
# 6) WORLD EXPORT (SDF)
# -----------------------------
def build_world_sdf(path, height_img: Image.Image):
    """
    Compose a .world SDF with:
    - Heightmap placed either centered or bottom-aligned (toggle)
    - Start/goal markers
    - Road as many short segments following the surface closely
    """
    HEIGHT_URI = "file://" + os.path.abspath(OUTPUT_HEIGHT_IMG)
    TEX_URI    = "file://" + os.path.abspath(OUTPUT_TEXTURE)

    pos_z_for_sdf = (MAX_HEIGHT / 2.0) if HEIGHTMAP_BOTTOM_ALIGNED else 0.0

    world_header = """<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <scene><ambient>0.6 0.6 0.6 1</ambient><shadows>true</shadows></scene>
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

    # Start / Goal markers (sample Z from image using the SAME formula)
    sx, sy = path[0]
    gx, gy = path[-1]
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
          <material><ambient>0 1 1 1</ambient><diffuse>0 1 1 1</diffuse><emissive>0 0.9 0.9 1</emissive></material>
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

    # Road: dense segments closely following the surface
    dense_pts = resample_polyline_world(path, height_img, RESAMPLE_STEP_M,
                                        MAX_HEIGHT, pos_z_for_sdf, HEIGHTMAP_BOTTOM_ALIGNED)
    road_models = []
    if ADD_DEBUG_PINS:
        for k, (wx, wy, wz) in enumerate(dense_pts[::max(1, PIN_EVERY_N)]):
            road_models.append(f"""
    <model name="pin_{k}">
      <static>true</static>
      <pose>{wx} {wy} {wz + 1.0} 0 0 0</pose>
      <link name="link">
        <visual name="v">
          <geometry><cylinder><radius>0.03</radius><length>2.0</length></cylinder></geometry>
          <material><ambient>1 0 1 1</ambient><diffuse>1 0 1 1</diffuse><emissive>1 0 1 1</emissive></material>
        </visual>
      </link>
    </model>
""")

    for i in range(len(dense_pts) - 1):
        x1g, y1g, z1g = dense_pts[i]
        x2g, y2g, z2g = dense_pts[i + 1]

        mid_x = (x1g + x2g) / 2.0
        mid_y = (y1g + y2g) / 2.0
        # hover + half thickness -> clearly above the surface everywhere
        mid_z = (z1g + z2g) / 2.0 + HOVER_HEIGHT + (ROAD_THICKNESS / 2.0)

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
        <visual name="visual">
          <geometry><box><size>{length} {ROAD_WIDTH} {ROAD_THICKNESS}</size></box></geometry>
          <material>
            <ambient>0.9 0.9 0 1</ambient>
            <diffuse>1 1 0 1</diffuse>
            <emissive>1 1 0 1</emissive>
          </material>
        </visual>
      </link>
    </model>
""")

    world_footer = """
  </world>
</sdf>
"""

    return world_header + terrain_model + start_model + goal_model + "".join(road_models) + world_footer


# -----------------------------
# 7) MAIN
# -----------------------------
def main():
    print(f"Start: {START}, Goal: {GOAL}")

    map_type = input("Enter scenario type (mix/friction/hill): ").strip().lower()

    friction_map, height_map, height_img = generate_terrain(map_type)

    # 2) Plan path on image heights (meters)
    path = run_astar(friction_map, height_img)
    if not path:
        print("NO PATH FOUND.")
        return
    print(f"Path length (grid nodes): {len(path)}")

    # 3) Build SDF and write world
    sdf = build_world_sdf(path, height_img)
    with open(OUTPUT_WORLD, "w") as f:
        f.write(sdf)

    print(f"World saved to: {OUTPUT_WORLD}")
    print(f"Open with: gazebo {OUTPUT_WORLD}")


if __name__ == "__main__":
    main()