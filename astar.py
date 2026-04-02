# -*- coding: utf-8 -*-
import os
import csv
import math
import random
import time
from PIL import Image
try:
    from generate_terrain import generate_terrain  
except ImportError:
    print("CRITICAL ERROR: Could not find 'generate_terrain.py'.")
    print("Ensure both files are in the same folder.")
    exit(1)

class Config:
    def __init__(self):
        self.IMG_SIZE = 129
        self.GRID_SIZE = self.IMG_SIZE
        self.REAL_SIZE = 50.0
        self.MAX_HEIGHT = 4.0
        self.ROBOT_MASS = 5.0  # kg 
        self.GRAVITY = 9.81     # m/s^2

        self.HEIGHTMAP_BOTTOM_ALIGNED = False
        self.REMOVE_GROUND_PLANE = True
        self.HOVER_HEIGHT = 0.05
        self.ROAD_THICKNESS = 0.12
        self.ROAD_WIDTH = 0.30
        self.RESAMPLE_STEP_M = 0.30
        self.ADD_DEBUG_PINS = False
        self.PIN_EVERY_N = 8
        self.START = (random.randrange(5, 90), random.randrange(5, 90))
        self.GOAL = (115, 115)
        self.MIN_FRICTION = 0.008 
        self.MAX_SLOPE = 0.5
        self.CURRENT_DIR = os.getcwd()
        self.OUTPUT_TEXTURE = os.path.join(self.CURRENT_DIR, "terrain_texture.png")
        self.OUTPUT_HEIGHT_IMG = os.path.join(self.CURRENT_DIR, "heightmap.png")
        self.OUTPUT_WORLD = os.path.join(self.CURRENT_DIR, "thesis_3d_path.world")

class Terrain:
    def __init__(self, config, height_img, friction_map):
        self.config = config
        self.height_img = height_img
        self.friction_map = friction_map

    def bilinear_sample_u8(self, img: Image.Image, u: float, v: float) -> float:
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

    def csv_index_to_uv(self, grid_x: int, grid_y: int) -> tuple:
        u = grid_x / (self.config.GRID_SIZE - 1)
        v = grid_y / (self.config.GRID_SIZE - 1)
        return u, v

    def world_xy_from_grid(self, grid_x: int, grid_y: int) -> tuple:
        u, v = self.csv_index_to_uv(grid_x, grid_y)
        world_x = (u - 0.5) * self.config.REAL_SIZE
        world_y = (0.5 - v) * self.config.REAL_SIZE
        return world_x, world_y

    def world_to_uv(self, wx: float, wy: float, pos_x=0.0, pos_y=0.0) -> tuple:
        u = (wx - pos_x) / self.config.REAL_SIZE + 0.5
        v = 0.5 - (wy - pos_y) / self.config.REAL_SIZE
        return u, v

    def world_xy_to_grid(self, wx: float, wy: float) -> tuple:
        u, v = self.world_to_uv(wx, wy, pos_x=0.0, pos_y=0.0)
        gx = int(max(0, min(self.config.GRID_SIZE - 1, round(u * (self.config.GRID_SIZE - 1)))))
        gy = int(max(0, min(self.config.GRID_SIZE - 1, round(v * (self.config.GRID_SIZE - 1)))))
        return gx, gy

    def terrain_z_from_world_xy(self, wx: float, wy: float) -> float:
        pos_z = (self.config.MAX_HEIGHT / 2.0) if self.config.HEIGHTMAP_BOTTOM_ALIGNED else 0.0
        u, v = self.world_to_uv(wx, wy, pos_x=0.0, pos_y=0.0)
        p = self.bilinear_sample_u8(self.height_img, u, v)
        return pos_z + (p * self.config.MAX_HEIGHT)

    def get_height_m_from_img(self, grid_x: int, grid_y: int) -> float:
        u, v = self.csv_index_to_uv(grid_x, grid_y)
        p = self.bilinear_sample_u8(self.height_img, u, v)
        return p * self.config.MAX_HEIGHT

    def get_3d_step_cost(self, curr, nxt):
        cx, cy = curr
        nx, ny = nxt
        dist_2d_grid = math.hypot(nx - cx, ny - cy)
        if dist_2d_grid == 0:
            return float('inf')
        # Convert grid distance to real-world meters
        meters_per_cell = self.config.REAL_SIZE / (self.config.GRID_SIZE - 1)
        dist_m = dist_2d_grid * meters_per_cell

        friction = self.friction_map[nx][ny]
        h_curr = self.get_height_m_from_img(cx, cy)
        h_next = self.get_height_m_from_img(nx, ny)
        dh = h_next - h_curr
        
        # Check physical slope constraint
        slope = abs(dh) / dist_m
        if slope > self.config.MAX_SLOPE:
            return float('inf')
        weight = self.config.ROBOT_MASS * self.config.GRAVITY
        
        # Work done against friction
        energy_friction = friction * weight * dist_m
        energy_gravity = (weight * dh) if dh > 0 else 0.0
        
        return energy_friction + energy_gravity

class AStarPlanner:
    def __init__(self, config, terrain):
        self.config = config
        self.terrain = terrain

    def heuristic(self, curr, goal, start):
        cx, cy = curr
        gx, gy = goal
        sx, sy = start
        
        # Convert 2D grid distance to meters
        meters_per_cell = self.config.REAL_SIZE / (self.config.GRID_SIZE - 1)
        dist_2d_grid = math.hypot(gx - cx, gy - cy)
        dist_m = dist_2d_grid * meters_per_cell
        
        cz = self.terrain.get_height_m_from_img(cx, cy)
        gz = self.terrain.get_height_m_from_img(gx, gy)
        dh = gz - cz
        
        weight = self.config.ROBOT_MASS * self.config.GRAVITY
        
        # Admissible energy components
        min_e_friction = self.config.MIN_FRICTION * weight * dist_m
        min_e_gravity = (weight * dh) if dh > 0 else 0.0
        
        h_joules = min_e_friction + min_e_gravity
        
        #tie-breaker to prevent searching symmetrical paths
        dx1 = cx - gx
        dy1 = cy - gy
        dx2 = sx - gx
        dy2 = sy - gy
        cross_product = abs(dx1 * dy2 - dx2 * dy1)
        
        return h_joules + (cross_product * 0.001)

    def run_astar(self):
        import heapq
        open_list = []
        counter = 0
        heapq.heappush(open_list, (0, counter, self.config.START))

        came_from = {}
        g_score = {self.config.START: 0.0}
        f_score = {self.config.START: self.heuristic(self.config.START, self.config.GOAL, self.config.START)}
        closed = set()

        while open_list:
            current = heapq.heappop(open_list)[2]
            if current == self.config.GOAL:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.config.START)
                path.reverse()
                return path

            closed.add(current)
            cx, cy = current
            neighbors = [
                (cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy),
                (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)
            ]
            for nx, ny in neighbors:
                if not (0 <= nx < self.config.GRID_SIZE and 0 <= ny < self.config.GRID_SIZE):
                    continue
                if (nx, ny) in closed:
                    continue
                step = self.terrain.get_3d_step_cost(current, (nx, ny))
                if step == float('inf'):
                    continue
                tentative_g = g_score[current] + step
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score[(nx, ny)] = tentative_g + self.heuristic((nx, ny), self.config.GOAL, self.config.START)
                    counter += 1
                    heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))

        return []  # no path

class PathProcessor:
    def __init__(self, config, terrain):
        self.config = config
        self.terrain = terrain

    def get_bresenham_line(self, x0, y0, x1, y1):
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

    def get_dense_grid_cost(self, path_sparse):
        total_cost = 0.0
        is_valid = True
        for i in range(len(path_sparse) - 1):
            line = self.get_bresenham_line(path_sparse[i][0], path_sparse[i][1], path_sparse[i+1][0], path_sparse[i+1][1])
            for j in range(len(line) - 1):
                cost = self.terrain.get_3d_step_cost(line[j], line[j+1])
                if cost == float('inf'):
                    is_valid = False
                total_cost += cost
        return total_cost, is_valid

    def smooth_path_los(self, path):
        if len(path) <= 2:
            return path
            
        print("Smoothing path (Cost-Aware String-Pulling)...")
        smoothed_path = [path[0]]
        current_index = 0
        
        while current_index < len(path) - 1:
            furthest_visible = current_index + 1
            
            for j in range(current_index + 2, len(path)):
                shortcut_line = [path[current_index], path[j]]
                straight_cost, valid = self.get_dense_grid_cost(shortcut_line)
                
                astar_segment = path[current_index:j+1]
                astar_cost, _ = self.get_dense_grid_cost(astar_segment)
                
                if valid and straight_cost <= (astar_cost * 1.02):
                    furthest_visible = j
                else:
                    break
                    
            smoothed_path.append(path[furthest_visible])
            current_index = furthest_visible
            
        print(f"Original nodes: {len(path)} -> Smoothed nodes: {len(smoothed_path)}")
        return smoothed_path

    def optimize_path_rubberband(self, path, max_iters=30):
        if len(path) <= 2:
            return path
            
        print("Optimizing path (Rubberband Gradient Descent)...")
        current_path = path.copy()
        optimized = True
        iterations = 0
        
        while optimized and iterations < max_iters:
            optimized = False
            iterations += 1
            
            for i in range(1, len(current_path) - 1):
                prev_node = current_path[i-1]
                curr_node = current_path[i]
                next_node = current_path[i+1]
                
                cost_curr1, v1 = self.get_dense_grid_cost([prev_node, curr_node])
                cost_curr2, v2 = self.get_dense_grid_cost([curr_node, next_node])
                best_cost = cost_curr1 + cost_curr2
                best_node = curr_node
                
                cx, cy = curr_node
                neighbors = [
                    (cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy),
                    (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)
                ]
                
                for nx, ny in neighbors:
                    if not (0 <= nx < self.config.GRID_SIZE and 0 <= ny < self.config.GRID_SIZE): 
                        continue
                        
                    c1, val1 = self.get_dense_grid_cost([prev_node, (nx, ny)])
                    c2, val2 = self.get_dense_grid_cost([(nx, ny), next_node])
                    
                    if val1 and val2 and (c1 + c2) < best_cost - 0.0001:
                        best_cost = c1 + c2
                        best_node = (nx, ny)
                        
                if best_node != curr_node:
                    current_path[i] = best_node
                    optimized = True
                    
        print(f"Rubberband Optimization finished in {iterations} iterations.")
        return current_path

class MetricsCalculator:
    def __init__(self, config, terrain):
        self.config = config
        self.terrain = terrain

    def calculate_path_metrics(self, path, timing_data):
        print("\n--- PATH METRICS ---")
        print(f"A* Search Time:           {timing_data['astar']:.4f} seconds")
        print(f"LOS Smoothing Time:       {timing_data['smooth']:.4f} seconds")
        print(f"Rubberband Opt. Time:     {timing_data['opt']:.4f} seconds")
        print(f"Total Planning Time:      {timing_data['total']:.4f} seconds")
        
        total_dist_2d = 0.0
        total_dist_3d = 0.0
        min_z = float('inf')
        max_z = float('-inf')
        max_step_dz = 0.0
        pos_z_for_sdf = (self.config.MAX_HEIGHT / 2.0) if self.config.HEIGHTMAP_BOTTOM_ALIGNED else 0.0
        
        for i in range(len(path)):
            cx, cy = path[i]
            wx, wy = self.terrain.world_xy_from_grid(cx, cy)
            cz = self.terrain.terrain_z_from_world_xy(wx, wy)
            
            if cz < min_z: min_z = cz
            if cz > max_z: max_z = cz
            
            if i > 0:
                px, py = path[i-1]
                pwx, pwy = self.terrain.world_xy_from_grid(px, py)
                pz = self.terrain.terrain_z_from_world_xy(pwx, pwy)
                
                dist2d = math.hypot(wx - pwx, wy - pwy)
                total_dist_2d += dist2d
                
                dz = cz - pz
                total_dist_3d += math.hypot(dist2d, dz)
                
                if abs(dz) > max_step_dz:
                    max_step_dz = abs(dz)
                    
        overall_elev_change = max_z - min_z
        
        processor = PathProcessor(self.config, self.terrain)
        true_cost, is_valid = processor.get_dense_grid_cost(path)
        
        print(f"Total Path Distance (2D): {total_dist_2d:.2f} meters")
        print(f"Total Path Distance (3D): {total_dist_3d:.2f} meters")
        print(f"Total Energy Consumed:    {true_cost:.2f} Joules")
        print(f"Physically Valid (Slope): {'YES' if is_valid else 'NO (Exceeded MAX_SLOPE)'}")
        print(f"Minimum Elevation:        {min_z:.2f} meters")
        print(f"Maximum Elevation:        {max_z:.2f} meters")
        print(f"Overall Elevation Span:   {overall_elev_change:.2f} meters")
        print(f"Steepest Single Step:     {max_step_dz:.2f} meters")
        print("--------------------\n")

    def verify_path_costs(self, final_path, straight_path_grid):
        print("\n--- PATH VERIFICATION ---")
        
        processor = PathProcessor(self.config, self.terrain)
        astar_cost, _ = processor.get_dense_grid_cost(final_path)
        print(f"Optimized Path Cost:    {astar_cost:.4f}")

        straight_cost, is_valid = processor.get_dense_grid_cost([straight_path_grid[0], straight_path_grid[-1]])
            
        if not is_valid:
            print("Straight Line Cost:     INVALID (Exceeds MAX_SLOPE)")
            print("Conclusion: Path had to bend because a straight line hit an unscalable cliff.")
        else:
            print(f"Straight Line Cost:     {straight_cost:.4f}")
            diff = straight_cost - astar_cost
            print(f"Difference:             {diff:.4f}")
            if diff > 0:
                print("Conclusion: Algorithm successfully routed around a hill or high-friction area!")
        print("-------------------------\n")

class WorldBuilder:
    def __init__(self, config, terrain):
        self.config = config
        self.terrain = terrain

    def resample_polyline_world(self, path_grid, step_m):
        dense = []
        for i in range(len(path_grid) - 1):
            x1, y1 = path_grid[i]
            x2, y2 = path_grid[i + 1]
            w1x, w1y = self.terrain.world_xy_from_grid(x1, y1)
            w2x, w2y = self.terrain.world_xy_from_grid(x2, y2)
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
                wz = self.terrain.terrain_z_from_world_xy(wx, wy)
                dense.append((wx, wy, wz))
        w2x, w2y = self.terrain.world_xy_from_grid(path_grid[-1][0], path_grid[-1][1])
        wz_end = self.terrain.terrain_z_from_world_xy(w2x, w2y)
        dense.append((w2x, w2y, wz_end))
        return dense

    def build_world_sdf(self, final_path, straight_path_grid, normal_path, dijkstra_path):
        HEIGHT_URI = "file://" + os.path.abspath(self.config.OUTPUT_HEIGHT_IMG)
        TEX_URI    = "file://" + os.path.abspath(self.config.OUTPUT_TEXTURE)

        pos_z_for_sdf = (self.config.MAX_HEIGHT / 2.0) if self.config.HEIGHTMAP_BOTTOM_ALIGNED else 0.0

        world_header = """<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <scene><ambient>0.6 0.6 0.6 1</ambient><shadows>true</shadows><grid>false</grid></scene>
"""
        if not self.config.REMOVE_GROUND_PLANE:
            world_header += '    <include><uri>model://ground_plane</uri></include>\n'

        terrain_model = f"""
    <model name="thesis_terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>{HEIGHT_URI}</uri>
              <size>{self.config.REAL_SIZE} {self.config.REAL_SIZE} {self.config.MAX_HEIGHT}</size>
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
                <size>{self.config.REAL_SIZE}</size>
              </texture>
              <uri>{HEIGHT_URI}</uri>
              <size>{self.config.REAL_SIZE} {self.config.REAL_SIZE} {self.config.MAX_HEIGHT}</size>
              <pos>0 0 {pos_z_for_sdf}</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
"""

        sx, sy = final_path[0]
        gx, gy = final_path[-1]
        sxw, syw = self.terrain.world_xy_from_grid(sx, sy)
        gxw, gyw = self.terrain.world_xy_from_grid(gx, gy)

        sz = self.terrain.terrain_z_from_world_xy(sxw, syw)
        gz = self.terrain.terrain_z_from_world_xy(gxw, gyw)

        start_model = f"""
    <model name="start_marker">
      <static>true</static>
      <pose>{sxw} {syw} {sz + self.config.HOVER_HEIGHT + 0.25} 0 0 0</pose>
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
      <pose>{gxw} {gyw} {gz + self.config.HOVER_HEIGHT + 0.25} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.4</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse><emissive>1 0 0 1</emissive></material>
        </visual>
      </link>
    </model>
"""

        dense_pts = self.resample_polyline_world(final_path, self.config.RESAMPLE_STEP_M)
        road_models = []

        for i in range(len(dense_pts) - 1):
            x1g, y1g, z1g = dense_pts[i]
            x2g, y2g, z2g = dense_pts[i + 1]

            mid_x = (x1g + x2g) / 2.0
            mid_y = (y1g + y2g) / 2.0
            mid_z = (z1g + z2g) / 2.0 + self.config.HOVER_HEIGHT + (self.config.ROAD_THICKNESS / 2.0)

            gx_grid, gy_grid = self.terrain.world_xy_to_grid(mid_x, mid_y)
            mu = self.terrain.friction_map[gx_grid][gy_grid]
            
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
          <geometry><box><size>{length} {self.config.ROAD_WIDTH} {self.config.ROAD_THICKNESS}</size></box></geometry>
          <surface>
            <friction>
              <ode><mu>{mu}</mu><mu2>{mu}</mu2></ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{length} {self.config.ROAD_WIDTH} {self.config.ROAD_THICKNESS}</size></box></geometry>
          <material>
            <ambient>1 1 0 1</ambient> <diffuse>1 1 0 1</diffuse>
            <emissive>1 1 0 1</emissive>
          </material>
        </visual>
      </link>
    </model>
""")

        dense_straight_pts = self.resample_polyline_world(straight_path_grid, 0.5)
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

        normal_models = []
        if normal_path:
            dense_normal_pts = self.resample_polyline_world(normal_path, self.config.RESAMPLE_STEP_M)
            for i in range(len(dense_normal_pts) - 1):
                x1g, y1g, z1g = dense_normal_pts[i]
                x2g, y2g, z2g = dense_normal_pts[i + 1]

                mid_x = (x1g + x2g) / 2.0
                mid_y = (y1g + y2g) / 2.0
                mid_z = (z1g + z2g) / 2.0 + self.config.HOVER_HEIGHT + 0.1 + (self.config.ROAD_THICKNESS / 4.0)
                
                dx = x2g - x1g
                dy = y2g - y1g
                dz = z2g - z1g

                dist_2d = math.hypot(dx, dy)
                length = math.hypot(dist_2d, dz)
                yaw   = math.atan2(dy, dx)
                pitch = -math.atan2(dz, dist_2d) if dist_2d > 1e-6 else 0.0

                normal_models.append(f"""
        <model name="normal_astar_segment_{i}">
          <static>true</static>
          <pose>{mid_x} {mid_y} {mid_z} 0 {pitch} {yaw}</pose>
          <link name="link">
            <visual name="visual">
              <geometry><box><size>{length} {self.config.ROAD_WIDTH * 0.5} {self.config.ROAD_THICKNESS * 0.5}</size></box></geometry>
              <material>
                <ambient>0 1 1 1</ambient> <diffuse>0 1 1 1</diffuse>
                <emissive>0 1 1 1</emissive>
              </material>
            </visual>
          </link>
        </model>
    """)

        dijkstra_models = []
        if dijkstra_path:
            dense_dijkstra_pts = self.resample_polyline_world(dijkstra_path, 0.5)
            for j, (wx, wy, wz) in enumerate(dense_dijkstra_pts):
                dijkstra_models.append(f"""
        <model name="dijkstra_marker_{j}">
          <static>true</static>
          <pose>{wx} {wy} {wz + 0.15} 0 0 0</pose>
          <link name="link">
            <visual name="visual">
              <geometry><sphere><radius>0.08</radius></sphere></geometry>
              <material>
                <ambient>0 0 1 1</ambient> <diffuse>0 0 1 1</diffuse>
                <emissive>0 0 1 1</emissive>
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
                "".join(road_models) + "".join(straight_line_models) + 
                "".join(normal_models) + "".join(dijkstra_models) + world_footer)

class StandardAStarPlanner:
    def __init__(self, config, terrain):
        self.config = config
        self.terrain = terrain

    def standard_cost(self, curr, nxt):
        cx, cy = curr; nx, ny = nxt
        dist_2d = math.hypot(nx - cx, ny - cy)
        if dist_2d == 0: return float('inf')
        h_curr = self.terrain.get_height_m_from_img(cx, cy)
        h_next = self.terrain.get_height_m_from_img(nx, ny)
        dh = h_next - h_curr
        if abs(dh) / dist_2d > self.config.MAX_SLOPE: return float('inf')
        return math.sqrt(dist_2d**2 + dh**2)

    def standard_heuristic_func(self, curr, goal):
        cx, cy = curr; gx, gy = goal
        cz = self.terrain.get_height_m_from_img(cx, cy)
        gz = self.terrain.get_height_m_from_img(gx, gy)
        return math.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2)

    def run_standard_astar(self):
        import heapq
        open_list = []
        counter = 0
        heapq.heappush(open_list, (0, counter, self.config.START))
        came_from = {}
        g_score = {self.config.START: 0.0}
        f_score = {self.config.START: self.standard_heuristic_func(self.config.START, self.config.GOAL)}
        closed = set()

        while open_list:
            current = heapq.heappop(open_list)[2]
            if current == self.config.GOAL:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.config.START)
                path.reverse()
                return path

            closed.add(current)
            cx, cy = current
            neighbors = [(cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy), (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)]
            for nx, ny in neighbors:
                if not (0 <= nx < self.config.GRID_SIZE and 0 <= ny < self.config.GRID_SIZE): continue
                if (nx, ny) in closed: continue
                step = self.standard_cost(current, (nx, ny))
                if step == float('inf'): continue
                tentative_g = g_score[current] + step
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score[(nx, ny)] = tentative_g + self.standard_heuristic_func((nx, ny), self.config.GOAL)
                    counter += 1
                    heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))
        return []

class DijkstraPlanner:
    def __init__(self, config, terrain):
        self.config = config
        self.terrain = terrain

    def standard_cost(self, curr, nxt):
        cx, cy = curr; nx, ny = nxt
        dist_2d = math.hypot(nx - cx, ny - cy)
        if dist_2d == 0: return float('inf')
        h_curr = self.terrain.get_height_m_from_img(cx, cy)
        h_next = self.terrain.get_height_m_from_img(nx, ny)
        dh = h_next - h_curr
        if abs(dh) / dist_2d > self.config.MAX_SLOPE: return float('inf')
        return math.sqrt(dist_2d**2 + dh**2)

    def run_dijkstra(self):
        import heapq
        open_list = []
        counter = 0
        heapq.heappush(open_list, (0, counter, self.config.START))
        came_from = {}
        g_score = {self.config.START: 0.0}
        f_score = {self.config.START: 0.0} 
        closed = set()

        while open_list:
            current = heapq.heappop(open_list)[2]
            if current == self.config.GOAL:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.config.START)
                path.reverse()
                return path

            closed.add(current)
            cx, cy = current
            neighbors = [(cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy), (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)]
            for nx, ny in neighbors:
                if not (0 <= nx < self.config.GRID_SIZE and 0 <= ny < self.config.GRID_SIZE): continue
                if (nx, ny) in closed: continue
                step = self.standard_cost(current, (nx, ny))
                if step == float('inf'): continue
                tentative_g = g_score[current] + step
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score[(nx, ny)] = tentative_g 
                    counter += 1
                    heapq.heappush(open_list, (f_score[(nx, ny)], counter, (nx, ny)))
        return []
# -----------------------------

def main():
    config = Config()
    print(f"Truly Randomized Start: {config.START}, Goal: {config.GOAL}")

    map_type = input("Enter scenario type (mix/friction/hill): ").strip().lower()

    # 1) Generate terrain maps using separate module
    friction_map, height_map, height_img = generate_terrain(map_type)
    
    terrain = Terrain(config, height_img, friction_map)
    planner = AStarPlanner(config, terrain)
    processor = PathProcessor(config, terrain)
    metrics = MetricsCalculator(config, terrain)
    builder = WorldBuilder(config, terrain)
    
    # 2) Plan RAW A* path on image heights (meters)
    t0 = time.time()  # Start timer
    raw_path = planner.run_astar()
    t1 = time.time()  # A* finished
    
    if not raw_path:
        print("NO PATH FOUND.")
        return
    print(f"Path length (raw grid nodes): {len(raw_path)}")

    # 3) SMOOTH THE PATH (Cost-Aware String-Pulling)
    smoothed_path = processor.smooth_path_los(raw_path)
    t2 = time.time()  # Smoothing finished
    
    # NEW: Pull the string tight against the obstacles!
    final_path = processor.optimize_path_rubberband(smoothed_path)
    t3 = time.time()  # Rubberband finished

    # 4) Generate mathematically direct line grid coordinates early for use in Verify and Build
    straight_line_grid = processor.get_bresenham_line(final_path[0][0], final_path[0][1], final_path[-1][0], final_path[-1][1])

    # 5) VERIFY the Final path cost vs a straight line in terminal output
    metrics.verify_path_costs(final_path, straight_line_grid)

    # ---> NEW: Calculate and Display Metrics <---
    timing_data = {
        'astar': t1 - t0,
        'smooth': t2 - t1,
        'opt': t3 - t2,
        'total': t3 - t0
    }
    print("\n==================================")
    print("   CUSTOM A* METRICS")
    print("==================================")
    metrics.calculate_path_metrics(final_path, timing_data)

    print("\nCalculating Standard A* path...")
    standard_planner = StandardAStarPlanner(config, terrain)
    t_n0 = time.time()
    raw_normal_path = standard_planner.run_standard_astar()
    t_n1 = time.time()
    
    # Smooth the standard path!
    smooth_normal_path = processor.smooth_path_los(raw_normal_path)
    normal_path = processor.optimize_path_rubberband(smooth_normal_path)
    t_n2 = time.time()

    print("\n==================================")
    print("   STANDARD A* METRICS")
    print("==================================")
    metrics.calculate_path_metrics(normal_path, {
        'astar': t_n1 - t_n0, 'smooth': t_n2 - t_n1, 'opt': 0.0, 'total': t_n2 - t_n0
    })
    

    print("\nCalculating Dijkstra path...")
    dijkstra_planner = DijkstraPlanner(config, terrain)
    
    t_d0 = time.time()
    raw_dijkstra_path = dijkstra_planner.run_dijkstra()
    t_d1 = time.time()
    
    # Smooth and optimize the Dijkstra path!
    smooth_dijkstra_path = processor.smooth_path_los(raw_dijkstra_path)
    t_d2 = time.time()
    
    dijkstra_path = processor.optimize_path_rubberband(smooth_dijkstra_path)
    t_d3 = time.time()

    print("\n==================================")
    print("   DIJKSTRA METRICS")
    print("==================================")
    metrics.calculate_path_metrics(dijkstra_path, {
        'astar': t_d1 - t_d0, 
        'smooth': t_d2 - t_d1, 
        'opt': t_d3 - t_d2, 
        'total': t_d3 - t_d0
    })

    sdf = builder.build_world_sdf(final_path, straight_line_grid, normal_path, dijkstra_path)
    with open(config.OUTPUT_WORLD, "w") as f:
        f.write(sdf)

    print(f"World saved to: {config.OUTPUT_WORLD}")
    print(f"Open with: gazebo {config.OUTPUT_WORLD}")

if __name__ == "__main__":
    main()