import noise
import csv
import os
import math
from PIL import Image

# --- 1. CONFIGURATION ---
IMG_SIZE = 129
REAL_SIZE = 50.0
MAX_HEIGHT = 4.0
NOISE_SCALE = 20.0
SEED = 42

CURRENT_DIR = os.getcwd()
OUTPUT_TEXTURE = os.path.join(CURRENT_DIR, "terrain_texture.png")
OUTPUT_HEIGHT_IMG = os.path.join(CURRENT_DIR, "heightmap.png")
OUTPUT_WORLD = os.path.join(CURRENT_DIR, "thesis_3d.world")
OUTPUT_FRICTION_CSV = "friction_map.csv"
OUTPUT_HEIGHT_CSV = "height_map.csv"

COLOR_ASPHALT = (50, 50, 50)
COLOR_GRASS   = (34, 139, 34)
COLOR_MUD     = (101, 67, 33)

def get_terrain_data(val):
    if val < -0.05: return COLOR_ASPHALT, 0.013
    elif val < 0.3: return COLOR_GRASS, 0.05
    else:           return COLOR_MUD, 0.35

def generate():
    img_texture = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
    img_height = Image.new('L', (IMG_SIZE, IMG_SIZE)) 
    pixels_tex = img_texture.load()
    pixels_hgt = img_height.load()
    
    friction_csv_data = []
    height_csv_data = []

    for x in range(IMG_SIZE):
        row_friction = []
        row_height = []
        for y in range(IMG_SIZE):
            val_tex = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE, octaves=6, persistence=0.5, lacunarity=2.0, base=SEED)
            color, friction = get_terrain_data(val_tex)
            pixels_tex[x, y] = color
            row_friction.append(friction)

            val_hgt = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE, octaves=4, persistence=0.5, lacunarity=2.0, base=SEED + 100)
            normalized_0_to_1 = (val_hgt + 1.0) / 2.0
            physical_height_meters = normalized_0_to_1 * MAX_HEIGHT
            
            # FIX 1: Inverse Gamma Correction to counter Gazebo's 8-bit rendering curve
            gamma_corrected = math.pow(normalized_0_to_1, 1.0 / 2.2)
            pixel_brightness = int(gamma_corrected * 255)
            pixels_hgt[x, y] = max(0, min(255, pixel_brightness))
            
            row_height.append(round(physical_height_meters, 3))

        friction_csv_data.append(row_friction)
        height_csv_data.append(row_height)

    # FIX 2: Save raw images without transpose rotations to perfectly align with CSV matrices
    img_texture.save(OUTPUT_TEXTURE)
    img_height.save(OUTPUT_HEIGHT_IMG)

    with open(OUTPUT_FRICTION_CSV, "w", newline='') as f: csv.writer(f).writerows(friction_csv_data)
    with open(OUTPUT_HEIGHT_CSV, "w", newline='') as f: csv.writer(f).writerows(height_csv_data)

    sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <scene><ambient>0.6 0.6 0.6 1</ambient><shadows>true</shadows></scene>
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
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <use_terrain_paging>false</use_terrain_paging>
              <texture><diffuse>file://{OUTPUT_TEXTURE}</diffuse><normal>file://media/materials/textures/flat_normal.png</normal><size>{REAL_SIZE}</size></texture>
              <uri>file://{OUTPUT_HEIGHT_IMG}</uri>
              <size>{REAL_SIZE} {REAL_SIZE} {MAX_HEIGHT}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>"""
    with open(OUTPUT_WORLD, "w") as f: f.write(sdf_content)

if __name__ == "__main__":
    generate()