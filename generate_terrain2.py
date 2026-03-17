import noise
import csv
import os
from PIL import Image

# --- 1. CONFIGURATION ---
IMG_SIZE = 129         # Gazebo requires (2^n) + 1. Do not change.
REAL_SIZE = 50.0       # The world will be 50x50 meters
MAX_HEIGHT = 4.0       # The tallest hill will be 4 meters high
NOISE_SCALE = 20.0     # How wide the mud patches and hills are
SEED = 42              # Change this to generate a completely different map

# --- 2. FILE PATHS (Absolute paths prevent Gazebo crashes) ---
CURRENT_DIR = os.getcwd()
OUTPUT_TEXTURE = os.path.join(CURRENT_DIR, "terrain_texture.png")
OUTPUT_HEIGHT_IMG = os.path.join(CURRENT_DIR, "heightmap.png")
OUTPUT_WORLD = os.path.join(CURRENT_DIR, "thesis_3d.world")

# The math files for your A* algorithm
OUTPUT_FRICTION_CSV = "friction_map.csv"
OUTPUT_HEIGHT_CSV = "height_map.csv"

# --- 3. TERRAIN RULES ---
COLOR_ASPHALT = (50, 50, 50)
COLOR_GRASS   = (34, 139, 34)
COLOR_MUD     = (101, 67, 33)

def get_terrain_data(val):
    # Returns (RGB Color, Friction Coefficient)
    if val < -0.05: return COLOR_ASPHALT, 0.013
    elif val < 0.3: return COLOR_GRASS, 0.05
    else:           return COLOR_MUD, 0.35

def generate():
    print(f"Generating 3D Thesis Environment in: {CURRENT_DIR}")

    # Create image buffers
    img_texture = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
    img_height = Image.new('L', (IMG_SIZE, IMG_SIZE)) 
    pixels_tex = img_texture.load()
    pixels_hgt = img_height.load()
    
    # Create math buffers
    friction_csv_data = []
    height_csv_data = []

    # --- 4. THE GENERATION LOOP ---
    for x in range(IMG_SIZE):
        row_friction = []
        row_height = []
        
        for y in range(IMG_SIZE):
            # A. Calculate Surface Material (Mud/Grass/Asphalt)
            val_tex = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE, octaves=6, 
                                    persistence=0.5, lacunarity=2.0, base=SEED)
            color, friction = get_terrain_data(val_tex)
            
            pixels_tex[x, y] = color
            row_friction.append(friction)

            # B. Calculate Elevation (Hills/Valleys)
            # We use SEED + 100 so the hills don't perfectly match the mud patches
            val_hgt = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE, octaves=4, 
                                    persistence=0.5, lacunarity=2.0, base=SEED + 100)
            
            # Convert noise (-1 to 1) into a physical height in meters (0m to 4m)
            normalized_0_to_1 = (val_hgt + 1.0) / 2.0
            physical_height_meters = normalized_0_to_1 * MAX_HEIGHT
            
            # Convert physical height to an image pixel (0 to 255 grayscale)
            pixel_brightness = int(normalized_0_to_1 * 255)
            pixel_brightness = max(0, min(255, pixel_brightness)) # Keep it safe
            
            pixels_hgt[x, y] = pixel_brightness
            
            # Save the physical height (e.g., 2.34 meters) for the A* math
            # Round to 3 decimal places to keep the CSV file clean
            row_height.append(round(physical_height_meters, 3))

        friction_csv_data.append(row_friction)
        height_csv_data.append(row_height)

    # --- 5. SAVE EVERYTHING ---
    
    # Rotate images so Gazebo and Python arrays align perfectly
    img_texture.transpose(Image.ROTATE_90).save(OUTPUT_TEXTURE)
    img_height.transpose(Image.ROTATE_90).save(OUTPUT_HEIGHT_IMG)
    print("Saved Images (Texture & Heightmap)")

    with open(OUTPUT_FRICTION_CSV, "w", newline='') as f:
        csv.writer(f).writerows(friction_csv_data)
    print(f"Saved A* Data: {OUTPUT_FRICTION_CSV}")

    with open(OUTPUT_HEIGHT_CSV, "w", newline='') as f:
        csv.writer(f).writerows(height_csv_data)
    print(f"Saved A* Data: {OUTPUT_HEIGHT_CSV}")

    # Generate the Gazebo World File
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
          <surface>
            <friction><ode><mu>100</mu><mu2>50</mu2></ode></friction>
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
  </world>
</sdf>
"""
    with open(OUTPUT_WORLD, "w") as f:
        f.write(sdf_content)

if __name__ == "__main__":
    generate()
