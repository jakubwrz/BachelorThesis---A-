import csv
from PIL import Image
import noise

# Configuration constants
IMG_SIZE = 129  # Adjust as needed
NOISE_SCALE = 100  # Adjust as needed
SEED = 12  # Adjust as needed
MAX_HEIGHT = 1.0  # Adjust as needed
OUTPUT_TEXTURE = "terrain_texture.png"
OUTPUT_HEIGHT_IMG = "heightmap.png"
OUTPUT_FRICTION_CSV = "friction_map.csv"
OUTPUT_HEIGHT_CSV = "height_map.csv"


def get_terrain_data(noise_value, terrain_type="mix"):
    """
    Determine terrain color and friction based on noise value and terrain type.
    
    Args:
        noise_value: Perlin noise value
        terrain_type: Type of terrain ("friction", "hill", "mix")
        
    Returns:
        Tuple of (color_rgb, friction_coefficient)
    """
    if terrain_type == "mix":
        if noise_value < -0.05:
            return (50, 50, 50), 0.013      # Asphalt (Dark gray, low friction)
        elif noise_value < 0.3:
            return (34, 139, 34), 0.05      # Grass (Green, medium friction)
        else:
            return (101, 67, 33), 0.35      # Mud (Brown, high friction)
            
    elif terrain_type == "friction":
        # Handled dynamically in the main loop based on path
        return (128, 128, 128), 0.5 
        
    elif terrain_type == "hill":
        # Handled dynamically in the main loop based on distance
        return (128, 128, 128), 0.5


def generate_terrain(terrain_type="mix"):
    """
    Generate terrain maps based on the specified type.
    
    Args:
        terrain_type: Type of terrain to generate ("friction", "hill", "mix")
        
    Returns:
        Tuple of (friction_map, height_map_m, img_height)
    """
    if terrain_type not in ["friction", "hill", "mix"]:
        raise ValueError("terrain_type must be one of: 'friction', 'hill', 'mix'")
    
    img_texture = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
    img_height  = Image.new('L',   (IMG_SIZE, IMG_SIZE))
    p_tex = img_texture.load()
    p_hgt = img_height.load()

    friction_map = []
    height_map_m = [] 

    if terrain_type == "mix":
        # Original mixed terrain logic
        # 1st Pass: Generate noise and find the absolute min and max
        raw_noise = []
        min_h = float('inf')
        max_h = float('-inf')

        for x in range(IMG_SIZE):
            row_raw = []
            for y in range(IMG_SIZE):
                val_hgt = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE,
                                        octaves=4, persistence=0.5, lacunarity=2.0, base=SEED+100)
                row_raw.append(val_hgt)
                if val_hgt < min_h: min_h = val_hgt
                if val_hgt > max_h: max_h = val_hgt
            raw_noise.append(row_raw)

        # 2nd Pass: Normalize and generate
        for x in range(IMG_SIZE):
            row_f = []
            row_h = []
            for y in range(IMG_SIZE):
                # Texture & friction
                val_tex = noise.pnoise2(x/NOISE_SCALE, y/NOISE_SCALE,
                                        octaves=6, persistence=0.5, lacunarity=2.0, base=SEED)
                color, friction = get_terrain_data(val_tex, terrain_type)
                p_tex[x, y] = color
                row_f.append(friction)

                # Strict normalization maps the lowest point to 0 and highest to 1
                if max_h > min_h:
                    normalized_0_1 = (raw_noise[x][y] - min_h) / (max_h - min_h)
                else:
                    normalized_0_1 = 0.0
                    
                physical_h_m = normalized_0_1 * MAX_HEIGHT 

                px = int(max(0, min(255, round(normalized_0_1 * 255))))
                p_hgt[x, y] = px
                row_h.append(physical_h_m)
                
            friction_map.append(row_f)
            height_map_m.append(row_h)
            
    elif terrain_type == "friction":
        # Flat terrain with friction path in the middle
        path_width = IMG_SIZE // 8  # Path width
        high_friction = 0.9
        low_friction = 0.1
        
        for x in range(IMG_SIZE):
            row_f = []
            row_h = []
            for y in range(IMG_SIZE):
                # Check if in the middle path (vertical strip)
                in_path = abs(y - IMG_SIZE // 2) < path_width
                friction = high_friction if in_path else low_friction
                
                # Color based on friction
                if in_path:
                    color = (50, 50, 50)       # Dark gray (Asphalt)
                else:
                    color = (176, 224, 230)    # Light blue (Ice)
                    
                p_tex[x, y] = color
                row_f.append(friction)
                
                # Flat height
                physical_h_m = 0.0
                px = 0
                p_hgt[x, y] = px
                row_h.append(physical_h_m)
                
            friction_map.append(row_f)
            height_map_m.append(row_h)
            
        # FIX: Add a tiny 1-pixel bump in the corner so Gazebo doesn't divide by zero!
        p_hgt[0, 0] = 1 
        
    elif terrain_type == "hill":
        # Flat friction with hill in the middle
        hill_radius = IMG_SIZE // 4
        hill_height = MAX_HEIGHT
        
        for x in range(IMG_SIZE):
            row_f = []
            row_h = []
            for y in range(IMG_SIZE):
                # Constant friction
                friction = 0.5
                
                # Hill in the center
                center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                if distance <= hill_radius:
                    color = (101, 67, 33)   # Brown (Rock/Dirt) for the hill
                    height_factor = 1 - (distance / hill_radius) ** 2
                    physical_h_m = height_factor * hill_height
                else:
                    color = (34, 139, 34)   # Green (Grass) for the flat ground
                    physical_h_m = 0.0
                    
                p_tex[x, y] = color
                row_f.append(friction)
                
                px = int(max(0, min(255, round(physical_h_m / MAX_HEIGHT * 255))))
                p_hgt[x, y] = px
                row_h.append(physical_h_m)
                
            friction_map.append(row_f)
            height_map_m.append(row_h)

    img_texture.save(OUTPUT_TEXTURE)
    img_height.save(OUTPUT_HEIGHT_IMG)

    with open(OUTPUT_FRICTION_CSV, "w", newline='') as f:
        csv.writer(f).writerows(friction_map)
    with open(OUTPUT_HEIGHT_CSV, "w", newline='') as f:
        csv.writer(f).writerows(height_map_m)

    return friction_map, height_map_m, img_height


if __name__ == "__main__":
    # Run terrain generation when script is executed directly
    import sys
    terrain_type = sys.argv[1] if len(sys.argv) > 1 else "mix"
    friction_map, height_map, height_image = generate_terrain(terrain_type)
    print(f"Terrain generation complete! Type: {terrain_type}")
