# from RGB_Damage_area import analyze_rgb_damage
from modules.RGB_Damage_area          import analyze_rgb_damage
import cv2
import os
import re

# =====================================================
# INPUT / OUTPUT FOLDERS
# =====================================================

input_folder = r"D:\PROJECTS\Solar\code\classification\imgs"
output_folder = r"output_images"

# =====================================================
# CREATE OUTPUT FOLDER & CLEAR OLD RESULTS
# =====================================================

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file)

    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Could not delete {file_path}: {e}")

print(f"Output folder cleared: {output_folder}")

# =====================================================
# ONLY PROCESS THESE CLASSES
# Everything else will be skipped
# =====================================================

ALLOWED_CLASSES = {
    "HardShading_BirdPoop",
    "Corrosion_Discoloration_Delamination",
    "GlassBreak",
        "BacksheetDamage",
    "SoftShading_Soiling",
    "SnailTrails_Microcracks",
    
}

# =====================================================
# IMAGE CLASS -> FUNCTION CLASS MAPPING
#
# Left  = class extracted from filename
# Right = class passed to analyze_rgb_damage()
# =====================================================

CLASS_MAPPING = {
    "GlassBreak": "GlassBreak",
    "BacksheetDamage": "BacksheetDamage",
    "SoftShading_Soiling": "SoftShading_Soiling",
    "SnailTrails_Microcracks": "SnailTrails_Microcracks",

    # Custom mappings
    "Corrosion_Discoloration_Delamination": "SoftShading_Soiling",
    "HardShading_BirdPoop": "GlassBreak",
    'HotSpots': 'SoftShading_Soiling',
}


# =====================================================
# EXTRACT CLASS FROM FILENAME
# =====================================================

def extract_class(filename):
    """
    Examples:
        GlassBreak.png
        GlassBreak1.png
        GlassBreak(10).png
        GlassBreak_10.png
        HardShading_BirdPoop(5).png
        Corrosion_Discoloration_Delamination(1).png
    """

    name = os.path.splitext(filename)[0]

    # Remove:
    # (10)
    # _10
    # 10
    #  10

    name = re.sub(r'[\s_]*\(?\d+\)?$', '', name)

    return name.strip()


# =====================================================
# PROCESS IMAGES
# =====================================================

total = 0
processed = 0
skipped = 0
errors = 0

print("\nStarting processing...\n")

for file_name in os.listdir(input_folder):

    if not file_name.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ):
        continue

    total += 1

    image_path = os.path.join(input_folder, file_name)

    # -----------------------------------------
    # Extract class from filename
    # -----------------------------------------

    image_class = extract_class(file_name)

    # -----------------------------------------
    # Process only selected classes
    # -----------------------------------------

    if image_class not in ALLOWED_CLASSES:
        print(
            f"SKIPPED : {file_name} "
            f"(Class '{image_class}' not selected)"
        )
        skipped += 1
        continue

    # -----------------------------------------
    # Check mapping
    # -----------------------------------------

    if image_class not in CLASS_MAPPING:
        print(
            f"SKIPPED : {file_name} "
            f"(Class '{image_class}' not mapped)"
        )
        skipped += 1
        continue

    function_class = CLASS_MAPPING[image_class]

    # -----------------------------------------
    # Run analysis
    # -----------------------------------------

    try:

        result = analyze_rgb_damage(
            image_path,
            function_class
        )

        output_name = (
    f"{os.path.splitext(file_name)[0]}_{result.damage_percentage:.2f}pct.jpg"
)

        output_path = os.path.join(
            output_folder,
            output_name
        )

        cv2.imwrite(
            output_path,
            result.annotated_image
        )

        print(
            f"OK : {file_name:<45}"
            f" InputClass={image_class:<40}"
            f" FunctionClass={function_class:<25}"
            f" Detected={result.detected}"
            f" Confidence={result.confidence:.2f}"
            f" Damage={result.damage_percentage:.2f}%"
        )

        processed += 1

    except Exception as e:

        print(f"ERROR : {file_name} -> {e}")
        errors += 1


# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 90)
print(f"Total Images : {total}")
print(f"Processed    : {processed}")
print(f"Skipped      : {skipped}")
print(f"Errors       : {errors}")
print("=" * 90)
print(f"Results saved to: {output_folder}")