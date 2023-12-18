from PIL import Image

import cv2
import glob
import os
import shutil
import re
import os
import pandas as pd

# data root
LANDCOVER_AI_ROOT = '../data/landcover.ai.v1'
DEEPGLOBE_ROOT = '../data/deepglobe'
MERGED_ROOT = '../data/merge'


def map_class_to_color(image_path, output_path):
    # Open the image
    img = Image.open(image_path)

    # Get the width and height of the image
    width, height = img.size

    turquoise = (0, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    yellow = (255, 255, 0)
    pink = (255, 0, 255)
    white = (255, 255, 255)

    unknown = (0, 0, 0)
    urban_land = (1, 1, 1)
    forest_land = (2, 2, 2)
    water = (3, 3, 3)
    agriculture_land = (4, 4, 4)
    range_land = (5, 5, 5)
    barren_land = (6, 6, 6)

    # TODO ROAD ???
    # road = (2, 2, 2)

    emptySet = set()

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = img.getpixel((x, y))
            # Put image label in correct

            if r == 0 and g == 0 and b == 0:
                # unknown
                img.putpixel((x, y), unknown)
                emptySet.add("black")
            elif r == 0 and g == 255 and b == 255:
                # urban_land
                img.putpixel((x, y,), urban_land)
                emptySet.add("turquoise")
            elif r == 0 and g == 255 and b == 0:
                # forest_land
                img.putpixel((x, y), forest_land)
                emptySet.add("green")
            elif r == 0 and g == 0 and b == 255:
                # water
                img.putpixel((x, y), water)
                emptySet.add("blue")
            elif r == 255 and g == 255 and b == 0:
                # agriculture_land
                img.putpixel((x, y), agriculture_land)
                emptySet.add("yellow")
            elif r == 255 and g == 0 and b == 255:
                # range_land
                img.putpixel((x, y), range_land)
                emptySet.add("pink")
            elif r == 255 and g == 255 and b == 255:
                # barren_land
                img.putpixel((x, y), barren_land)
                emptySet.add("white")
            else:
                # missmatch unknown class
                print(f"rgb out of class bounce ({r},{g},{b})")
                return
    print(output_path)
    print(emptySet)

    # Save the modified image
    img.save(output_path)


def split_images(dir, OUTPUT_DIR="../data/merge"):
    IMGS_DIR = dir + "/images"
    MASKS_DIR = dir + "/masks"
    OUTPUT_COLOR_MASKS_DIR = OUTPUT_DIR + "/colormasks"
    TARGET_SIZE = 512
    # OUTPUT_DIR = "../data/merge"

    landcoverai = False
    deepglobe = False

    if "landcover.ai" in dir:
        img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
        mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))
        img_paths.sort()
        mask_paths.sort()
        landcoverai = True

    if "deepglobe" in dir:
        copy_files_with_word(dir, IMGS_DIR, "sat")
        copy_files_with_word(dir, MASKS_DIR, "mask")
        img_paths = glob.glob(os.path.join(IMGS_DIR, "*.jpg"))
        mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.png"))
        img_paths.sort()
        mask_paths.sort()
        deepglobe = True

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_COLOR_MASKS_DIR):
        os.makedirs(OUTPUT_COLOR_MASKS_DIR)

    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    out_img_path = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k))
                    if os.path.exists(out_img_path):
                        print(f"The image-file {out_img_path} already exists.")
                        continue
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k))
                    out_color_mask_path = os.path.join(OUTPUT_COLOR_MASKS_DIR, "{}_{}_m.png".format(mask_filename, k))
                    # if os.path.exists(out_color_mask_path):
                    #     print(f"The mask-file {out_color_mask_path} already exists.")
                    #     continue
                    cv2.imwrite(out_color_mask_path, mask_tile)
                    if os.path.exists(out_mask_path):
                        print(f"The mask-file {out_mask_path} already exists.")
                        continue
                    if deepglobe:
                        # map class to color
                        map_class_to_color(out_color_mask_path, out_mask_path)
                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
    check_pixel_size(OUTPUT_DIR)


def check_pixel_size(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if not image_files:
        print("No image files found in the specified folder.")
        return

    first_image_path = os.path.join(folder_path, image_files[0])
    with Image.open(first_image_path) as reference_image:
        reference_size = reference_image.size

    globalSize = 0

    for image_file in image_files[1:]:
        current_image_path = os.path.join(folder_path, image_file)
        with Image.open(current_image_path) as current_image:
            globalSize = current_image.size
            if current_image.size != reference_size:
                print(
                    f"Pixel size of {current_image_path} is different from the reference image. {current_image.size} doesn't match {reference_size}")
                return

    print(f"All images have the same pixel size of {globalSize}.")
    print(f"Amount of pictures {len(image_files)}.")


def copy_files_with_word(source_dir, destination_dir, keyword):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)

        # Check if the file contains the specified keyword
        if os.path.isfile(source_path) and keyword in filename:
            newFilename = re.sub(r'(_sat|_mask)', '', filename)
            destination_path = os.path.join(destination_dir, newFilename)

            # Copy the file to the destination directory
            shutil.copyfile(source_path, destination_path)
            print(f"Copied: {filename} to {newFilename}")
