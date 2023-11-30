

# data root
LANDCOVER_AI_ROOT = '../data/landcover.ai.v1'
DEEPGLOBE_ROOT = '../data/deepglobe'
MERGED_ROOT = '../data/merge'


from PIL import Image

def map_class_to_color(image_path, output_path):
    # Open the image
    img = Image.open(image_path)

    # Get the width and height of the image
    width, height = img.size

    turquoise = (0, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    building = 1
    woodland = 2
    water = 3
    road = 4

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = img.getpixel((x, y))

            # Check if the pixel values fall within the specified range (1 to 5)
            if 1 <= r <= 5 and 1 <= g <= 5 and 1 <= b <= 5:
                if r == g and g == b:
                    if r == building:
                        img.putpixel((x, y), turquoise)
                    elif r == woodland:
                        img.putpixel((x, y), green)
                    elif r == water:
                        img.putpixel((x, y), blue)
                    elif r == road:
                        img.putpixel((x, y), black)
                    else:
                        print(f"not a valid class. Pixel: ({r},{g},{b})")
                        return
                else:
                    print(f"not a number between 1 and 5. Pixel: ({r},{g},{b})")
                    return

    # Save the modified image
    img.save(output_path)


# Example usage:
input_image_path = "C:/Users/Uni/Documents/DLIVP/dlivp-workspace/data/merge/N-33-130-A-d-4-4_214_m.png"
output_image_path = "../data/N-33-130-A-d-4-4_214_m.png"

modify_image(input_image_path, output_image_path)