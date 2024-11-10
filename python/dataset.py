from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = 933120000

def split_image(image_path, output_folder, square_size):
    image = Image.open(image_path)
    width, height = image.size

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the number of squares needed based on `square_size`
    columns = (width + square_size - 1) // square_size  # Ceiling division
    rows = (height + square_size - 1) // square_size

    # Loop through the grid to crop the image
    for i in range(columns):
        for j in range(rows):
            left = i * square_size
            upper = j * square_size
            right = min(left + square_size, width)
            lower = min(upper + square_size, height)
            cropped_image = image.crop((left, upper, right, lower))
            cropped_image.save(os.path.join(output_folder, f"{i}_{j}.png"))


# split_image("imgs/naip.png", "output/base", 500)
# split_image("imgs/usda.png", "output/crops", 500)

def delete_non_square_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            b = False
            with Image.open(image_path) as img:
                if img.width != img.height:
                    b = True
            if b:
                os.remove(image_path)
                print(f"Deleted {image_path}")

# delete_non_square_images("output/base")
# delete_non_square_images("output/crops")


def delete_images_without_red(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            red_found = False
            with Image.open(image_path) as img:
                for pixel in img.getdata():
                    if pixel[0] > 200:
                        red_found = True
                        break
            if not red_found:
                os.remove(image_path)
                print(f"Deleted {image_path}")

# delete_images_without_red("output/crops")
# def compare_and_delete(folder1, folder2):
#     files1 = set(os.listdir(folder1))
#     files2 = set(os.listdir(folder2))
#
#     unique_files = files1.union(files2) - files1.intersection(files2)
#     for file in unique_files:
#         file1_path = os.path.join(folder1, file)
#         file2_path = os.path.join(folder2, file)
#
#         if file in files1 and os.path.exists(file1_path):
#             os.remove(file1_path)
#             print(f"Deleted {file1_path}")
#
#         if file in files2 and os.path.exists(file2_path):
#             os.remove(file2_path)
#             print(f"Deleted {file2_path}")
#
# compare_and_delete("output/base", "output/crops")

def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                pixels = img.load()
                for i in range(img.width):
                    for j in range(img.height):
                        r, g, b = pixels[i, j][:3]
                        if r > 200:
                            pixels[i, j] = (255, 255, 255)  # white
                        else:
                            pixels[i, j] = (0, 0, 0)  # black
                img.save("output/processed/" + filename)
process_images("output/crops")