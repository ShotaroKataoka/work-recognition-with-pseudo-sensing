import os
from glob import glob

from PIL import Image


def load_resize_image(image_path, size=(100, 100)):
    with Image.open(image_path) as img:
        cropped_img = img.crop((80, 30, 810, 545))
        padding = (810 - 80) - (545 - 30)
        padded_img = Image.new('RGB', (810 - 80, 810 - 80), (255, 255, 255))
        padded_img.paste(cropped_img, (0, padding // 2))

        resized_img = padded_img.resize(size)
    return resized_img

def get_i_k(image_file):
    i, k = map(int, os.path.basename(image_file).split('.')[0].split('_')[1:])
    return i, k

def make_grid_wavelets(image_dir, total_width=1980):
    image_files = glob(os.path.join(image_dir, 'wavelet_*.png'))
    max_i = 0
    max_k = 0
    for image_file in image_files:
        i, k = get_i_k(image_file)
        max_i = max(max_i, i)
        max_k = max(max_k, k)
    grid_cols = max_k + 1
    grid_rows = max_i + 1

    single_width = 1980 // grid_cols
    single_height = single_width

    total_height = single_height * grid_rows
    final_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    for index, image_file in enumerate(image_files):
        i, k = get_i_k(image_file)
        resized_image = load_resize_image(image_file, size=(single_width, single_height))
        final_image.paste(resized_image, (k * single_width, i * single_height))

    final_image.save(os.path.join(image_dir, 'grid_wavelets.png'))


if __name__ == "__main__":
    root_dir = 'pseudo_sensing_output'
    sensor_dirs = glob(os.path.join(root_dir, '*', 'wavelets'))
    for sensor_dir in sensor_dirs:
        print(f"Processing {sensor_dir}")
        make_grid_wavelets(sensor_dir)
