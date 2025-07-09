import os
from PIL import Image, ImageChops

input_folder = "/Users/hosannahhailye/Documents/Hosi/University/2nd P.2/summer '25/APS360/Data/images_original/pop" 
output_folder = "/Users/hosannahhailye/Documents/Hosi/University/2nd P.2/summer '25/APS360/Data/images_original/pop_clean"   

def crop_whitespace(image):
    #crops the whitespace around the spectrogram
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

def process_images(input_folder, output_folder, size=(227, 227)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            path = os.path.join(input_folder, filename)
            img = Image.open(path).convert("RGB")
            cropped = crop_whitespace(img)
            resized = cropped.resize(size)
            resized.save(os.path.join(output_folder, filename))

process_images(input_folder, output_folder)
print("finito")