import os
import scipy
import imageio
from PIL import Image

def resize_image(f_name, output_name, resize_shape=(90, 160)):
    im = Image.open(f_name)
    sampled_image = im.resize(resize_shape)
    imageio.imwrite(output_name, sampled_image)
    
def downsample(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for loc in os.listdir(src_path):
        loc_path = "/" + loc
        if not os.path.exists(dst_path + loc_path):
            os.makedirs(dst_path + loc_path)
        for category in os.listdir(src_path + loc_path):
            if category != "here":
                continue
            category_path = loc_path + "/" + category
            if not os.path.exists(dst_path + category_path):
                os.makedirs(dst_path + category_path)
            for image in os.listdir(src_path + category_path):
                image_path = category_path + "/" + image
                if not os.path.exists(dst_path + image_path):
                    os.makedirs(dst_path + image_path)
                for chapter in os.listdir(src_path + image_path):
                    chapter_path = image_path + "/" + chapter
                    counter = 1
                    if not os.path.exists(dst_path + chapter_path):
                        os.makedirs(dst_path + chapter_path)
                    for sample in os.listdir(src_path + chapter_path):
                        sample_path = chapter_path + "/" + sample
                        if counter >= 10:
                            counter = 0
                            if not os.path.exists(dst_path + sample_path):
                                if category == "here":
                                    resize_image(src_path + sample_path, dst_path + sample_path, (200, 216))
                                if category == "tomtom":
                                    resize_image(src_path + sample_path, dst_path + sample_path, (150, 216))
                                else:
                                    resize_image(src_path + sample_path, dst_path + sample_path, (90, 160))
                        counter += 1
                    print("Finishes chapter:" + chapter_path)
        print("Finishes location:" + loc_path)
    print("All done!")
downsample("F:/Drive360challenge_images/Drive360Images", "./data/full_downsampled")
