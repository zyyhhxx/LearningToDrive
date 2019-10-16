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
        for camera in os.listdir(src_path + loc_path):
            if camera == "go_pro_4":
                camera_path = loc_path + "/" + camera
                if not os.path.exists(dst_path + camera_path):
                    os.makedirs(dst_path + camera_path)
                for image in os.listdir(src_path + camera_path):
                    image_path = camera_path + "/" + image
                    if not os.path.exists(dst_path + image_path):
                        os.makedirs(dst_path + image_path)
                    for chapter in os.listdir(src_path + image_path):
                        chapter_path = image_path + "/" + chapter
                        if not os.path.exists(dst_path + chapter_path):
                            os.makedirs(dst_path + chapter_path)
                        for sample in os.listdir(src_path + chapter_path):
                            sample_path = chapter_path + "/" + sample
                            if not os.path.exists(dst_path + sample_path):
                                resize_image(src_path + sample_path, dst_path + sample_path)
                        print("Finishes chapter:" + chapter_path)
        print("Finishes location:" + loc_path)
    print("All done!")
downsample("./data/Sample1", "./data/Sample1_downsampled_2")
