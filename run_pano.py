import argparse
import os
import time

from panoramic_video_generator import PanoramicVideoGenerator


def main(file_path, num_images):
    file_name = os.path.basename(file_path)
    exp_no_ext = file_name.split('.')[0]
    os.system('mkdir dump')
    os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
    os.system(('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(os.path.join('dump', '%s', '%s%%03d.jpg'))) % (
        file_path, exp_no_ext, exp_no_ext))

    s = time.time()
    panorama_generator = PanoramicVideoGenerator(os.path.join('dump', '%s') % exp_no_ext,
                                                 exp_no_ext, num_images)
    panorama_generator.align_images(translation_only='boat' in file_path)
    panorama_generator.generate_panoramic_images(9)
    print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

    panorama_generator.save_panoramas_to_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate panoramic images from a sequence of images.")
    parser.add_argument("--input_dir", type=str, required=True, default='boat.mp4', help="path to input images")
    parser.add_argument("--num_images", type=int, required=False, default=2100, help="number of images in the sequence")
    args = parser.parse_args()
    main(*vars(args).values())
