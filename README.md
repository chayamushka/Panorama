# Panoramic Video Generator
This project takes a video as input and produces a panoramic output gif.

## Requirements
 - ffmpeg
 - NumPy
 - SciPy
 - OpenCV
## Usage
Place the input video in the videos folder
Run the script with the following command:
```
python main.py --input_dir <input_video_name> --num_images <number_of_images>
```
## Options
 - input_dir: name of the input video (e.g. my_video.mp4)
 - num_images: number of images to generate from the video. The default value is 2100.
## Output
The output video will be saved in the root directory with the name panoramic_<input_video_name>.

## Notes
 - The input video filming direction should be towards the right.
 - The translation_only argument in the align_images function is currently set to True only if the input video file name contains the word "boat". You may change this behavior if desired.
 - You should be able to use the command 'ffmpeg' in the command line [link for help](https://bobbyhadz.com/blog/ffmpeg-is-not-recognized-as-internal-or-external-command)



## Examples!
Here are some examples of panoramic images and videos generated using this project:

### Input 1

https://user-images.githubusercontent.com/62551833/212699365-f4d634d3-bf7c-455b-9a6d-5bb51a79369c.mp4
### Output 1

![boat_movie](https://user-images.githubusercontent.com/62551833/212699173-17d68a72-b57a-4c30-970e-71be6095fb39.gif)

### Input 2

https://user-images.githubusercontent.com/62551833/212699605-29cd338a-fe28-4be6-bd03-265d2963c22d.mp4

### Output 2

![iguazu_movie](https://user-images.githubusercontent.com/62551833/212699677-e787b564-5dbf-4cdf-9c27-780c178d29dc.gif)


