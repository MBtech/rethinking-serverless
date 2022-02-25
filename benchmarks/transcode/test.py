import ffmpeg
import urllib.request
import boto3

# in_filename = 'input.mp4'
in_filename = "https://faas-testing.s3-us-west-2.amazonaws.com/video/beach.mp4"
out_filename = 'thumbnail.jpg'
time = 1
width=1080
# urllib.request.urlretrieve("https://faas-testing.s3-us-west-2.amazonaws.com/video/fjord.mp4", in_filename)
ffmpeg.input(in_filename, ss=time).filter('scale', width, -1).output(out_filename, vframes=1).run()

ffmpeg.input(in_filename).output('output.mkv').overwrite_output().run()

s3 = boto3.resource('s3')
BUCKET = "faas-testing"

s3.Bucket(BUCKET).upload_file('output.mkv', 'video/output/output.mkv')