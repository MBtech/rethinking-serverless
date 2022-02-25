import ffmpeg
import json
import boto3

def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """

    data = json.loads(req)
    in_filename = data['input']
    out_filename = data['output']

    ffmpeg.input(in_filename).output(out_filename).overwrite_output().run()

    s3 = boto3.resource('s3')
    BUCKET = "faas-testing"

    s3.Bucket(BUCKET).upload_file(out_filename, 'video/output/'+out_filename)

    return out_filename
