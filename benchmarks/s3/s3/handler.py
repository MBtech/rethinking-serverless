import boto3
from time import time
import json

s3_client = boto3.client('s3')

def handle(event):
    data = json.loads(event)

    input_bucket = data['input_bucket']
    object_key = data['object_key']
    output_bucket = data['output_bucket']

    path = '/tmp/'+object_key.split('/')[-1]

    start = time()
    s3_client.download_file(input_bucket, object_key, path)
    download_time = time() - start

    start = time()
    s3_client.upload_file(path, output_bucket, object_key)
    upload_time = time() - start

    return {"download_time": download_time, "upload_time": upload_time}