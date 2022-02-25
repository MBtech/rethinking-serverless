# Serverless Applications
This directory contains serverless applications that can run on OpenFaaS. 

## Applications
- Face Detection
- Linpack
- OCR
- Face Blur
- S3 (Transfer in/out)
- Transcode (video transcoding)

Each application is in its corresponding directory. These directories contain a yaml file (often named stack.yml) that contain the necessary information to build a dockerfile using `faas-cli`.
The starting point of the function is usually `handler.py` in the directory that you can see in the yaml file under `functions.<function_name>.handler` 
 
## Generating a docker file for a function
A docker file to create a docker image to run on OpenFaaS can be generated using `faas-cli build`. For example the following generates a docker file to build gzip application's docker image. 
```
faas-cli build -f gzip.yml --shrinkwrap
```

Create a multi-arch docker image:
```
docker buildx create --use --name multi-arch --driver-opt image=moby/buildkit:master
```