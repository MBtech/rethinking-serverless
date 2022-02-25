#!/bin/bash

faas-cli build --build-arg ADDITIONAL_PACKAGE="ffmpeg libva-intel-driver py-pip" -f transcode.yml

docker buildx build --build-arg ADDITIONAL_PACKAGE="ffmpeg libva-intel-driver py-pip" --push --platform linux/arm/v7,linux/arm64/v8,linux/amd64 --tag mbilalce/transcode . 