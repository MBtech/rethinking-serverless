docker buildx build --push --platform linux/arm64/v8,linux/amd64 --build-arg ADDITIONAL_PACKAGE="g++" --tag mbilalce/linpack .