#!/bin/bash
docker buildx build --file service/Dockerfile --no-cache --platform linux/amd64 --push -t europe-west3-docker.pkg.dev/composed-hold-390914/docker-registry/ml-webservice:latest  .