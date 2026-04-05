#!/bin/bash

echo "Downloading datasets.tgz..."
curl -L -o datasets.tgz "https://cdn.intra.42.fr/document/document/41980/datasets.tgz"
echo "Downloaded successfully!"

tar -xzf datasets.tgz 

rm datasets.tgz
echo "Datasets extracted to ./datasets"