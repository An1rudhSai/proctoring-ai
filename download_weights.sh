#!/bin/bash

# Define the Google Drive file ID
FILE_ID="1JTC2zLC1pK8nbDJFMIx4VPjQPdnf0Bdd"
DESTINATION="models/yolov3.weights"

# Use wget to download the file
wget --no-check-certificate 'https://docs.google.com/uc?export=download' \
     --post-data='id='"$FILE_ID"'&confirm=$(wget --quiet --save-cookies /tmp/gcookie --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='"$FILE_ID"'&confirm=t' -O- | sed -n '/confirm=/s/.*confirm=\(.*\)$/\1/p')' \
     -O "$DESTINATION"
