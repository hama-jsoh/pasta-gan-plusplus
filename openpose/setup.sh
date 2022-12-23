#!/bin/bash

download_google_drive() {

	wget --load-cookies ~/cookies.txt \
	  "https://docs.google.com/uc?export=download&confirm=\
	  $(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate \
	  'https://docs.google.com/uc?export=download&id=$1' \
	  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" \
	  -O $2 \
	  && rm -rf ~/cookies.txt

}

main() {

	download_google_drive 13R1vpJrjxFNt7t-jNn-69DRCpECLsxNM pose_iter_440000.caffemodel \
	 && mkdir pretrained_models \
	 && mv pose_iter_440000.caffemodel pretrained_models

	download_google_drive 1elkx5n4xr1Re6mTHaBMcMSRObioo8LLN pose_deploy_linevec.prototxt \
	 && mv pose_deploy_linevec.prototxt pretrained_models

}

main
