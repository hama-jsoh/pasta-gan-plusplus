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

	# set pastagan model
	download_google_drive 1k5QTVzd1B67--Y7WGejbRVA1Cgg6Wy2P network-snapshot-004408.pkl \
	 && mkdir -p pasta-gan-plusplus/checkpoints/pasta-gan++/ \
	 && mv network-snapshot-004408.pkl pasta-gan-plusplus/checkpoints/pasta-gan++/

	# set graphonomy model
	download_google_drive 1sbP4gVY5ryLM_zTDNZTdOkKN7lydybJz inference.pth \
	 && mkdir -p pasta-gan-plusplus/graphonomy/data/pretrained_model/ \
	 && mv inference.pth pasta-gan-plusplus/graphonomy/data/pretrained_model/

	# set openpose model
	download_google_drive 13R1vpJrjxFNt7t-jNn-69DRCpECLsxNM pose_iter_440000.caffemodel \
	 && mkdir -p pasta-gan-plusplus/pretrained_models \
	 && mv pose_iter_440000.caffemodel pasta-gan-plusplus/pretrained_models/

	# set openpose prototxt
	download_google_drive 1elkx5n4xr1Re6mTHaBMcMSRObioo8LLN pose_deploy_linevec.prototxt \
	 && mv pose_deploy_linevec.prototxt pasta-gan-plusplus/pretrained_models/

}

main
