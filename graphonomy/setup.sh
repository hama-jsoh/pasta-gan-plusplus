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

	download_google_drive 1sbP4gVY5ryLM_zTDNZTdOkKN7lydybJz inference.pth \
	 && mkdir -p ./data/pretrained_model/ \
	 && mv inference.pth ./data/pretrained_model/

}

main
