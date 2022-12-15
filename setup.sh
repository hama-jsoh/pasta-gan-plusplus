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

	download_google_drive 1QgIQJ83FXE9XLUhKdY1RK-cHr5PGAa8V UPT_512_320.zip \
	 && unzip UPT_512_320.zip \
	 && rm UPT_512_320.zip
  
	download_google_drive 1k5QTVzd1B67--Y7WGejbRVA1Cgg6Wy2P network-snapshot-004408.pkl \
	 && mkdir -p pasta-gan-plusplus/checkpoints/pasta-gan++/ \
	 && mv network-snapshot-004408.pkl pasta-gan-plusplus/checkpoints/pasta-gan++/
	 
}

main
