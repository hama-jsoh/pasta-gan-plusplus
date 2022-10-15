# Simple PASTA-GAN-plusplus inference server

## Getting started

### docker build & run
```bash
USER_ID=$UID docker-compose up -d
docker exec -it [container_name] bash
```

### Inference
```bash
bash inference.sh 3
```
  
*options*
+ 1 : upper
+ 2 : pants
+ 3 : full

### utils

1. resize
```bash
cd utils/
python3 resize_image.py
```

2. make test_pairs.txt
```bash
cd utils/
python3 make_pairs.py
```
