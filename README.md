# Simple PASTA-GAN-plusplus inference server

## Getting started

### Dataset & model
+ download dataset --> UPT dataset : [Google_drive](https://drive.google.com/file/d/1QgIQJ83FXE9XLUhKdY1RK-cHr5PGAa8V/view?usp=sharing)
+ download pretrained_model --> [Google_drive](https://drive.google.com/file/d/1k5QTVzd1B67--Y7WGejbRVA1Cgg6Wy2P/view?usp=sharing)

----

### 1. Auto Setup
```bash
https://github.com/hama-jsoh/pasta-gan-plusplus.git && cd pasta-gan-plusplus && bash setup.sh
```

### 2. Docker build & run
```bash
USER_ID=$UID docker-compose up -d
docker exec -it [container_name] bash
```

### 3. Inference(test)
```bash
bash test.sh 3
```
  
> *options*
> + 1 : upper
> + 2 : pants
> + 3 : full

### Utils

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
