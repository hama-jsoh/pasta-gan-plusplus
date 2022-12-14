# PASTA-GAN-plusplus Inference Server

----

<p align="center">
  <b> PASTA-GAN++ with AWS </b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.0.1-green?style=flat-square">
  <img src="https://img.shields.io/badge/Python-3.8.x-3776AB?style=flat-square&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/CUDA-11.0.3-76B900?style=flat-square&logo=NVIDIA&logoColor=white">
  <img src="https://img.shields.io/badge/CUDNN-8-76B900?style=flat-square&logo=NVIDIA&logoColor=white">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AWS-Lambda-FF9900?style=flat-square&logo=AWS-Lambda&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon-EC2-FF9900?style=flat-square&logo=Amazon-EC2&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon-RDS-527FFF?style=flat-square&logo=Amazon-RDS&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon-S3-569A31?style=flat-square&logo=Amazon-S3&logoColor=white">
  
  <img src="https://img.shields.io/badge/Amazon-CloudWatch-FF4F8B?style=flat-square&logo=Amazon-CloudWatch&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon-API_Gateway-A100FF?style=flat-square&logo=Amazon-API-Gateway&logoColor=white">
</p>

----

## 1. Getting started

### Dataset & model
+ [Training] download dataset --> UPT dataset : [Google_drive](https://drive.google.com/file/d/1QgIQJ83FXE9XLUhKdY1RK-cHr5PGAa8V/view?usp=sharing)  
+ [Required] download pastagan-plusplus pretrained_model --> [Google_drive](https://drive.google.com/file/d/1k5QTVzd1B67--Y7WGejbRVA1Cgg6Wy2P/view?usp=sharing)
+ [Required] download openpose pretrained_model --> [Google_drive](https://drive.google.com/drive/folders/1Oz_fDTMDSttZMu-Va6kGvE81kaggm3sC?usp=share_link)
+ [Required] download graphonomy pretrained_model --> [Google_drive](https://drive.google.com/file/d/1zOyVygz_4OEcdfqbAR7ZFL5OXOgjd3RW/view?usp=share_link)

----

### Step 1: Set Up the Folder Structure
`setup.sh`  <-- Automatic setup script  
- download & setup dataset
- download & setup model(checkpoint)
```bash
git clone https://github.com/hama-jsoh/pasta-gan-plusplus.git && cd pasta-gan-plusplus && bash setup.sh
```

### Step 2: Docker build & run
```bash
USER_ID=$UID docker-compose -f docker/docker-compose.yaml up -d
```
  - Step 2-1: Enter container  
    ```bash
    docker exec -it pastagan_plusplus_dev bash
    ```
### Step 3: Inference
```bash
python3 -W ignore inference.py
```
### Step 4: Check result
```bash
cd test_results/full && ls
```

----

## 2. EXPERIMENT
### Step 1: TEST
```bash
bash test.sh 3
```
  
> *options*
> + 1 : upper
> + 2 : pants
> + 3 : full

### Step 2: Check result
```bash
cd test_results/full && ls
```

----

## 3. Sample Code
```python
# inference.py
...

if __name__ == "__main__":
    # 1. openpose(preprocessing)
    openpose = PreProcessor(
        UriInput("keypoints", "./test_samples/image"),
        FileOutput("json", "./test_samples/keypoints")
    )
    openpose.start()

    # 2. graphonomy(preprocessing)
    graphonomy = PreProcessor(
        UriInput("parsing", "./test_samples/image"),
        FileOutput("img", "./test_samples/parsing")
    )
    graphonomy.start()

    # 3. write_txt(permutation)
    with open("./test_samples/test_pairs.txt", "w") as f:
        filelist = os.listdir("./test_samples/image")
        cloth, human = filelist
        f.write(f"{cloth} {human}")

    # 4. synthesis_result
    generate_images(
        dataroot='test_samples',
        testtxt='test_pairs.txt',
        outdir='test_results/full',
        testpart='full'
    )
```
