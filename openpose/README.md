# Simple OpenPose

## Getting Started

### Step 1: Set Up the Folder Structure (models)
```bash
bash setup.sh
```


### Step 2: Put data
```
data/
└── human
    ├── human_00.jpg
    ├── human_01.jpg
    └── ...
```

### Step 3: Run
```bash
python3 openpose.py
```
  
### Step 4: Check output
```bash
ls data/samples
```

#
### Simple Example
```python
import openpose


if __name__ == "__main__":

    # openpose configuration
        pose = OpenPose(
		model="coco",
		verbose=True,
	)

    # run openpose
    kpts = pose.Inference(dataroot="./data/human")

    # fileio
    pose.FileOutput(
	dict_obj=kpts,
        indent=False,
    )
```
