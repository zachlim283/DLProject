<h1 align="center" style="border-bottom: none">
    <b>Inertial-Based Real-Time Human Action Recognition For Physiotherapy</b>
    <br>
    Deep Learning
    <br>
</h1>

<p align="center">
    A physical therapy exercise classifier using inertial sensor data from the PHYTMO (PHYsical Therapy MOnitoring) dataset, which contains inertial measurement unit (IMU) sensor recordings of common physical therapy exercises.
</p>


<table align="center">
    <h2>Group 19: </h2>
    <tr>
        <td align="center"><a href="https://github.com/zachlim283"><img src="https://avatars.githubusercontent.com/u/91017355?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zach Lim</b><br/>1002141</sub></a><br/>
        <td align="center"><a href="https://github.com/bloomspx"><img src="https://avatars.githubusercontent.com/bloomspx?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Soh Pei Xuan</b><br/>1005552</sub></a><br/>
    </tr>
</table>

# Table of Contents
-   [File Directories](#file-directories)
-   [Getting Started](#getting-started)

# File Directories
```
┣ 📂 input                 # stores all input data
┣ 📂 logs                  # stores all training logs
┣ 📂 models                # stores all trained model weights
┣ 📂 notebooks/training    # stores all .ipynb used for training models
┣ 📂 src                   # stores all helper functions for train / test 
┣ 📂 tensorboard           # stores all tensorboard logs / graphs during training
┣ 📄 main.ipynb            
┣ 📄 requirements.txt     
 ```

# Getting Started

1. Create Virtual Environment and Install Dependencies
```
python -m venv .env
source .env/bin/activate  # for linux/macos
.env\Scripts\activate  # for windows
pip install -r requirements.txt
```

2. Refer to `main.ipynb` to run and test models