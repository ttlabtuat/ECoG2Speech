# ECoG2Speech

ECoG2Speech is a deep learning-based text or speech decoding from ECoG.


## Installation
There are two options for installation: using `requirements.txt` or `Dockerfile`.

### Using "requirements.txt"
```sh
pip install -r requirements.txt
```

### Using "Dockerfile"
Build a Docker image (e.g., ecog2speech_img). If you are using GPUs, comment out line L1 and uncomment line L2 in the Dockerfile: "FROM nvidia/cuda:11.2-runtime-ubuntu20.04".

```sh
 docker build ./ -t ecog2speech_img
```

Run a container. If you are using GPUs, add the "--gpus all" option.

```sh
 docker run -v ${PWD}:/work --rm -it ecog2speech_img
```


## Usage

### Easy running
You can execute the entire process, from data preparation to testing, using randomly shuffled data.

Making sample data:

```./make_sample_data.py```


Training and testing:

```./sample_run.sh```



### Preparing data
You'll need to prepare two types of data:
- ECoG data (e.g., aaa.npy, bbb.npy,...) 
- Wave data (e.g., aaa.wav, bbb.wav,...)
- Electrode configuration file (e.g., elec.csv)

The ECoG data should be in the form of a numpy ndarray file (npy) with two dimensions (time and electrodes).

The wave data should be in the form of a wave file. 
Wave files need to be prepared, each corresponding to an ECoG data.


The Electrode configuration file describes the numbers of electrodes. For example, if you have the numbers: "1, 2, 3, 6", the description would be as follows:

```1,2,3,6``` 


### Making list file
You'll need to make the list file (csv format) to input the data into training or testing scripts. 
The header of the csv file is the follows:

```ecog_filename,ecog_fs,wav_filename,transcripts,electrodes ```


- ecog_filename: Path to an ecog file
- ecog_fs: sampling frequency of the ecog file
- wav_filename: Path to a wave file
- transcript: Overt, percieved, or covert speech content
- electrodes: Electrode configuration file



### Preparing configuration file
The sample configuration file is attached on "./conf/config_xA000.ini". 
The naming convention for this configuration file is strict. 
If you choose to copy and modify this configuration file for your custom settings, please take care to follow the naming convention when specifying the name of the destination file.

```The naming convention: config_x${XXXX}.ini```

You are free to choose any name for ${XXXX}, except for the use of the letter 'x'.



### Training
To initiate the training process, execute the following command:

```shell
./ecog2text_train.py ${list_file} ${config} --model ${model_name}
```

- `${list_file}` should be replaced with the path to the list file intended for training.
- `${config}` should be substituted with the path to the configuration file (config_x\${XXXX}.ini).
- `${model_name}` should be the desired name for the output model file. 

Feel free to choose any name for `${model_name}` as per your preference. 

The training results are saved under the following name:

```./exp/x${XXXX}_${list_file_name}/${model_name}```


### Testing
Execute the following command:

```./ecog2text_eval.py ${list_file} ${model_directory}```


- `${list_file}` should be replaced with the path to the list file intended for testing.
- `${model_directory}` should be replaced with the path to the training results.
