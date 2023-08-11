# MMClassification

Allows processing of images with [MMClassification](https://github.com/open-mmlab/mmclassification).

Uses PyTorch 1.9.0 and CUDA 11.1.

## Version

MMClassification github repo tag/hash:

```
v0.23.1
d2e505415040bf5329ab218bb6fe3d899f176cd5
```

and timestamp:

```
June 16th, 2022
```

## Quick start

### Inhouse registry

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --gpus=all --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmclassification:0.23.1_cuda11.1
  ```

### Docker hub

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --gpus=all --shm-size 8G \
    -v /local/dir:/container/dir \
    -it waikatodatamining/mmclassification:0.23.1_cuda11.1
  ```

### Build local image

* Build the image from Docker file (from within /path_to/mmclassification/0.23.1_cuda11.1)

  ```bash
  docker build -t mmcls .
  ```
  
* Run the container

  ```bash
  docker run --gpus=all --shm-size 8G -v /local/dir:/container/dir -it mmcls
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Publish images

### Build

```bash
docker build -t mmclassification:0.23.1_cuda11.1 .
```

### Inhouse registry  

* Tag

  ```bash
  docker tag \
    mmclassification:0.23.1_cuda11.1 \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmclassification:0.23.1_cuda11.1
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmclassification:0.23.1_cuda11.1
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  

* Tag

  ```bash
  docker tag \
    mmclassification:0.23.1_cuda11.1 \
    waikatodatamining/mmclassification:0.23.1_cuda11.1
  ```
  
* Push

  ```bash
  docker push waikatodatamining/mmclassification:0.23.1_cuda11.1
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ``` 

## Scripts

The following scripts are available:

* `mmcls_config` - for expanding/exporting default configurations (calls `/mmclassification/tools/misc/print_config.py`)
* `mmcls_train` - for training a model (calls `/mmclassification/tools/train.py`)
* `mmcls_predict_poll` - for applying a model to images (uses file-polling, calls `/mmclassification/tools/predict_poll.py`)
* `mmcls_predict_redis` - for applying a model to images (via [Redis](https://redis.io/) backend), 
  add `--net=host` to the Docker options (calls `/mmclassification/tools/predict_redis.py`)


## Usage

* The dataset has a simple format, with each sub-folder representing a class.
  
* Store class names in an environment variable called `MMCLS_CLASSES` **(inside the container)**:

  ```bash
  export MMCLS_CLASSES=\'class1\',\'class2\',...
  ```
  
* Alternatively, have the class anmes stored in a text file with the classes separated by commas and the `MMCLS_CLASSES`
  environment variable point at the file.
  
  * The classes are stored in `/data/labels.txt` either as comma-separated list (`class1,class2,...`) or one per line.
  
  * Export `MMCLS_CLASSES` as follows:

    ```bash
    export MMCLS_CLASSES=/data/labels.txt
    ```

* Use `mmcls_config` to export the config file (of the model you want to train) from `/mmclassification/configs` 
  (inside the container), then follow [these instructions](#config).

* Train

  ```bash
  mmcls_train /path_to/your_data_config.py \
      --work-dir /where/to/save/everything
  ```

* Predict and output JSON files with the classes and their associated scores

  ```bash
  mmcls_predict_poll \
      --model /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --prediction_in /path_to/test_imgs \
      --prediction_out /path_to/test_results
  ```
  Run with `-h` for all available options.

* Predict via Redis backend

  You need to start the docker container with the `--net=host` option if you are using the host's Redis server.

  The following command listens for images coming through on channel `images` and broadcasts
  predicted images on channel `predictions`:

  ```bash
  mmcls_predict_redis \
      --model /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --redis_in images \
      --redis_out predictions
  ```
  
  Run with `-h` for all available options.


## Example config files

You can output example config files using (stored under `/mmclassification/configs` for the various network types):

```bash
mmcls_config /path/to/my_config.py
```

You can browse the config files [here](https://github.com/open-mmlab/mmclassification/tree/v0.23.1/configs).


## <a name="config">Preparing the config file</a>

* If necessary, change `num_classes` to number of labels (background not counted).
* Change `dataset_type` to `ExternalDataset` and any occurrences of `type` in the `train`, `test`, `val` 
  sections of the `data` dictionary.
* Change `data_prefix` to the path of your dataset parts (the directory containing `train` and `val` directories).
* Set `ann_file` occurrences to `None`   
* Interval in `checkpoint_config` will determine the frequency of saving models while training 
  (10 for example will save a model after every 10 epochs).
* In the `runner` property, change `max_epocs` to how many epochs you want to train the model for.
* Change `load_from` to the file name of the pre-trained network that you downloaded from the model zoo instead
  of downloading it automatically.
* If you want to include the validation set, add `, ('val', 1)` to `workflow`.

_You don't have to copy the config file back, just point at it when training._

**NB:** A fully expanded config file will get placed in the output directory with the same
name as the config plus the extension *.full*.


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```

## Caching models

PyTorch downloads base models, if necessary. However, by using Docker, this means that 
models will get downloaded with each Docker image, using up unnecessary bandwidth and
slowing down the startup. To avoid this, you can map a directory on the host machine
to cache the base models for all processes (usually, there would be only one concurrent
model being trained):  

```
-v /somewhere/local/cache:/.cache
```

Or specifically for PyTorch:

```
-v /somewhere/local/cache/torch:/.cache/torch
```

**NB:** When running the container as root rather than a specific user, the internal directory will have to be
prefixed with `/root`. 
