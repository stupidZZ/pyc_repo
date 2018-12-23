# Requirements
* python 3.6(not test for python2 yet)
* requests
* requests_ntlm
* urllib3
* easydict

# Example
````bash
# Step 0: set some alias for convenience
export CLUSTER="wu1"
export VC="resrchvc"
export FOLDER="<your folder name>"
export USR="<your user name>"     # only for login philly
alias philly-fs="bash ./philly-fs/linux/philly-fs.bash"

# Step 1: create your folders
# (if you already create, please skip)
philly-fs -mkdir //philly/${CLUSTER}/${VC}/${FOLDER}
philly-fs -mkdir //philly/${CLUSTER}/${VC}/${FOLDER}/data
philly-fs -mkdir //philly/${CLUSTER}/${VC}/${FOLDER}/model
philly-fs -mkdir //philly/${CLUSTER}/${VC}/${FOLDER}/model/pretrained_model

# Step2: upload pretrained_model and data to your folder
# (if you already upload, please skip)
# (you had better zip small files first, 
#  e.g., JPEGImages, Annotations, SegmentationClass, Segmentation Object, 
#  and use phillyzip reader to read small files)
philly-fs -cp -r VOCdevkit //philly/${CLUSTER}/${VC}/${FOLDER}/data
philly-fs -cp resnet_v1_101-0000.params //philly/${CLUSTER}/${VC}/${FOLDER}/model/pretrained_model

# Step3: upload script run_on_philly.py
philly-fs -cp run_on_philly.py //philly/${CLUSTER}/${VC}/${FOLDER}

# Step4: submit jobs to philly
# (you may replace cfgs/submit_template.yaml with your job config)
python submit.py --cfg cfgs/submit_template.yaml --usr ${USR} --folder-user ${FOLDER}
# (after executing above command, you should input your password for login)
````

# Submit Your Own Experiment to Philly
Next, we will use cfgs/submit_template.yaml as  an example to demonstrate how to write an experiment config.
**Here "experiment" means you can submit multiple jobs in one time for some purpose, e.g. tuning hyper-parameters(grid search).** 
## philly basic config

````yaml
cluster: wu1
vc: resrchvc
submit_user: v-wesu
user_folder: "/hdfs/resrchvc/v-wesu"
````
* **cluster**: philly physical cluster name 
* **vc**: philly virtual cluster name
* **submit_user**: alias(must be in the above vc group) you use to submit your job
* **user_folder**: folder you create on hdfs/gfs(we assume run_on_philly.py is in this folder)
## git related config 
(if you don't use git, you can skip this part.)
````yaml
git:
  enabled: true
  philly_script: "run_on_philly.py"
  root: "github.com"
  repo: "msracver/Deformable-ConvNets"
  bran: master
  auth: 'git' # replace with your deployment key
  home_dir: '/tmp'
  work_dir: 'Deformable-ConvNets'
````
* **enabled**: flag to enable git
* **philly_script**: the script to handle running job with git, default is "run_on_philly.py".
* **root**: "github.com", "gitlab.com", etc.
* **repo**: git repo name
* **bran**: git branch
* **auth**: deployment key/personal access token(P.S.: if you use gitlab, maybe you should use "oauth2:<your_deploy_key>" instead)
* **home_dir**: the root path to clone code to philly, default is "/tmp"
* **work_dir**: relative path of your work directory to above home_dir 

## job related config
````yaml
job:
  name_prefix: test
  type: custom_v2
  exec_file: "experiments/faster_rcnn/rcnn_end2end_train_test.py"
  ngpu: 4
  rack: anyConnected
  docker_repo: custom
  lib: mxnet
  tag: v110_py27_cuda9
  params:
    - name: --cfg
      value:
        - "experiments/faster_rcnn/cfgs/resnet_v1_101_voc0712_rcnn_end2end.yaml"
        - "experiments/faster_rcnn/cfgs/resnet_v1_101_voc0712_rcnn_dcn_end2end.yaml"
      displayed: true
````
* **name_prefix**: prefix of job name displayed on Philly Web Portal.
* **type**: now only support custom_v2 job, so don't need to modify this.
* **exec_file**: relative path of file-to-execute to work directory  
* **ngpu**: number of gpu for each job in your experiment. 
* **rack**: this is for specifying Philly RackId, usually you don't need specify this, just use "anyConnected"
* **docker_repo**: "custom" or "test" 
    * "custom" for using docker in "philly/jobs/custom"
    * "test" for using docker in "philly/jobs/test"
* **lib**: framework you use, e.g., mxnet, pytorch, tensorflow
* **tag**: docker tag, e.g. v110_py27_cuda9
* **params**: list of command-line parameters to use when running file-to-execute
    * **name**: parameter name, e.g., '--cfg'
    * **value**: list of value in your experiment.
    * **displayed**: whether to display this parameter in job name of Philly Web Portal

**Here we provide an toy example for grid search, for example, you can specify params like following**:
```yaml
params:
  - name: --lr
    value: 
    - 0.1
    - 0.01
    - 0.001
    displayed: true
  - name: --wd
    value:
    - 0.0001
    - 0.0005
    displayed: true
  - name: --momentum
    value: 
    - 0.9
    - 0.99
    displayed: true
```
**Then we will submit 3\*2\*2 = 12 jobs to Philly, each job running with a command-line parameter combination of lr, wd and momentum.**



