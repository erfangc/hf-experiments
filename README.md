# hf-experiments

Choose:

* Amazon Linux
* AMI: `Deep Learning AMI GPU PyTorch 1.11.0 (Amazon Linux 2) <yyyyMMdd>`
* Instance Type: `g5.xlarge`
* Keypair: any
* Network settings: default

Next:

```ssh
ssh -i "master-key-dev.pem" ec2-user@ec2-<public_id>.compute-1.amazonaws.com
```

Next:

> For latest go to [Start locally | PyTorch](https://pytorch.org/get-started/locally/)

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Next:
```
git clone https://github.com/erfangc/hf-experiments.git
git config --global credential.helper store
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs -y
```

Next:
```shell
cd hf-experiments
```

Next:
```shell
pip install -r requirements.txt
```

Next, login
> Find the tokens [here](https://huggingface.co/settings/tokens)
```shell
huggingface-cli login
```