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
```

Next:
```shell
cd hf-experiments
```

Next:
```shell
pip install -r requirements.txt
```