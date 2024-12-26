import gdown

file_id = "1Y5cV5IkWvQeM-js7aQZZAcjGXzcdHm_1"
url = f"https://drive.google.com/uc?id={file_id}"
output = "pretrain_ckpt.pth.tar"

gdown.download(url, output, quiet=False)