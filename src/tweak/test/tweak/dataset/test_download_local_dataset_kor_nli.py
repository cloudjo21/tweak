from datasets import load_dataset, load_from_disk
from pprint import pprint


#dataset = load_dataset(
#  path="/mnt/d/temp/user/ed/mart/kor_nli",
#  #data_dir="/mnt/d/temp/user/ed/mart/kor_nli",
#  #download_mode="forece_redownload"
#)
dataset = load_from_disk(
 '/mnt/d/temp/user/ed/mart/kor_nli'
)

pprint(dataset)
pprint(dataset['train'])
pprint(dataset['validation'])
