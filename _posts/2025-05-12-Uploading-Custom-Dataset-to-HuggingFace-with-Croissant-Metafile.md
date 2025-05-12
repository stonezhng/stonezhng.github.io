---
layout: post
title:  "Uploading Custom Dataset to HuggingFace with Croissant Metafile"
date:   2025-05-12 03:00:00 -0400
tags: HuggingFace Croissant
categories: coding
---

# Why this Matters

HuggingFace has a whole system to reformat the uploaded dataset. If you do not follow it you bascially can just use it like an online drive, but if you follow all the requirements to upload your dataset formally it can basically automatically convert your data to parquet format, and generate a fancy dataset card, as well as the croissant metafile, as the meta information of the dataset. 

I want to use the ImageFolder format from the two formats supported by HuggingFace. However, when searching on how to do this from scratch I do feel the offical tutorials are quite limited, like the examples are mostly about data of strings or images. It is 2025 and many are using preprocessed features from the raw data, so the inputs may be just a 512 or 768 dim vector. Instructions on uploading these types of data are not that clear so I write to record my own experience. My final uploaded dataset can be found here: https://huggingface.co/datasets/stonezh/PairedMNIST


# Local File Structure

Let us assume the local file is organized in this way:

📁 test  
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label1.npy  
│   └── ...  
├── 📄 metadata.jsonl  
├── 📁 second_img 
    ├── img0.npy  
    ├── img1.npy  
    └── ...   
📁 train  
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label1.npy  
│   └── ...  
├── 📄 metadata.jsonl  
├── 📁 second_img 
    ├── img0.npy  
    ├── img1.npy  
    └── ...  
📁 val 
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label1.npy  
│   └── ...  
├── 📄 metadata.jsonl  
├── 📁 second_img 
    ├── img0.npy  
    ├── img1.npy  
    └── ...  

This is the backbone structure of ImageFolder. The data itself is basically just MNIST with two images, and label being the multiplication of the two figures in the two images, but instead of assuming the image is an image and the label is a label, I d rather just treat them as an 1D vector, as if they are preprocessed latent features. So each imgX.npy is just a 784 dim vector, and each label is just a 1 dim vector.


# Step 1: Create Local Metafile

Just write a local python script to scan the files and get a meta file. I recommend using `jsonlines` package to save the metafile as `.jsonl` file. Suppose we call the metafile as `metafile.jsonl` (name does not matter you can name it anything you like), it should look like this:

```json
{"first_img": "first_img/img0.npy", "second_img": "second_img/img0.npy", "label": "label/label0.npy", "description": "this is the 0th sample"}
{"first_img": "first_img/img1.npy", "second_img": "second_img/img1.npy", "label": "label/label1.npy", "description": "this is the 1th sample"}
{"first_img": "first_img/img2.npy", "second_img": "second_img/img2.npy", "label": "label/label2.npy", "description": "this is the 2th sample"}
```

The folder now should look like this

📁 test  
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy   
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label0.npy   
│   └── ...  
├── 📄 metadata.jsonl  
├── 📁 second_img  
    ├── img0.npy  
    ├── img1.npy   
    └── ...  
📁 train  
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy   
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label0.npy   
│   └── ...  
├── 📄 metadata.jsonl  
├── 📁 second_img  
    ├── img0.npy  
    ├── img1.npy   
    └── ...  
📁 val
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy   
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label0.npy   
│   └── ...  
├── 📄 metadata.jsonl  
├── 📁 second_img  
    ├── img0.npy  
    ├── img1.npy   
    └── ...  


# Write Custom dataset.GeneratorBasedBuilder


Under each folder (train, val, test), put a corresponding python file named train.py, val.py and test.py respectively. For test.py it looks like

```python
import pandas as pd
from huggingface_hub import hf_hub_url
import datasets
import os
import numpy as np


METADATA_URL = hf_hub_url(
    "stonezh/PairedMNIST",
    filename="metadata.jsonl",
    repo_type="dataset",
)

_DESCRIPTION = "TODO"
_HOMEPAGE = "NONE"
_LICENSE = "NONE"
_CITATION = "NONE"

# _FEATURES = datasets.Features(
#     {
#         "first_image": datasets.Sequence(datasets.Value("float64"), length=784),  # https://discuss.huggingface.co/t/setting-dataset-feature-value-as-numpy-array/20940/2
#         "second_image": datasets.Sequence(datasets.Value("float64"), length=784),
#         "label": datasets.Sequence(datasets.Value("float64"), length=1)
#     },
# )

_FEATURES = datasets.Features(
    {
        "first_image": datasets.Array2D(shape=(1, 784), dtype='float64'),  # https://discuss.huggingface.co/t/setting-dataset-feature-value-as-numpy-array/20940/2
        "second_image":  datasets.Array2D(shape=(1, 784), dtype='float64'),
        "label":  datasets.Array2D(shape=(1, 1), dtype='float64')
    },
)

# _FEATURES = datasets.Features(
#     {
#         "first_image": datasets.Image,  # https://discuss.huggingface.co/t/setting-dataset-feature-value-as-numpy-array/20940/2
#         "second_image": datasets.Image,
#         "label": datasets.Image
#     },
# )

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=datasets.Version("0.0.1"))


class PairedMNIST(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # print(dl_manager._data_dir)
        # metadata_path = os.path.join(dl_manager._data_dir, "metadata.jsonl")
        # first_img_path = os.path.join(dl_manager._data_dir, "first_img")
        # second_img_path = os.path.join(dl_manager._data_dir, "second_img")
        # label_path = os.path.join(dl_manager._data_dir, "label")

        metadata_path = os.path.join(dl_manager._data_dir, "metadata.jsonl")
        first_img_path = dl_manager._data_dir
        second_img_path = dl_manager._data_dir
        label_path = dl_manager._data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "first_img_path": first_img_path,
                    "second_img_path": second_img_path,
                    "label_path": label_path,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, first_img_path, second_img_path, label_path):
        # print(metadata_path)
        metadata = pd.read_json(metadata_path, lines=True)

        for row_idx, row in metadata.iterrows():
            cur_first_img_path = os.path.join(first_img_path, row["first_img"])
            # cur_first_img_path = row["first_img"]
            first_img = tuple(np.load(cur_first_img_path).tolist())
            # first_img = open(cur_first_img_path, "rb").read()

            cur_second_img_path = os.path.join(second_img_path, row["second_img"])
            # cur_second_img_path = row["second_img"]
            second_img = tuple(np.load(cur_second_img_path).tolist())
            # second_img = open(cur_second_img_path, "rb").read()

            cur_label_path = os.path.join(label_path, row["label"])
            # cur_label_path = row["label"]
            label = tuple([np.load(cur_label_path).item()])
            # label = open(cur_label_path, "rb").read()

            # return key and value pair
            yield row_idx, {
                "first_image": {
                    # "path": cur_first_img_path,
                    # "bytes": first_img
                    first_img
                },
                "second_image": {
                    # "path": cur_second_img_path,
                    # "bytes": second_img,
                    second_img
                },
                "label": {
                    # "path": cur_label_path,
                    # "bytes": label
                    label
                }
            }
```

In the other two files, change `name=datasets.Split.TEST` in `_split_generators` function to `name=datasets.Split.TRAIN` and `name=datasets.Split.VALIDATION`. 

Some explanation

 * in `_FEATURES` I define all the three to be `datasets.Array2D`, simply because there is no datasets.Array1D and idk why ...
 * in `_generate_examples` each np array is wrapped by `tuple(..tolist())`, because HuggingFace expect this function to return a hashable value in each dict value, so neither np.array nor list can be processed, and converting it to tuple is the only way I can think of. Many tutorials use bytes but that seems to work when you set `datasets.Image` in `_FEATURES`, and that is exactly what I try to avoid

 Now the structure should look like this

 📁 test  
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label1.npy  
│   └── ... 
├── 📄 metadata.jsonl  
├── 📁 second_img
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...   
└── 📄 test.py  
📁 train  
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label1.npy  
│   └── ... 
├── 📄 metadata.jsonl  
├── 📁 second_img
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...   
└── 📄 train.py  
📁 val 
├── 📁 first_img  
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...  
├── 📁 label  
│   ├── label0.npy  
│   ├── label1.npy  
│   └── ... 
├── 📄 metadata.jsonl  
├── 📁 second_img
│   ├── img0.npy  
│   ├── img1.npy  
│   └── ...   
└── 📄 val.py


# Upload the Data  

First some environment settings:

* `pip install huggingface_hub`
* `pip install datasets`


Then follow the instruction here to login:

https://huggingface.co/docs/datasets/en/upload_dataset

using `huggingface-cli login` and your access tokens.


Then, suppose all the three folders are in a folder named `paired_mnist`. In this `paired_mnist` folder run the following script:

```Python
from datasets import load_dataset
from datasets import DatasetDict

ddict = DatasetDict({"train": load_dataset("train", data_dir="train")["train"], "val": load_dataset("val", data_dir="val")["validation"], "test": load_dataset("test", data_dir="test")["test"]})

ddict.push_to_hub("stonezh/PairedMNIST")  # <- I am using my online repo id here
```





