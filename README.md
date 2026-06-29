# MirrorMQA

MirrorMQA: A new benchmark dataset for mirror reasoning of MLLMs.

## Contents

- [MirrorMQA](#mirrormqa)
  - [1 Overview](#1-overview)
    - [Examples](#examples)
    - [Detail Information](#detail-information)
  - [2 Access MirrorMQA](#2-access-mirrormqa)
    - [Download Images](#download-images)
    - [Data Split](#data-split)
    - [Data Format](#data-format)
  - [3 Experiment and Evaluation](#3-experiment-and-evaluation)
    - [Experiment](#experiment)
    - [Evaluation](#evaluation)
  - [4 License](#4-license)

## 1 Overview

**MirrorMQA** is a manually annotated dataset designed for multimodal mirror reasoning in a multiple-choice question-answer format. The dataset includes **4,804 samples**, covering **8 visual categories**. To address the limitations of existing datasets, we clearly define annotation guidelines for MirrorMQA and focus on evaluating whether MLLMs can identify the correct mirrored image among multiple visual candidates.

### Examples

The following figures list several classic examples in our dataset. You can click the image and its corresponding `jsonl` file in [**_Examples_**](https://github.com/wuxinyehhh/MirrorMQA/tree/main/Examples) to view the details.

### Detail Information

The following table lists the detailed statistics of the split dataset. You can find the dataset files through the directory **_Dataset_** for more details.

Due to the fact that only redirecting to the specified file is valid in anonymous links, redirecting to the specified directory is invalid. Therefore, we use bold and italicized font to indicate the markings of all specified directories, making it easier for reviewers to search. Thank you!

## 2 Access MirrorMQA

We will make the complete dataset publicly available after the paper is accepted.

### Download Images

The image files will be released together with the complete dataset. The annotations in **_Dataset_** use relative image names or relative image paths, so the dataset can be used after placing the released images under the expected image directory.

### Data Split

As reported in the following table, MirrorMQA contains **4,804 samples**, divided into training, validation, and test sets according to an approximately **7:1:2** ratio. All split data files are in the directory **_Dataset_**.

| Split | File | #Samples |
| --- | --- | ---: |
| Train | `Dataset/train.jsonl` | 3,357 |
| Validation | `Dataset/val.jsonl` | 477 |
| Test | `Dataset/test.jsonl` | 970 |
| Total | - | 4,804 |

After manual filtering across more than ten websites, we obtain high-quality seed instances. To support subsequent manual creation, we categorize the collected images into eight visual categories, including letters, numbers, special symbols, clocks, icons, shapes, arrows, and composite images. Based on these categories, we further expand the image components of MirrorMQA, including reference images and their corresponding candidate images, through manual creation.

### Data Format

Each `jsonl` file is of the following format:

```jsonl
{"question": "Given four option images labeled 1, 2, 3 and 4, Which of the options is the correct mirror image of the question image?", "image": "1-0362.png", "options": ["A.1", "B.2", "C.3", "D.4"], "answer": "A"}
{"question": "Given four option images labeled 1, 2, 3 and 4, Which of the options is the correct mirror image of the question image?", "image": "1-0193.png", "options": ["A.1", "B.2", "C.3", "D.4"], "answer": "D"}
{"question": "Given four option images labeled 1, 2, 3 and 4, Which of the options is the correct mirror image of the question image?", "image": "1-0315.png", "options": ["A.1", "B.2", "C.3", "D.4"], "answer": "C"}
{"..."}
```

Each line is an individual data point. `image` denotes the name of the image. `question` is the manually annotated question. The four options correspond to four candidate images, and one of them is the correct mirror image of the question image. The corresponding option is used as the correct answer.

## 3 Experiment and Evaluation

### Experiment

We disclose the inference code in the directory **_Code/experiment_**, and the fine-tuning code in the directory **_Code/finetune_**.

For open-source MLLMs, you can directly execute the Python files in **_Code/experiment_** to perform inference on models before and after fine-tuning.

```text
nohup python deepseek_vl2.py > log/deepseek_vl2_exp.log 2>&1 &
nohup python instructblip.py > log/instructblip_exp.log 2>&1 &
nohup python intern.py > log/intern_exp.log 2>&1 &
nohup python janus.py > log/janus_exp.log 2>&1 &
nohup python llama.py > log/llama_exp.log 2>&1 &
nohup python llava.py > log/llava_exp.log 2>&1 &
nohup python minicpm.py > log/minicpm_exp.log 2>&1 &
nohup python mplug.py > log/mplug_exp.log 2>&1 &
nohup python phi.py > log/phi_exp.log 2>&1 &
nohup python qwen.py > log/qwen_exp.log 2>&1 &
```

For open-source MLLMs, you need to execute bash files in the directory **_Code/finetune_** to perform fine-tuning:

```text
nohup bash deepseek.sh > log/deepseek_train.log 2>&1 &
nohup bash intern3_5.sh > log/intern3_5_train.log 2>&1 &
nohup bash janus.sh > log/janus_train.log 2>&1 &
nohup bash llama.sh > log/llama_train.log 2>&1 &
nohup bash llava_ov.sh > log/llava_ov_train.log 2>&1 &
nohup bash mplug.sh > log/mplug_train.log 2>&1 &
nohup bash phi.sh > log/phi_train.log 2>&1 &
```

For Gemini-2.5-flash and GPT-5.5, you can directly execute our Python files in **_Code/close_models_** to perform zero-shot, few-shot, and text-only inference, provided that you prepare the corresponding API key.

```text
python gemini_0.py
python gemini_1.py
python gemini_2.py
python gemini_3.py
python gpt_0.py
python gpt_1.py
python gpt_2.py
python gpt_3.py
```

Gemini needs to be applied for on the [official website](https://ai.google.dev/gemini-api/docs), and GPT access needs to be purchased on the [official website](https://platform.openai.com/).

### Evaluation

You can process the results of model inference through the code we provide to calculate overall accuracy, overall precision, recall, F1 indicators, accuracy based on mirror categories, and accuracy based on rules. We integrate the calculation process into the Python files in the directory **_Code/eval_**.

## 4 License

This project is licensed under the [Apache-2.0 License](LICENSE).
