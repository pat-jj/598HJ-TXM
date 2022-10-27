## CS598 - Text Mining Assignment#1 by Patrick Jiang (pj20)

This document is to show you how to run the code to reproduce my result.

```./code``` includes the code for 1. preprocess 2. classification 3. postprocess 4. evaluation

```./data``` includes the data for final submission

```./data_process``` has the data needed for the classification task, including: 1. ```cate_results``` obtained 
by the phrase mining process in Step-2 of this assignment, 2.```original_data``` including categories and 100 
labels for each dataset (```movies``` and ```news```) provided by the assignment.

To start, you need to first go to the ```./code``` folder:

```
$ cd code
```

## Step 1: Preprocess
```
$ python preprocess.py
```

This command will build a ```data_wstc``` folder in ```../data_process```. 

For each dataset, it creates 4 files: 
1. ```labels.txt```, which is the 100-labels simply copied from ```/data_process/original_data```
2. ```classes.txt```, which indexed (from 0) the categories provided by ```/data_process/original_data```
3. ```keywords.txt```, which split the ```cate_results/res_topic``` by comma and indexed them from 0 by categories.
4. ```embedding.txt```, which is the embedding we obtained as the result of step2 (CatE) of the assignment


## Step 2: Run the WeSTClass classification code
The classification code is modified from the source code of WeSTClass [https://github.com/yumeng5/WeSTClass] 

You will need to get the same environment to run this code.

To start training & inference on a dataset, you need to run the following command:

```
$ python main.py --dataset ${dataset} --sup_source ${sup_source} --model ${model} --with_evaluation False
```

where you need to specify the dataset in ```${dataset}``` (could be either ```'movies'``` or ```'news'```), 

the weak supervision type in ```${sup_source}``` (could be one of ```['labels', 'keywords']```), 

and the type of neural model to use in ```${model}``` (could be one of ```['cnn', 'rnn']```).


For example, to run ```cnn``` with the weak supervision type ```keywords``` on dataset ```news```, you should type:

```
$ python main.py --dataset news --sup_source keywords --model cnn --with_evaluation False
```

The complete prediction output will be stored in ```../data_process/data_wstc/{dataset}/out.txt```


## Step 3: Postprocess
After obtaining the prediction output for all documents in a dataset, we move it as our final submission.
The command is as follows:

```
$ python postprocess.py --dataset ${dataset} --lines ${lines} --pred_all ${pred_all} --out ${out}
```

where ```${dataset}``` is one of ```'movies'``` or ```'news'```, ```${lines}``` is 100 by default 
(specified by the assignment requirement), ```${pred_all}``` is the file of the complete prediction output 
(```out.txt``` by default), ```${out}``` is the name of processed file (```test_prediction.txt``` by default).

For example, to get the prediction result of first 100 documents in ```'movies'``` dataset, you may run:

```
$ python postprocess.py --dataset movies
```


## Step 4: Evaluation (optional)

To test the accuracy of the prediction results with the given 100 labels, you can run the following code:

```
$ python evaluate.py --dataset ${dataset}
```

The accuracy will be printed out on the terminal.

My latest result:
```
(txm) [pj20@sunlab-serv-03 code]$ python evaluate.py --dataset news
>>> Accuracy:  0.83
>>> F1-score:  0.83

(txm) [pj20@sunlab-serv-03 code]$ python evaluate.py --dataset movies
>>> Accuracy:  0.72
>>> F1-score:  0.72
```
