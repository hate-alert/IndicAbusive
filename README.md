# Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages
Solving the problem of **abusive speech detection** in **8 (10 types) languages** from **14 publicly available** sources.

### New update -- :tada: :tada: all our BERT models are available [here](https://huggingface.co/Hate-speech-CNERG). Be sure to check it out :tada: :tada:.


***Please cite our paper in any published work that uses any of these resources.***

~~~bibtex
@article{das2022data,
  title={Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages},
  author={Das, Mithun and Banerjee, Somnath and Mukherjee, Animesh},
  journal={arXiv preprint arXiv:2204.12543},
  year={2022}
}
~~~

------------------------------------------
***Folder Description*** :point_left:
------------------------------------------
~~~

./Dataset   --> Coming Soon
./Codes     --> Coming Soon
~~~
## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.

------------------------------------------
***Dataset***
------------------------------------------
Check out the `Dataset ` folder to know more about how we curated the dataset for different languages.  :warning: There are few datasets which requires crawling them hence we can gurantee the retrieval of all the datapoints as tweets may get deleted.
:warning:

-----------------------------------------
***Models used for our this task***
------------------------------------------
1. **[m-BERT](http://arxiv.org/abs/1810.04805)** is pre-trained on 104 languages with the largest Wikipedia utilizing a masked language modeling (MLM) objective. It is a stack of transformer encoder layers with 12 ``attention heads," i.e., fully connected neural networks augmented with a self-attention mechanism. m-BERT is restricted in the number of tokens it can handle (512 at max). To fine-tune m-BERT, we also add a fully connected layer with the output corresponding to the CLS token in the input. This CLS token output usually holds the representation of the sentence passed to the model. The m-BERT model has been well studied in abusive speech, has already surpassed existing baselines, and stands as a state-of-the-art.

2. **[MuRIL](https://arxiv.org/abs/2103.10730)** stands for Multilingual Representations for Indian Languages and aims to improve interoperability from one language to another. This model uses a BERT base architecture pretrained from scratch utilizing the Wikipedia, Common Crawl, PMINDIA, and Dakshina corpora for 17 Indian languages and their transliterated counterparts.

	
### Links to the individual model :angel:
1. [Bengali](https://huggingface.co/Hate-speech-CNERG/bengali-abusive-MuRIL)
2. [Hindi](https://huggingface.co/Hate-speech-CNERG/hindi-abusive-MuRIL)
3. [Hindi-CodeMixed](https://huggingface.co/Hate-speech-CNERG/hindi-codemixed-abusive-MuRIL)
4. [Kannada-CodeMixed](https://huggingface.co/Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL)
5. [Malayalam-CodeMixed](https://huggingface.co/Hate-speech-CNERG/malayalam-codemixed-abusive-MuRIL)
6. [Marathi](https://huggingface.co/Hate-speech-CNERG/marathi-codemixed-abusive-MuRIL)
7. [Tamil-CodeMixed](https://huggingface.co/Hate-speech-CNERG/tamil-codemixed-abusive-MuRIL)
8. [Urdu](https://huggingface.co/Hate-speech-CNERG/urdu-abusive-MuRIL)
9. [Urdu-CodeMixed](https://huggingface.co/Hate-speech-CNERG/urdu-codemixed-abusive-MuRIL)
10.[English](https://huggingface.co/Hate-speech-CNERG/english-abusive-MuRIL)
11. [AllInOne](https://huggingface.co/Hate-speech-CNERG/indic-abusive-allInOne-MuRIL)

### For more details about our paper

Mithun Das, Somnath Banerjee, and Animesh Mukherjee. 2022. "[Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages](https://arxiv.org/pdf/2204.12543.pdf)". ACM HT'22
