### Problem Definition

Adversarial attacks are the phenomenon in which machine learning models can be tricked to make false predictions by slightly modifying the input. There is not a lot of research that happened on comparing the effect of one attack on multiple models as well as the multiple datasets. So, we have studied the effect of an adversarial attack (TextFooler) on multiple BERT-based NLP models on both IMDB and Yelp datasets and proposed an ensemble defense solution.

### Motivation
There is sample literature exploring adversarial attacks on image deep neural networks (DNN) systems for object detection systems. Comparisons and trade-offs between different proposed attack types have been made and documented regularly over the past years. However, when it comes to adversarial attack effects on BERT NLP models, comparison surveys are scarce and sparse. Our primary motivation to perform this analysis is the fact that it is an area which hasn't been thoroughly explored yet. The insights that will be gained from our study will help both the designers of adversarial attacks to strengthen their methods in the future and to identify specific weaknesses in a BERT based text classification model. We also hope to make informative deductions based on the different datasets that will be used during experimentation.

### BERT Architecture
BERT stands for Bidirectional Encoder Representations from Transformers. It is a deep learning based unsupervised language model developed by researchers at Google AI. 

<img src=images/bert_arch.png align=center>

*What is meant by bidirectional?*

Up until the conception of BERT, all models either read sentences from left to right or right to left which limited the context in which each word of the sentence was viewed. BERT is bidirectional or more precisely non-directional as it considers all surrounding words as context without being biased to any direction.


BERT uses the transformer architecture, but only the encoder part of it. BERT also has a very unique way of representing embeddings. Apart from token embeddings which are somewhat common across all NLP applications, there are also segment and position embeddings. Segment embeddings indicate which sentence the current token is part of. Position embeddings indicate the relative position of the token in the entire sequence. Like I mentioned before, BERT doesn’t use the decoder of the transformer architecture. Instead, in the text classification case for example, the classifier layer acts as a decoder. Another salient feature of BERT is that is uses the masked LM strategy. This means that 15% of the tokens are masked and BERT predicts these tokens on the basis of their surrounding unmasked tokens.

[Reference to the BERT paper](https://arxiv.org/pdf/1810.04805.pdf)

### Proposed Design

<img width="467" alt="final2" src="https://user-images.githubusercontent.com/14026267/206038861-7aa8d53a-cd1b-4338-80fb-f4a546940644.png">


The design takes into account various component which are called Models in our project. One is DataSet Model that will return different type of dataset we want to train our model. Next is the our Attack Model in which we define our TextFooler attack. We have a Transformer that loads Auto Tokenizer and Model to attack which then is passed onto the HuggingFace ModelWrapper used for this project. We then build the model on the attack and run our attacker model for final evalution.


### TextFooler
TextFooler is an adversarial attack technique which identifies the most important words in the input data and replaces them by grammatically correct synonyms. What is meant by the importance of a word in this case? The words which contribute most to the label or class of the sentence are relatively more important than others. The TextFooler attack operates in 2 steps, one is the identification of the important words and the second is replacement. The replacement process is further broken down into 3 steps. First, a synonym of the important word is sampled. This includes a ranking system to check which synonym fits best. Next, part-of-speech checking is carried out to make sure that the identified synonym fits grammatically correctly in the said sentence. Finally semantic similarity checking is performed, where the semantic structure of the adversarial example is compared to the original sentence.

Here is an example as quoted in the paper. The first sentence is classified as negative. The TextFooler model identifies that the words contrived, situations and totally might be the most contributing words to this label. It then replaces these words as shown and the classifier model ends up classifying this new sentence with a positive label.

<img src="/images/textfooler_example.png" align=center>

### IMDB Movie Reviews Dataset
The [IMDB Dataset](https://huggingface.co/datasets/imdb) has 50k movie reviews for natural language processing or Text Analytics. This dataset is for binary sentiment classification containing substantially more data than the previous bench marks datasets. It contains a set of 25k highly polar movie reviews for training and 25k for testing.

* **49,582** unique reviews

* **12,500** reviews with sentiment positive

* **12,500** reviews with sentiment negative

* **80.23 MB** Dataset Size

### Snippet of Data Samples and Labels

|  **Review**                                                                                            |  **Sentiment** |                         
|--------------------------------------------------------------------------------------------------------|----------------|
|One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The... | positive
|A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B... | positive
|Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his par... | negative
|This show was an amazing, fresh & innovative idea in the 70's when it first aired. The first 7 or 8 ... | negative

### Yelp Review Sentiment Dataset
The [Yelp Dataset](https://huggingface.co/datasets/yelp_polarity) consists of reviews from Yelp. It is extracted from Yelp Dataset challenge 2015 data. This dataset is for binary sentiment classification consisting with 598k highly polar yelp reviews out of which 560k are for training and 38k are for testing. This dataset is constructed by considering stars 0-2 as negative, and 3-4 as positive. Negative polarity is class 1 and positive polarity is class 2.

* **37,999** unique reviews

* **2,80,000** reviews with sentiment positive

* **2,80,000** reviews with sentiment negative

* **158.67 MB** Dataset Size

### Snippet of Data Samples and Labels

|  **Review**                                                                                            |  **Sentiment** |                         
|--------------------------------------------------------------------------------------------------------|----------------|
|Last summer I had an appointment to get new tires and had to wait a super long time. I also went in ... | 1
|Friendly staff, same starbucks fair you get anywhere else. Sometimes the lines can get long.            | 2
|The food is good. Unfortunately the service is very hit or miss. The main issue seems to be with the... | 1
|Even when we didn't have a car Filene's Basement was worth the bus trip to the Waterfront. I always ... | 2

### Analysis on Text fooler attack performance
Text fooler attack performance can be varied based on different parameters like similarity threshold, word embedding distance, number of perturbed words etc. As part of this project, a study on text fooler attack performance is performed under different settings like max number of perturbed words, word embedding distance, allowed similarity of adversarial sentence by cosine similarity, under pre-transformations and search methods.

### Percentage of Max words perturbed
Max words perturbed constraint basically represents a maximum allowed percentage of the perturbed words to the overall words. With increase in perturbation, there would be a significant change in the characters in the words or even the whole words which would result in misclassification by the model. As we can undrestand, model accuracy under the attack reduces. Below table shows the accuracy under attack(%) and average perturbed word(%) by tweaking Max words perturbed for IMDB dataset.

|Max_Words_Perturbed(%)|Accuracy under attack(%)|Average perturbed word(%)|
|----------------------|------------------------|-------------------------|
|       0.0001         |        60              |      0                  |
|       0.01           |        60              |      0                  |
|       0.75           |        10              |      28.86              |

### Levenshtein Distance -- Similarity between two sentences
The text fooler selects the adversarial examples based on the maximal sentence similarity only over a certain threshold. It employs levenshtein distance (number of deletions, insertions or substituions required to transform the original review to the review after attack) to select the adversarial example. With other constraints constant, we evaluated the accuracy of model by varying the levenshtein distance. Below table shows the accuracy under attack(%) and average perturbed word(%) by tweaking levenshtein distance.

|Levenshtein distance|Accuracy under attack(%)|Average perturbed word(%)|
|--------------------|------------------------|-------------------------|
| 12                 |        0              |      9.37                |
| 30                 |        40              |     1.41                |
| 50                 |        45              |     0.83                |

### Pre-transformations
Pre-transformations include inserting spaces or a character at the beginning or at the end and also deleting characters, to change the context of the sentence that can potentially lead to the misclassification by the model. We observd that such pre-transformations aid the attacking efficieny and result in decreasing the model accuracy under attack. Below table shows the accuracy under attack(%) and average perturbed word(%) before and after the pre-transformations.

|Pre-transformations|Accuracy under attack(%)|Average perturbed word(%)|
|-------------------|------------------------|-------------------------|
| Before            |        70              |     0.38                |
| After             |        20              |     1.1                 |

### Search Methods
We observed that by varying the search methods from greedy search to greedy-word-swap, the model accuracy under attack will be affected. Greedy search method chooses from a list of possible perturbations greedily. This is implemented by calling "Beam search" with beam_width set to 1 where as greedy-word-swap greedily chooses from a list of possible perturbations in the order of index, after ranking indices by importance. It uses wir method for ranking the importance. Below table shows the accuracy under attack(%) for greedy search and greedy word swap search methods.

|Search method      |Accuracy under attack(%)|
|------------------ |------------------------|
| Greedy search     |        60              |  
| Greedy word swap  |        70              | 

### Modelling of BERT Models and evaluating with TextFooler

We analyzed various Bert Models on two dataset mainly YELP Polarity and IMDB. The diagram gives a brief idea of different Bert Models used. Most of these models have been fined tuned on some particular dataset like Bert on Amazon polarity has only been trained on Amazon Polarity dataset. We tried to use such models also for our attack analysis to see how well these attack model works on models which have been trained on a particular dataset.

<img width="704" alt="final4" src="https://user-images.githubusercontent.com/14026267/206046775-7d2075f3-2f4d-459b-b91e-ba265e63560f.png">

#### Training Summary of BERT Models
Traning of these models was one of the most important part of this project and we did spend fair amount of time on that. Following are the details on how our training of these Bert Model took place:

|System Paramter            |                                                              |
|---------------------------|--------------------------------------------------------------|
| Transformer Library       |        Hugging Face                                          |  
| Training Enviroment       |        Colab/Kaggle                                          | 
| Compute Used.             |        GPU P100                                              |  
| Training HyperParameter   |        BatchSize, Epochs, Label Number, Max Sequence Length  | 
| Evalution Size            |        100                                                   |
| Attack Sample size        |        100                                                   | 

Following is the table for Training HyperParameter used:

|Training Hyper Paramter            |    Value                                                     
|-----------------------------------|--------------|
| Batch Size                        |      128     |  
| Number of Epoch                   |      3       | 
| Number of Label                   |      2       | 
| Max Sequence Length               |      64/2480 |


#### Accuracy Analysis (YELP and BERT)

#### YELP Dataset 

|Model                      |Accuracy before attack(%)|Accuracy after attack(%) |Run Time|
|---------------------------|------------------------ |-------------------------|-------------------------|
| Bert on YELP Polarity     |        92.78            |     8.00                |  4 hr 30 min            |
| RoBERTa base              |        94.27            |     6.00                |  3 hr                   |
| XLnet BERT                |        93.93            |     4.00                |  3 hr 30 min            |
| Distil RoBERTa            |        93.6             |     2.00                |  1 hr 33 min            |
| BERT on Amazon Polarity   |        93.28            |     1.00                |  3 hr                   |
| BERT uncased              |        92.9             |     0.2                 |  2 hr 30 min            |
| BERT cased                |        93.08            |     0.0                 |  3 hr                   |
| AlBERT base               |        91.00            |     0.0                 |  3 hr                   |
| Tiny BERT                 |        88.34            |     0.0                 |  17 min                 |
| BERT multilingual         |        91.59            |     0.0                 |  1 hr 49 min            |


#### IMDB Dataset 

|Model                      |Accuracy before attack(%)|Accuracy after attack(%) |Run Time|
|---------------------------|------------------------ |-------------------------|-------------------------|
| Bert on YELP Polarity     |        81.76            |     0.00                |  12 min                 |
| RoBERTa base              |        86.75            |     1.00                |  13 min                 |
| XLnet BERT                |        85.63            |     0.00                |  1 hr                   |
| Distil RoBERTa            |        94.34            |     0.00                |  1 hr 33 min            |
| BERT on Amazon Polarity   |        84.92            |     2.00                |  8 min                  |
| BERT uncased              |        82.49            |     0.2                 |  7 min                  |
| BERT cased                |        83.88            |     4.0                 |  8 min                  |
| AlBERT base               |        84.08            |     1.0                 |  15 min                 |
| Tiny BERT                 |        75.36            |     0.0                 |  1 min                  |
| BERT multilingual         |        91.94            |     5.0                 |  1 hr                   |

The above tables provides us an information on various bert models there accuracy before the attack and after attack. We also did an analysis on both training and testing run time. For YELP Dataset Bert of YELP Polarity gives the best accuracy after attack that is around 8 percent. As Bert on YELP Polarity was trained for YELP Polarity dataset only it was expected that attack model will work best on that. However, for IMDB Dataset BERT multilingual gave best accuracy after attack that is 5 percent. We also concluded that for both YELP and IMDB dataset Tiny Bert takes least amount of training run time. This is maninly because of architecture of tiny bert with less layers in the network. In terms of accuracy after attack Tiny Bert gives worst accuracy after attack for both YELP and IMBD dataset. We also looked at testing time for all these models. 


### Ensemble Model
As we understood with our evaluation from the above models and their accuracies under the attacks, it is clear that any single BERT model can't withstand or good enough to defend the attacks. It is also learnt from the text fooler paper, with perturbations even less than 20%, the accuracies of the state-of-the-art models drop below 10%. Thus, an ensemble solution will be an ideal defense strategy to tackle adversarial attacks. However, just ensembling the state-of-the-art models may also not be the best solution. Let's take a look at such cases where ensemble might not be of help:

### Easy case
Easy case is considered when the attacking is benign or not sufficient enough to misguide the models. Below samples from our experiments show that the models correctly classify the sentences, marking the attack as failed.

* Bert-base-uncased
<img src=images/bert-easy.PNG align=center>

* Alberta-base-cased
<img src=images/alberta-easy.PNG align=center>

* Roberta-base
<img src=images/roberta-easy.PNG align=center>

### Hard case
Hard case is considered when the attacking is clever enough with replacing the words and misguiding the models. However, humans can observe the overall context of the sentences being preserved and can classify the sentences correctly. In that case, the models are completely failing illustrating the hardness of the attack. Below samples illustrate such attacking examples which are misclassifed by the models:

* Bert-base-uncased
<img src=images/bert-hard.PNG align=center>

* Alberta-base-cased
<img src=images/alberta-base-hard.PNG align=center>

* Roberta-base
<img src=images/Roberta-hard.PNG align=center>

Ensemble solution thus, may not help with the cases where individual classifiers can themselves defend against the attack (mild-attack) (or) where all the models collectively fail to classify correctly (harsh-attack). These cases may not be ideal real-world scenario and not targetted as part of our solution. Our defense strategy is to handle the case where few models can misclassify the adversarial example while they are rescued by other models in the ensemble for correct classification. To reflect (or) create such attack, we fine-tuned and selected the constraints for the textfooler from our experiments, ensuring that 1 in 10 words is replaced and are not discernible by humans.

| Constraint                 |     Value        | 
|----------------------------|------------------|  
| Max Words Perturbed        |     0.1          |
| Levenshtein Distance       |     12           |
| Embedding cosine similarity|     0.7          |
| Search method              | Greedy Word Swap |
| Pre-transformations        |    False         |

### Experimental Analysis for Ensemble Solution
With the above defined set of constraints, the accuracy of the models are evaluated under the attack. These models are fine-tuned on IMDB dataset and evaluated on the same. The difference in accuracies of these models before and after the attack are as follows:

| Model                   | Accuracy before attack(%)  |  Accuracy after attack(%)  | Accuracy Drop |  
|-------------------------|----------------------------|----------------------------|---------------|  
| RoBERTa base            |          92                |             88             |        4      |
| XLNet base cased        |          90                |             82             |        8      |
| BERT base uncased       |          88                |             80             |        8      |
| AlBERTa base            |          88                |             76             |        12     |
| DistilBERT base uncased |          90                |             78             |        12     |
| BERT fine tuned         |          86                |             74             |        12     |
| XLM RoBERTa             |          96                |             80             |        16     |
| BERT IMDB Hidden        |          80                |             54             |        26     |


Thus we can see that Roberta, Xlnet, Bert base uncased, and Alberta are performing well with the set of attack constraints. To extend these models to be more generic, we evaluated on the pretrained models from huggingface. The models are also evaluated on YELP dataset to be more generic. The accuracy of the models vary as below:


| Model                   | Accuracy drop on IMDB (%) |  Accuracy drop on YELP (%) |
|-------------------------|---------------------------|----------------------------|
| ALBERTa base            |          0                |             0              |
| RoBERTa cased           |          0                |             0              |
| BERT base uncased       |          4                |             8              |
| XLNet base cased        |          26               |             24             |


It can be observed that the XLNet base cased models fails with the pretrained models dropping large accuracy under the attack. The other models Alberta, Roberta, and Bert base uncased perform relatively same. Hence an ensemble solution of these models would be a good defense strategy for the textfooler attacking and adversarial examples.

A majority voting of these models would work for final decision. The advantages of the diversity of these models in ensemble solution would help to arrive at a final correct classification label. Please find the below sample where the adversarial sample is classified wrongly by Bert-base-uncased, but identified by Roberta and Alberta models proving the ensembling a good strategy.

### Ensemble Model illustraing the majority voting of the models

* Bert-base-uncased
<img src=images/bert-medium.PNG align=center>

* Alberta-base-cased
<img src=images/alberta-medium.PNG align=center>

* Roberta-base
<img src=images/roberta-medium.PNG align=center>

### Conclusion
Overall, the project has specifically targetted to study adversarial NLP examples and TextFooler attacking model. Different BERT models have been studied, experimented, and evalauted to converge an ensemble solution that provides a good defense strategy against the textfooler attack.

### References
* [Jin, D., Jin, Z., Zhou, J.T. and Szolovits, P., 2020, April. Is bert really robust? a strong baseline for natural language attack on text classification and entailment. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 05, pp. 8018-8025).](https://arxiv.org/pdf/1907.11932.pdf)
* [Morris, J.X., Lifland, E., Yoo, J.Y., Grigsby, J., Jin, D. and Qi, Y., 2020. Textattack: A framework for adversarial attacks, data augmentation, and adversarial training in nlp. arXiv preprint arXiv:2005.05909.](https://arxiv.org/pdf/2005.05909.pdf)
* [Li, L., Ma, R., Guo, Q., Xue, X. and Qiu, X., 2020. Bert-attack: Adversarial attack against bert using bert. arXiv preprint arXiv:2004.09984.](https://arxiv.org/pdf/2004.09984.pdf)
* [Zhang, X., Zhao, J. and LeCun, Y., 2015. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28.](https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf)
* [Maas, A., Daly, R.E., Pham, P.T., Huang, D., Ng, A.Y. and Potts, C., 2011, June. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies (pp. 142-150).](https://aclanthology.org/P11-1015.pdf)
* [TextAttack code repository](https://github.com/QData/TextAttack)
