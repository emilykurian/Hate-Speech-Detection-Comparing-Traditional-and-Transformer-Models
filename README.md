# Hate-Speech-Detection-Comparing-Traditional-and-Transformer-Models
Aims to provide an in depth assessment of hate speech detection models, comparing traditional machine learning methods with advanced transformer models like BERT.

The proliferation of hate speech on social media platforms has emerged as a pressing problem in our digitally interconnected society. Hate speech, which encompasses discriminatory, offensive, or harmful content targeting individuals or groups based on personal attributes, poses severe consequences, including emotional distress and real-world violence. The ease with which hate speech spreads online, driven by the amplifying power of social media, highlights the urgency of developing effective hate speech detection systems. Therefore, the central problem addressed in this project is the development of a robust and accurate hate speech detection system that can identify and mitigate the spread of hate speech on social media platforms, thereby fostering a safer and more inclusive online environment.  

The intended contribution of this project is to provide a comprehensive assessment of hate speech detection models, comparing traditional machine learning methods with advanced transformer models like BERT. Upon the project's culmination, we expect to gain an in-depth grasp of the merits and limitations inherent in traditional and transformer models when it comes to identifying hate speech. This acquired understanding will play a pivotal role in enhancing content moderation tools, ultimately contributing to the continual endeavor to establish safer digital spaces. Our overarching objective is to supply valuable insights that can guide the decision-making process in implementing hate speech detection systems, tailoring model choices to specific contexts, and adopting a proactive stance in mitigating online toxicity.

# Methodology  

* Dataset : Using a dataset provided by Davidson et al.. The Twitter API was used by Davidson et al. to compile a dataset of 84.4 million tweets.
* Will be using the python language and using python idle and google colaboratory tool for BERT as it has higher GPU requirements.
* Libraries :
    * streamlit - to see the predictions of the hate speech detection model in real-time  
    * NLTK for text processing  
    * Pandas for data handling  
    * NumPy for numerical operations  
    * Scikit-Learn for machine learning tools  
    * regular expressions (re) for text manipulation  
    * Transformers (Hugging Face Transformers): provides pre-trained models,including BERT.  
    * PyTorch or TensorFlow - need these deep learning frameworks as the backend for the Transformers library to run BERT models.  
    * Tokenizers: handle tokenization of text data  
    * Matplotlib or Seaborn for data visualization

* Required computations resources:  
 GPU(s) -  A GPU that has at least 12GB of RAM is required. The requirement is met by Google Colaboratory -on the free Tesla K80 GPU.  
  
#### Traditional Models:
* Data Preprocessing  
* Text Cleaning and Preprocessing:  
    * Convert text to lowercase, remove URLs, special characters, punctuation, numbers and stopwords  
    * Apply stemming to words.  
    * Text Vectorization  
* Model Training and Evaluation
  
#### BERT base Model:  
* Data Preprocessing and Text Cleaning  
* Tokenization with BERT Tokenizer  
* Defining Constants (max_length (maximum token length) and batch_size)   
* Mapping Examples to Dictionaries  
      * map input features (input_ids, token_type_ids, attention_mask) and labels to dictionary format  
* Encoding  
    * converting text reviews into BERT-compatible input features and labels.  
    * Input features include input IDs, token type IDs, and attention masks.  
* Preparing Train and Test Datasets  
* Model Configuration:  
    * It defines the learning rate, number of epochs, optimizer, loss function, and metric (accuracy).  
* Training and Testing the Model  
* Precision,  Recall and F1-score will be used to evaluate the performance of each model.  



