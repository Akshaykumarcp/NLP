## NLP research for data science
- Data science is a huge domain and it is easy to get lost in the domain while exploring!
- This repo is to keep track the things explored in the journey of Natural Language Processing (NLP), Natural Language Understanding (NLU), etc...

---

## NLP Research

| Model Name                  | Research Link                                                                          |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| RNN seq2seq and seq2seq with attention                                          |  [link](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)    |
| Attention Is All You Need                                          |  [Official Paper](https://arxiv.org/abs/1706.03762), [illustrated-transformer](https://jalammar.github.io/illustrated-transformer), [github huggingface](https://github.com/huggingface/transformers),    |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding                                           | [Official Paper](https://arxiv.org/abs/1810.04805), [Google](https://github.com/google-research/bert), [illustrated-bert](https://jalammar.github.io/illustrated-bert), [mccormickml](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/)
| GPT2: Language Models are Unsupervised Multitask Learners                                           | [Offical Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [github openai](https://github.com/openai/gpt-2), [better-language-models](https://openai.com/blog/better-language-models) ,  [illustrated-gpt2](https://jalammar.github.io/illustrated-gpt2)    |
| GPT3: Language Models are Few-Shot Learners                                           | [Official Paper](https://arxiv.org/pdf/2005.14165.pdf), [springboard gpt3](https://www.springboard.com/blog/data-science/machine-learning-gpt-3-open-ai), [gpt3 code explain](https://simonwillison.net/2022/Jul/9/gpt-3-explain-code)  , [gpt3 viz](https://jalammar.github.io/how-gpt3-works-visualizations-animations)   |
| GPT-NeoX                                           | [Announcement](https://blog.eleuther.ai/announcing-20b/), [Official Paper](https://arxiv.org/abs/2204.06745), [Github](https://github.com/EleutherAI/gpt-neox)  |
| GPT-J                                           | [Github](https://github.com/kingoflolz/mesh-transformer-jax), [Announcement](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/) |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach                                           | [Official Paper](https://arxiv.org/abs/1907.11692)|
| RoBERTa: A Robustly Optimized BERT Pretraining Approach                                           | [Official Paper](https://arxiv.org/abs/1907.11692)|
| DeBERTa: Decoding-enhanced BERT with Disentangled Attention                                           | [Official Paper](https://arxiv.org/abs/2006.03654), [microsoft blog](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark), [huggingface](https://huggingface.co/docs/transformers/model_doc/deberta)|
| DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing                                           | [Official Paper](https://arxiv.org/abs/2111.09543)|

---

## NLP General Concept's

| Concept                  | Link                                                                          |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| word2vec                                          |  [illustrated-word2vec](https://jalammar.github.io/illustrated-word2vec)   |
| Sample Language Models                                          |  [link](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)   |


### Sequence models

- Why sequence models
    - Examples:
        - Speech recognition - audio to text
        - Music generation
        - sentiment classification - text to sentiment
        - DNS squence analysis
        - Machine translation
        - Video activity recognition
        - NER
- Notations:
    - Vocabulary - have words and indices. example 10k of vocab
    - Represent word:
        - One hot - sparse. example 10k dim if 10k vocab
- RNN
    - why not standard NN?
        - cons:
            - input and output can be different lengths in different examples
            - doesn't share features learned across different positions of text
            - large input based on vocab
    - How RNN works?
        - take first word and feed to NN layer
        - predict output
        - read second word
        - get input from previous timestep and predict next word
        - same, continues
        - at each timestep, pass activation on next timestep
        - set of params are shared accross timestep
        - tanh/relu is used widely

    - different types:
        - many to many
            - Example: NER
            - Example: machine Translation
                - Alternate NN arch
                    - First read - encoder
                    - Second decode - decoder
        - many to one
            - Example: sentiment classification. input - text. output - 0/1
        - one to one
            - Example: standard NN
        - one to many
            - Example: music generation
    - Language model and sequence generation
        - given sentence, provide probability
        - build
            - training set: large corpus of english text
            - tokenize: form vocab and map to indices/vector
            - Model
                - NN arch
                - Cost function
            - Sampling novel sequences
                - word level
                - character level
    - Cons:
        - Predicts based on previous occured word. doesn't look at future words
            - Example: For NER, need to know future word for identifying a person name
            - fixed in Bi-directional RNN
        - vanishing gradients
            - long range dependencies
            - Fix for VG and LRD: Gated Recurrent Unit (GRU)
        - influenced by nearer timestep sequences

- GRU
    - introduced memory cell
- LSTM
- Bi-directional NN

### Word representation
- using vocab. ex: 10k
- Different ways:
    - One hot
        - cons:
            - doesn't generalize
                - Example: orange and apple are juice. doesn't generalize
            - sparse
    - word embedding
        - featurized representations
        - generalize
        - transfer learning and word embeddings
            - learn word embeddings from large text corpus / download pre-trained embedding
            - transfer embedding to new task with smaller training set
            - optional: continue to finetune the word embeddings with new data
                - usually do when biggger dataset
- properties of word embeddings:
    - analogy reasoning. man-> women. king->queen. Both map have similar embedding values
    - used cosine similarity widely
- Embedding matrix (EM)
    - assume 10k vocab, 300dim
    - 300 by 10k matrix
- learn word embeddings
    - words --> indicies --> one hot encoding --> EM and one hot multiplication --> get embeddings --> provide to NN --> softmax having 10k final layer --> predict
    - backprop --> GD --> predict next word
    - learn matrix E
    - other context / target pairs
        - context:
            - last 4 words
            - previous and next 4 words
            - last 1 word
            - nearby 1 word
- word2vec
    - skip grams
        - context - word - dataset
        - model
            - vocab: 10k
            - one hot - E - embedding - softmax - predict - cost function
        - cons:
            - problem with softmax classification
                - evaluate prob of vocab size. sum involved. computation more
                - fix: hirarchial softmax
    - negative sampling
        - similar to skip gram but more efficient
        - define new learning problem
        - context - word - target - dataset
        - Model
            - logistic regression
            - instead of 10k softmax calculation, here k+1 log regression model is trained
        - select negative examples:
            - sample based on how often words occur
            - uniform
            - heuristic values between above
- glove (global vectors for word representation)
    - model
        - minimize [formula]
- Note on featurization
    - no interpretation
    - analogy works
- debiasing word embeddings
    - can reflect gender, ethnicity, age, sexual orientaion and other biases of text used to train the model
    - Address:
        - Identify bias direction
        - neutralize
        - equalize pairs

### sequence models and attention mechanism
- basic models
    - seq-to-seq model
        - translation task
            - encoder outputs vector
            - decoder outputs translation one at a time
        - image captioning
            - image - convnet (alexnet) - feature vector - encoder
            - feed to rnn for caption generation
    - pick most likely sentence
        - machine translation as building a conditional LM
            - decoder section as a LM
            - encoder as a conditional LM
        - dont want to predict random translations
            - fix: beam search
                - why not greedy search?
                    - most probable word is picked
                        - could be common word such as going, the, etc
    - beam search algorithm
        - considers multiple alternatives
        - example b = 3
        - lenth normalization
- error analysis
- Bleu score evaluation

- Attention model
    - problem of long sequences
        - bleu score reduces as len of sentence increases
    - intuition
    - model

### Transformer Network
- RNN, GRU, LSTM
    - complexity increased
    - process 1 seq at a time
- self attention
- multi-head attention
- transformer n/w

