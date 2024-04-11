# NLP Project - Medial QA (AIT - DSAI)

- [Team Members](#team-members)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Solution Requirements](#solution-requirements)
- [Proposed Model Architectures](#proposed-model-architectures)
- [Experiment Design](#experiment-design)
- [Task Distribution](#task-distribution)
- [Paper Summaries](#paper-Summaries)


## Team Members
- Myo Thiha (st123783)
- Rakshya Lama Moktan (st124088)
- Kaung Htet Cho (st124092)

## Problem Statement
Many people have a tough time finding reliable health advice. They run into problems like high costs for talking to doctors, medical terms that are hard to understand, and not being able to get the help they need when they need it. This shows there's a huge need for health advice that is quick to get, easy to understand, and available to everyone, no matter where they are.

## Motivation
We're inspired to tackle this challenge by creating a new kind of system that can talk to people and give them the health advice they're looking for instantly. Using the latest technology, we want to build a system that anyone can talk to about their health concerns — a system that understands them and offers clear, correct answers. Our project aims to make getting health advice as simple as asking a question, which could help everyone get better access to the health information they need and reduce the workload on healthcare professionals by providing answers to common health questions. This way, we're not just making health advice more accessible; we're also supporting a better, more efficient healthcare system for the future.

## Solution Requirements

- Natural Language Processing Deployment - Use NLP to understand and respond to patient queries efficiently.
- Information Retrieval Enhancements - Increase the speed and accuracy of information retrieval.
- Database Organization - Preprocess and organize the database into categorized bins or vector repositories.
- Classifier Utilization - Use a classifier to determine the appropriate vector repository for information retrieval based on user inputs.
- Prompt Template Creation - Develop efficient prompt templates to guide user interactions smoothly.
- Language Model Adjustment - Fine-tune the Large Language Model (LLM) with a Patient QA dataset to improve its response quality.
- Experimental Analysis - Conduct experiments to identify the best models for the classifier and LLM.
- Feature Evaluation through Ablation Study - Perform an ablation study to assess the impact of various features on system performance.
- Web Application Development - Create a web-based chat interface for patients, medical researchers, and doctors, facilitating easier access and communication.


## Proposed Model Architectures
Our proposed solution is centered around leveraging the capabilities of Large Language Models (LLM), with a focus on employing Long Short-Term Memory (LSTM) networks as our baseline architecture. This section outlines the models we plan to use and integrate into our system to enhance its performance in understanding and responding to medical queries.

### LSTM (Baseline)
The foundation of our model relies on LSTM networks, known for their efficiency in processing sequential data and their ability to remember long-term dependencies. LSTMs will serve as our baseline to capture the context and nuances in patient inquiries.
Transformer:
We aim to incorporate Transformer models due to their advanced attention mechanisms, which allow for better understanding and generation of responses by focusing on relevant parts of the input data.

### Sentence-BERT
To enhance the semantic understanding of patient queries, we will use Sentence-BERT, a modification of the BERT model trained for generating semantically meaningful sentence embeddings. This will improve the system's ability to match queries with the most relevant information.

### T5
The Text-to-Text Transfer Transformer (T5) model will be employed for its versatility in handling various NLP tasks with a unified text-to-text approach. This will allow our system to reformulate patient queries into actionable tasks, such as information retrieval or answering questions directly.

### Llama
We plan to explore the use of Llama models for their potential in understanding and generating human-like responses, contributing to the system's ability to interact more naturally with users.

### BioBERT / BioGPT
Given the healthcare context of our application, incorporating domain-specific models like BioBERT or BioGPT will be crucial. These models have been pre-trained on large-scale biomedical corpora, equipping our system with a deeper understanding of medical terminology and concepts.

### Conclusion
Through the strategic integration of these models, we aim to create a robust, intelligent system capable of providing accurate, personalized, and contextually relevant medical advice. Each model brings unique strengths to our architecture, from capturing long-term dependencies and understanding semantic relationships to generating human-like responses and leveraging domain-specific knowledge.

## Experiment Design

## Dataset Preparation and Preprocessing
Begin by combining the datasets from Hugging Face's "medical-qa-datasets" and Kaggle's "diagnoise-me" to form a comprehensive corpus for training and evaluation.

Medical-qa-datasets: https://huggingface.co/datasets/lavita/medical-qa-datasets
Diagnose-me: https://www.kaggle.com/datasets/dsxavier/diagnoise-me
Dialog-dataset: https://drive.google.com/drive/folders/11sglwm6-cY7gjeqlZaMxL_MDKDMLdhym

The dataset shall be transformed into a vector database and subjected to a series of preprocessing steps to distill the necessary information, which will subsequently be furnished to our Natural Language Processing (NLP) model. The system's classifier is designed to partition the vector database, enabling the model to exclusively access information from the pertinent dataset. Furthermore, the Large Language Model (LLM) will undergo fine-tuning through an ablation study to achieve the desired outcomes. Comparative analysis will be conducted between the results obtained from the LLM and those derived from the baseline model (Long Short-Term Memory, LSTM), along with additional models, to ascertain their relative efficacy.

### Evaluation

1. Intent Classification Metrics: Accuracy, Precision, Recall, and F1-score for measuring the correctness of intent classification.
2. Entity Recognition Metrics: Entity-level Precision, Recall, and F1-score for evaluating the accuracy of entity identification.
3. Response Generation Metrics: BLEU, Perplexity, and Coherence Score for assessing the quality and relevance of generated responses.
4. Dialogue Quality Metrics: Task Completion Rate, Average Turns per Dialogue, and User Satisfaction Score for evaluating the effectiveness and user satisfaction of the conversation.
5. Human Evaluation Metrics: Intelligibility, Appropriateness, and Engagement for qualitative assessment of the chatbot's responses and user experience.


## Task Distribution


| Task                        | Member          |
|-----------------------------|-----------------|
| Dataset                     | Myo Thiha       |
| Preparation and Cleaning    | Kaung Htet Cho  |
| LSTM (Baseline)             | Rakshya         |
| Transformer                 | Myo Thiha       |
| Sentence Bert               | Kaung Htet Cho  |
| T5                          | Myo Thiha       |
| Llama                       | Rakshya         |
| BioBert / Biogpt            | Kaung Htet Cho  |
| Web Application Developer   | Rakshya         |


## Paper Summaries
### The AI Doctor Is In: A Survey of Task-Oriented Dialogue Systems for Healthcare Applications
**Author:** Mina Valizadeh, Natalie Parde

**Citations:** 34

**Read by:** Rakshya Lama Moktan

**Year:** 2022

**Link:** https://aclanthology.org/2022.acl-long.458/

The paper explores how task-oriented dialogue systems are used in healthcare, focusing on how these systems are built, managed, and evaluated. It describes two main approaches to designing these systems: the pipeline approach, where different components handle different tasks like understanding language and managing dialogue, and the end-to-end approach, where a single model is trained to handle everything at once.
ialogue management, which is crucial for decision-making and overall system functionality, is discussed in terms of different strategies. Rule-based approaches follow predefined rules for interaction, intent-based approaches try to understand what the user wants and act accordingly, hybrid approaches combine rule-based and intent-based methods, and corpus-based approaches use data from real human conversations to generate responses.
The paper also looks at modality, which refers to how users interact with the system, whether through text, speech, or graphical interfaces. It discusses how the choice of modality can affect the quality of the interaction.
Evaluation methods for these systems are also examined, including both human feedback and automated measurements. Human feedback can provide subjective insights into user satisfaction, while automated measurements offer objective metrics like task completion rates and response times.
In conclusion, the paper notes that while task-oriented dialogue systems are widely used in healthcare, there's a lack of rigorous technical reviews of these systems. It aims to address this gap by providing detailed insights into their implementation and performance.

### MIE: A Medical Information Extractor towards Medical Dialogues
**Author:** Yuanzhe Zhang, Zhongtao Jiang, Tao Zhang, Shiwan Liu, Jiarun Cao, Kang Liu, Shengping Liu, Jun Zhao

**Citations:** 41

**Year:** 2020 

**Read by:** Rakshya Lama Moktan

**Link:** https://aclanthology.org/2020.acl-main.576/

MIE, or Medical Information Extractor, is a deep matching model specifically tailored for extracting medical information from doctor-patient dialogues. Constructed upon a deep matching architecture, MIE addresses the unique challenges posed by medical dialogue interactions. The model comprises four pivotal components: Annotation Module, Encoder Module, Matching Module, and Aggregate Module. Annotation is facilitated through a sliding window approach, ensuring accurate labeling of information within the dialogue. The Encoder Module utilizes Bi-LSTM with self-attention for effective encoding of dialogue turns. Attention mechanisms in the Matching Module calculate attention values towards original utterances, aiding in identifying relevant information. The Aggregate Module employs two strategies, MIE-single and MIE-multi, to handle category-item pairs within and across utterances, respectively. Additionally, MIE incorporates a Scorer Module for scoring candidate utterances based on the output of the Aggregate Module. Learning is facilitated through cross-entropy loss, utilizing a Skip-gram representation for Chinese characters and the Adam optimizer. Evaluation, though detailed evaluation methodologies are not provided, demonstrates promising results, particularly with the MIE-multi model. In conclusion, MIE emerges as a valuable tool for converting medical dialogues into Electronic Medical Records (EMRs), showcasing its efficacy in accurately extracting medical information from doctor-patient interactions.


### A Multi-Persona Chatbot for Hotline Counselor Training

**Author:** Orianna DeMasi, Yu Li, Zhou Yu

**Citations:** 14

**Year:** 2020

**Read by:** Kaung Htet Cho

**Link:** https://aclanthology.org/2020.findings-emnlp.324/

The paper proposes developing "Crisisbot", a chatbot to simulate hotline visitors with different personas to help train human counselors. The goal is to provide a realistic, low-risk practice environment.
To enable Crisisbot to simulate multiple distinct personas, the authors: a) Develop a counselor strategy annotation scheme to identify user intents in counselor messages. This includes 25 strategies grouped into functional, procedural, active listening and other classes. b) Propose a multi-task training framework that constructs persona-relevant responses by mimicking example conversations rather than using pre-defined personas.
The multi-task framework has two key components: a) A Prompt Generation Module that uses the counselor strategy annotations to retrieve relevant example exchanges from both global (full context) and local (recent utterances) views. These are used to generate prompts. b) A Response Ranking Module that uses a fine-tuned small language model to rank response candidates generated by a large language model conditioned on the example prompts. It interleaves generated and retrieved prototype sub-utterances to construct detailed responses.
Automatic evaluation shows the approach increases diversity of responses and persona-relevant sub-utterances compared to baseline models.
Human evaluation with crowdworkers and experienced counselors reveals a discrepancy - crowdworkers prefer the detailed responses despite slightly lower coherence, while counselors prefer more generic but coherent responses from baselines.
Counselor written feedback highlights the importance of the system's response variety for effective training, even if the conversation flow still needs improvement.
In summary, the key contribution is the multi-task framework leveraging counseling strategies to curate varied personas mimicking examples, evaluated with metrics for specificity. The mixed results emphasize the need to involve target users during system development.

### PlugMed: Improving Specificity in Patient-Centered Medical Dialogue Generation using In-Context Learning

**Author:** Chengfeng Dou, Zhi JinB, Wenpin JiaoB, Haiyan Zhao, Yongqiang Zhao, Zhenwei Tao

**Citations:** 2

**Year:** 2023

**Read by:** Kaung Htet Cho

**Link:** https://aclanthology.org/2023.findings-emnlp.336.pdf

PlugMed is a plug-and-play medical dialogue system that aims to improve the specificity of responses from large language models (LLMs) using in-context learning.It has two key components:
A Prompt Generation (PG) module that retrieves relevant example dialogues from both global and local views to generate prompts for the LLM. The global view considers the entire dialogue history, while the local view focuses on recent utterances and the patient's chief complaint.
A Response Ranking (RR) module that uses a fine-tuned small language model to rank and select the best response from the LLM's outputs for the different prompts.
They introduce new automatic evaluation metrics to assess specificity:
Intent accuracy measures if the model's dialogue actions match the ground truth
High-frequency medical term accuracy using a Top-N term matching approach
Experiments on three medical dialogue datasets show PlugMed improves the specificity of LLM responses in terms of generating more accurate intents and medical terminology compared to baselines.Human evaluations also confirm PlugMed generates higher quality responses that are more aligned with a doctor's diagnostic strategy.
In summary, the key innovation is using retrieved example dialogues in prompts to guide LLMs to follow a doctor-like dialogue strategy, along with automatic metrics to comprehensively evaluate the specificity of the generated responses. The multi-view retrieval and response ranking further optimize the approach.

### Task-oriented Dialogue System for Automatic Diagnosis

**Author:** Zhongyu Wei, Qianlong Liu, Baolin Peng, Huaixiao Tou, Ting Chen, Xuanjing Huang, Kam-fai Wong, Xiangying Dai

**Citations:** 196

**Year:** 2018

**Read by:** Myo Thiha

**Link:** https://aclanthology.org/P18-2033/

The paper introduces an innovative dialogue system aimed at automating medical diagnoses. It establishes a unique dataset from an online medical forum, encompassing both patients' self-reported symptoms and doctor-patient conversational data. This research marks a significant step in utilizing task-oriented dialogue systems within the healthcare sector, focusing on enhancing disease identification accuracy through the collection of additional symptoms during patient interactions.
A key achievement of this study is the creation of the first medical dataset designed for dialogue systems, segmented into explicit symptoms from self-reports and implicit symptoms derived from patient-doctor conversations. The system's framework leverages reinforcement learning, specifically a deep Q-network for dialogue management, optimizing interactions to improve diagnosis accuracy.
Experiments conducted on the dataset demonstrate the system's ability to outperform baseline models by effectively gathering more comprehensive symptom information through conversations. This results in higher success rates, better rewards, and fewer dialogue turns needed for diagnosis. The findings suggest that incorporating external medical knowledge about disease-symptom relationships could further refine the system's diagnostic capabilities.
In essence, this paper contributes significantly to automated healthcare diagnostics by developing a dialogue system capable of extracting detailed symptom data from patient interactions, thereby facilitating more accurate and efficient diagnoses. Future directions include enhancing the system through the integration of external medical knowledge, promising further advancements in automated diagnostic processes.

### Building blocks of a task-oriented dialogue system in the healthcare domain

**Author:** Heereen Shim, Dietwig Lowet, Stijn Luca, Bart Vanrumste

**Citations:** 4

**Year:** 2021

**Read by:** Myo Thiha

**Link:** https://biblio.ugent.be/publication/8723886/file/8723887

The paper outlines a comprehensive approach to developing healthcare dialogue systems, addressing the unique challenges of this field. It introduces a novel framework that incorporates three essential components crucial for the effective operation of healthcare dialogue systems: privacy-preserving data collection, grounding dialogue management in medical knowledge, and focusing on human-centric evaluations.
The framework begins by addressing data collection challenges, emphasizing the generation of simulated dialogue data through expert knowledge and crowdsourcing. This method circumvents privacy issues commonly associated with healthcare data, ensuring a rich dataset reflective of real-world healthcare dialogues without compromising individual privacy.
For dialogue management, the paper proposes an innovative model leveraging Reinforcement Learning (RL), initially trained with a user simulator and subsequently refined through interactions with real users. This adaptive learning model is designed to improve the system's ability to handle diverse queries effectively, by learning from actual user interactions.
Evaluation methods blend automatic metrics like success rate and matching rate with human-centric metrics, including usability and satisfaction from healthcare professionals and end-users. This dual approach ensures the system not only performs effectively from a technical standpoint but also meets user needs and expectations.
Overall, the paper presents a methodological advancement in healthcare dialogue systems by integrating essential components that tackle privacy concerns, enhance dialogue management with medical knowledge, and prioritize user experience. This strategy promises significant improvements in healthcare dialogue systems, offering a forward-looking blueprint for future developments.