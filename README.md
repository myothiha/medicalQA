# NLP Project - Medical Chatbot (AIT - DSAI)

- [Team Members](#team-members)
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Solution Requirements](#solution-requirements)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
  - [Classifier Evaluation](#classifier-evaluation)
    - [Triage Model](#triage-model)
    - [Relevant Model](#relevant-model)
  - [Chatbot Evaluation Methodology](#chatbot-evaluation-methodology)
  - [Precision and Comparison with GPT-2](#precision-and-comparison-with-gpt-2)
  - [Results](#results)
    - [Score Distribution](#score-distribution)
- [Discussions](#discussions)



## Team Members
- Myo Thiha (st123783)
- Rakshya Lama Moktan (st124088)
- Kaung Htet Cho (st124092)

## Introduction

In the rapidly evolving landscape of healthcare, accessibility to reliable medical advice remains a persistent challenge for many individuals worldwide. Cost barriers, the complexity of medical terminology, and limited accessibility to healthcare professionals underscore the critical need for innovative solutions that democratize access to healthcare information. Addressing this pressing issue requires a multifaceted approach that leverages cutting-edge technology to bridge the gap between individuals and trustworthy medical guidance.

To tackle these challenges head-on, our project aims to develop a user-friendly medical chatbot capable of providing convenient, affordable, and easily understandable health advice to users irrespective of their location or financial status. By integrating natural language processing (NLP) techniques, advanced information retrieval methods, and robust database organization, our solution seeks to enhance accessibility and simplify the comprehension of complex medical information. Through the deployment of state-of-the-art models, including a classifier for query relevance determination and a large language model fine-tuned with patient question-answer data, our system endeavors to provide accurate and contextually relevant responses to a wide array of health inquiries.

The expected results of our project encompass a transformative shift in how individuals access healthcare information. We anticipate a significant improvement in the efficiency and efficacy of medical advice delivery, with users benefitting from timely and accurate responses to their health queries. By democratizing access to healthcare information, our solution aims to empower individuals to make informed decisions about their well-being, thereby alleviating the burden on healthcare professionals and improving overall healthcare system efficiency.

The contribution of our project extends beyond the development of a standalone medical chatbot. Through rigorous experimentation, evaluation, and refinement, we aim to establish a robust framework for delivering reliable healthcare information through AI-powered systems. By conducting comparative analyses, assessing user feedback, and exploring avenues for improvement, our project seeks to set a benchmark for the development of future healthcare dialogue systems. Ultimately, our goal is to foster a paradigm shift in healthcare delivery, where technology serves as a catalyst for enhancing accessibility, efficiency, and quality of care.

## Problem Statement

The current landscape of healthcare presents numerous hurdles for individuals seeking guidance on their health concerns. Many people face difficulties in obtaining trustworthy medical advice due to various reasons:

- **Cost Barriers:** Traditional methods of seeking medical advice often involve significant expenses, discouraging individuals from seeking timely help for their health queries.
- **Complexity of Medical Terminology:** The abundance of technical jargon in medical literature and consultations poses a barrier for individuals with limited medical knowledge, hindering their understanding of their health conditions and treatment options.
- **Limited Accessibility:** Geographical constraints, long waiting times for appointments, and limited availability of healthcare professionals further exacerbate the challenge of accessing timely medical advice.

In light of these issues, there exists a pressing need for a solution that offers convenient, affordable, and easily understandable health advice to all individuals, irrespective of their location or financial status.

## Motivation

Our endeavor to address the aforementioned challenges is fueled by a vision to revolutionize the way individuals access healthcare information. By leveraging cutting-edge technology, we aim to develop a user-friendly system that facilitates seamless communication between users and a knowledgeable virtual assistant capable of addressing a wide array of health concerns.

Our motivation stems from a desire to:
- **Enhance Accessibility:** By creating a platform that individuals can access anytime, anywhere, we aim to democratize access to healthcare information, ensuring that everyone can make informed decisions about their well-being.
- **Simplify Complexity:** We are committed to developing a chatbot that can decipher complex medical terminology and convey information in a clear and understandable manner, empowering users to take control of their health.
- **Alleviate Healthcare Burden:** By providing accurate responses to common health inquiries, our project aims to alleviate the burden on healthcare professionals, allowing them to focus on more complex cases and improving overall efficiency within the healthcare system.

## Solution Requirement

- **Natural Language Processing Deployment:** Use NLP to understand and respond to patient queries efficiently.
- **Information Retrieval Enhancements:** Increase the speed and accuracy of information retrieval.
- **Database Organization:** Preprocess and organize the database into categorized bins or vector repositories.
- **Classifier Utilization:** Use a classifier to determine whether a query is emergency or relevant before processing it to improve model performance and provide only relevant information.
- **Prompt Template Creation:** Develop efficient prompt templates to guide user interactions smoothly.
- **Language Model Adjustment:** Fine-tune the Large Language Model (LLM) with a Patient QA dataset to improve its response quality.
- **Experimental Analysis:** Conduct experiments to identify the best models for the classifier and LLM.
- **Feature Evaluation through Ablation Study:** Perform an ablation study to assess the impact of various features on system performance.
- **Web Application Development:** Create a web-based chat interface for patients, medical researchers, and doctors, facilitating easier access and communication.

## Related Work

### [The AI Doctor Is In: A Survey of Task-Oriented Dialogue Systems for Healthcare Applications](https://aclanthology.org/2022.acl-long.458/)
- **Authors:** Mina Valizadeh, Natalie Parde
- **Citations:** 34
- **Year:** 2022
- **Summary:** The paper explores how task-oriented dialogue systems are used in healthcare, focusing on how these systems are built, managed, and evaluated. It describes two main approaches to designing these systems: the pipeline approach and the end-to-end approach. Evaluation methods for these systems are also examined, including both human feedback and automated measurements.

### [MIE: A Medical Information Extractor towards Medical Dialogues](https://aclanthology.org/2020.acl-main.576/)
- **Authors:** Yuanzhe Zhang, Zhongtao Jiang, Tao Zhang, Shiwan Liu, Jiarun Cao, Kang Liu, Shengping Liu, Jun Zhao
- **Citations:** 41
- **Year:** 2020
- **Summary:** MIE is a deep matching model specifically tailored for extracting medical information from doctor-patient dialogues. It addresses the unique challenges posed by medical dialogue interactions through various modules like Annotation Module, Encoder Module, Matching Module, and Aggregate Module.

### [A Multi-Persona Chatbot for Hotline Counselor Training](https://aclanthology.org/2020.findings-emnlp.324/)
- **Authors:** Orianna DeMasi, Yu Li, Zhou Yu
- **Citations:** 14
- **Year:** 2020
- **Summary:** The paper proposes developing "Crisisbot", a chatbot to simulate hotline visitors with different personas to help train human counselors. It utilizes a multi-task training framework to construct persona-relevant responses by mimicking example conversations.

### [PlugMed: Improving Specificity in Patient-Centered Medical Dialogue Generation using In-Context Learning](https://aclanthology.org/2023.findings-emnlp.336.pdf)
- **Authors:** Chengfeng Dou, Zhi JinB, Wenpin JiaoB, Haiyan Zhao, Yongqiang Zhao, Zhenwei Tao
- **Citations:** 2
- **Year:** 2023
- **Summary:** PlugMed is a plug-and-play medical dialogue system that aims to improve the specificity of responses from large language models (LLMs) using in-context learning. It utilizes a Prompt Generation (PG) module and a Response Ranking (RR) module to guide LLMs to follow a doctor-like dialogue strategy.

### [Task-oriented Dialogue System for Automatic Diagnosis](https://aclanthology.org/P18-2033/)
- **Authors:** Zhongyu Wei, Qianlong Liu, Baolin Peng, Huaixiao Tou, Ting Chen, Xuanjing Huang, Kam-fai Wong, Xiangying Dai
- **Citations:** 196
- **Year:** 2018
- **Summary:** The paper introduces an innovative dialogue system aimed at automating medical diagnoses. It establishes a unique dataset from an online medical forum, focusing on enhancing disease identification accuracy through the collection of additional symptoms during patient interactions.

### [Building blocks of a task-oriented dialogue system in the healthcare domain](https://biblio.ugent.be/publication/8723886/file/8723887)
- **Authors:** Heereen Shim, Dietwig Lowet, Stijn Luca, Bart Vanrumste
- **Citations:** 4
- **Year:** 2021
- **Summary:** The paper outlines a comprehensive approach to developing healthcare dialogue systems, addressing the unique challenges of this field. It introduces a novel framework that incorporates privacy-preserving data collection, grounding dialogue management in medical knowledge, and focusing on human-centric evaluations.

### [MedGPTEval: A Dataset and Benchmark to Evaluate Responses of Large Language Models in Medicine](https://arxiv.org/abs/2305.07340)
- **Authors:** Jie Xu, Lu Lu, Sen Yang, Bilin Liang, Xinwei Peng, Jiali Pang, Jinru Ding, Xiaoming Shi, Lingrui Yang, Huan Song, Kang Li, Xin Sun, Shaoting Zhang
- **Citations:** 14
- **Year:** 2023
- **Summary:** MedGPTEval provides a dedicated dataset and benchmark for evaluating large language models in the field of medicine. It offers a comprehensive set of medical questions and statements, enabling researchers and developers to assess the performance of language models in understanding and generating relevant and accurate responses to medical queries.

## Methodology
![image](https://github.com/myothiha/medicalQA/assets/45217500/d999ae9b-9dec-422c-b83d-fdb7ae2bff8c)

## Dataset

To construct a comprehensive and relevant dataset for our medical chatbot, we amalgamated various question-and-answer datasets along with medical datasets. This fusion resulted in a dataset comprising 42,513 rows of data. Furthermore, a specialized medical dataset was preprocessed to create a triage dataset, containing 583,613 rows. This extensive dataset forms the backbone of our chatbot's knowledge base, enabling it to provide accurate and informative responses to users' inquiries.

## Models

Our project utilizes three distinct models, each serving a specific purpose in enhancing the functionality and efficacy of the medical chatbot:

- **Relevancy Classification Model:** Prior to inputting a query into the chatbot system, it undergoes classification to determine its relevance. We employ a BART classifier that has been fine-tuned to discern whether a given query is pertinent or not. This step ensures that only relevant queries are passed through the subsequent stages of the chatbot's response generation process, optimizing the efficiency of the system.

- **Triage Classification Model:** For cases involving potential medical emergencies, a dedicated triage classification model is employed. Leveraging a BART classifier, which has been fine-tuned using the triage dataset obtained from OpenAI, this model swiftly identifies instances that require urgent attention. By promptly flagging such cases, the chatbot can prioritize responses and guide users toward appropriate medical assistance in critical situations.

- **Long Chain Model:** The centerpiece of our medical chatbot is a finely-tuned T5 model (after comparing with various models like DialoGPT, BART, ClinicalBERT, BIOMistral, Bio-BERT), structured in a Retrieval-Augmented Generation (RAG) format. This model is designed to handle a wide spectrum of medical inquiries by utilizing a vector database composed of extensive medical textbooks. By incorporating this rich knowledge base, the chatbot can generate highly informed and contextually relevant responses to diverse medical queries. The long-chain model ensures that users receive comprehensive and accurate information, contributing to an enhanced user experience and satisfaction.

By integrating these three models into our medical chatbot framework, we have developed a sophisticated and versatile system capable of addressing various health-related concerns with efficiency and precision.

## Evaluation

### Classifier Evaluation

#### Triage Model

The Triage Model exhibited commendable performance across multiple evaluation metrics:
![Triage Model Evaluation](https://github.com/myothiha/medicalQA/assets/45217500/09dda4ac-5861-4f0e-a353-095389f9339c)


- Precision: The precision of the Triage Model was calculated to be 0.82, indicating a high degree of accuracy in correctly identifying cases of medical emergency.
- Recall: With a recall score of 0.88, the Triage Model demonstrated its effectiveness in capturing a significant portion of true positive instances, thereby minimizing the likelihood of missing critical cases.
- F1-score: The F1-score, a harmonic mean of precision and recall, was calculated to be 0.85, reflecting a balanced performance in terms of both precision and recall.
- Accuracy: The Triage Model achieved a commendable accuracy rate of 0.77, highlighting its ability to correctly classify cases of medical emergency with a high degree of reliability.

### Relevant Model Evaluation
The evaluation of the Relevant Model yielded remarkable results, albeit with a caveat:
for markdown


![Relevant Model Evaluation](https://github.com/myothiha/medicalQA/assets/45217500/8e6122bd-57a7-4433-ab66-a0e6ee672031)

- Precision, Recall, F1-score, and Accuracy: The Relevant Model achieved perfect scores of 100% in precision, recall, F1-score, and accuracy. While these results suggest an outstanding performance, the possibility of overfitting cannot be discounted. Overfitting occurs when a model excessively adapts to the training data, resulting in poor generalization to unseen data.

n summary, while the Relevant Model demonstrated impeccable performance on the evaluation metrics, further analysis is warranted to ensure its robustness and generalizability beyond the training dataset. Strategies such as cross-validation and exploring alternative evaluation methods may help mitigate the risk of overfitting and provide a more comprehensive assessment of the model's capabilities.

## Chatbot Evaluation Methodology:
The evaluation of our medical chatbot primarily relied on the Delphi method, a form of expert evaluation, supplemented by precision metrics and a comparative analysis against GPT-2 models trained on medical exams.

## Delphi Method (Expert Evaluation):
Medical experts, namely doctors, played a pivotal role in evaluating the performance of our chatbot. The evaluation process involved the following steps:
Questionnaire Distribution: Doctors were provided with a structured questionnaire containing model answers generated by the chatbot in response to various medical queries.

- Criteria Evaluation: Experts were asked to assess the chatbot's performance based on predefined criteria, including:
- Accuracy: Doctors rated the correctness of the model answers on a binary scale (0: not correct, 1: correct), reflecting the accuracy of the information provided.
- Logic: Evaluation of the chatbot's ability to understand and handle medical jargon, with ratings ranging from 0 (no medical jargon understanding) to 3 (excellent medical knowledge and logic presented).
- Informativeness: Assessment of the depth and completeness of the provided answers, categorized into three levels: 0 (lacks proper information), 1 (missing key points), and 2 (sufficient information).
- Comprehension: Rating the clarity and understandability of the chatbot's responses on a binary scale (0: difficult to understand, 1: understandable).
- Tone: Evaluation of the overall tone and language used by the chatbot, distinguishing between bad (0) and good (1) communication.
- Repeated Answer: Identification of duplicated responses to gauge the chatbot's ability to provide diverse and unique answers.
- Expert Feedback: Based on their assessment, medical experts provided qualitative feedback and suggestions for improving the chatbot's performance, helping to refine and optimize its capabilities.

![Marking Criteria of Human Evaluation](https://github.com/myothiha/medicalQA/assets/45217500/eebef702-34f4-4a7d-9b0a-94b3097ec96c)


## Precision and Comparison with GPT-2:
In addition to expert evaluation, precision metrics were utilized to quantify the accuracy and effectiveness of the chatbot's responses. Furthermore, a comparative analysis was conducted to benchmark our chatbot against GPT-2 models trained specifically on medical exam datasets by using cosine similarity. By comparing precision scores and qualitative assessments, we gained insights into the chatbot's performance relative to existing state-of-the-art models.

![Precision of model with medical exam question](https://github.com/myothiha/medicalQA/assets/45217500/d9afd672-93e4-45f1-af0a-c0f9ba0ffd40)


The precision score of 0.28 indicates that the proportion of relevant instances among the total instances retrieved by the chatbot is relatively low. In other words, out of all the responses provided by the chatbot, only approximately 28% were deemed accurate and relevant according to the evaluation criteria.

While precision is an important metric for assessing the quality of the chatbot's responses, it's crucial to consider it in conjunction with other evaluation metrics such as recall, accuracy, and expert judgments. A precision score of 0.28 suggests that there may be room for improvement in the chatbot's ability to provide accurate and informative responses to users' medical queries.

Analyzing the factors contributing to the low precision score, such as the complexity of medical terminology, the diversity of user queries, and the adequacy of the chatbot's training data, can inform strategies for enhancing the chatbot's performance. Iterative refinement, incorporating feedback from medical experts, and fine-tuning the model based on real-world usage can help improve precision and overall effectiveness in addressing users' medical concerns.

By employing a multi-faceted evaluation approach encompassing expert judgment, precision metrics, and comparative analysis, we obtained a comprehensive understanding of our medical chatbot's strengths, weaknesses, and areas for improvement. This iterative process enables us to continuously enhance the chatbot's capabilities and deliver optimal performance in addressing users' medical queries.

## RESULTS
Dr. Garima Thakur, Dr. Abhigya Paudeyal. Dr Sudeshna Shrestha

| Accuracy | Logic | Informativeness | Comprehension | Tone | Repeated Answer | Total Score (%) |
|----------|-------|-----------------|---------------|------|-----------------|-----------------|
| 1        | 2     | 1               | 1             | 1    | 1               | 70              |
| 0        | 2     | 1               | 0             | 0    | 1               | 40              |
| 0        | 1     | 1               | 1             | 1    | 1               | 50              |
| 0        | 3     | 2               | 0             | 1    | 0               | 60              |
| 1        | 2     | 2               | 0             | 1    | 0               | 60              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 1     | 2               | 0             | 1    | 1               | 60              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 1        | 0     | 1               | 1             | 0    | 1               | 40              |
| 1        | 1     | 1               | 0             | 1    | 1               | 50              |
| 1        | 0     | 1               | 1             | 0    | 0               | 30              |
| 1        | 2     | 2               | 1             | 1    | 0               | 70              |
| 0        | 0     | 0               | 0             | 0    | 0               | 0               |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 0        | 1     | 1               | 0             | 0    | 0               | 20              |
| 0        | 0     | 0               | 1             | 0    | 0               | 10              |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 0        | 1     | 1               | 0             | 0    | 1               | 30              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 0        | 0     | 0               | 1             | 1    | 1               | 30              |
| 1        | 1     | 1               | 0             | 1    | 1               | 50              |
| 0        | 0     | 0               | 0             | 1    | 0               | 10              |
| 1        | 1     | 0               | 1             | 1    | 0               | 40              |
| 1        | 1     | 1               | 1             | 1    | 0               | 50              |
| 1        | 1     | 2               | 1             | 1    | 1               | 70              |
| 1        | 1     | 2               | 1             | 1    | 1               | 70              |
| 1        | 1     |                 |               |      |                 | 20               |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 2     | 1               | 0             | 1    | 1               | 60              |
| 1        | 2     | 1               | 1             | 0    | 1               | 60              |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 0        | 1     | 1               | 0             | 1    | 1               | 40              |
| 1        | 2     | 2               | 0             | 1    | 1               | 70              |
| 1        | 2     | 2               | 1             | 1    | 1               | 80              |
| 1        | 2     | 2               | 1             | 1    | 1               | 80              |
| 0        | 0     | 1               | 1             | 1    | 1               | 40              |
| 0        | 0     | 1               | 1             | 1    | 1               | 40              |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 0        | 1     | 0               | 1             | 1    | 1               | 40              |
| 0        | 1     | 1               | 0             | 1    | 1               | 40              |
| 1        | 2     | 1               | 1             | 1    | 1               | 70              |
| 1        | 1     | 1               | 1             | 1    | 1               | 60              |
| 1        | 1     | 0               | 1             | 1    | 1               | 50              |
| 1        | 2     | 0               | 1             | 1    | 1               | 60              |
| 1        | 2     | 1               | 1             | 1    | 1               | 70              |


Total accuracy: 77%

The provided data consists of evaluations conducted by Dr. Garima Thakur on various aspects of a system or model, likely related to a medical chatbot. Each evaluation entry includes ratings on six different criteria:
- Accuracy: Indicates whether the response provided by the system is correct (1) or not correct (0).
- Logic: Assesses the system's understanding and handling of medical jargon, ranging from 0 (no understanding) to 3 (excellent knowledge and logic).
- Informativeness: Rates the depth and completeness of the provided answers on a scale from 0 to 2.
- Comprehension: Measures the clarity and understandability of the responses, with 0 indicating difficulty in understanding and 1 indicating understandability.
- Tone: Evaluate the overall tone and language used in the responses, categorized as bad (0) or good (1).
- Repeated Answer: Indicates whether the response is a duplicate (0) or not (1).
Each evaluation entry also includes a "Total Score" that aggregates the ratings across the six criteria.

To summarize the provided data:

Total number of evaluations: 90

Total number of correct responses: 69

Total number of incorrect responses: 21

Total accuracy: 77%


### Score Distribution

The bar chart visualizes the evaluation criteria scores for a particular system or model across various criteria. Each criterion is represented on the x-axis, including "Accuracy," "Logic," "Informativeness," "Comprehension," "Tone," and "Repeated Answer." The corresponding scores for each criterion are depicted on the y-axis, ranging from 0 to 1.
From the graph, we can observe the following:

![Score Distribution](https://github.com/myothiha/medicalQA/assets/45217500/f3d12591-dcca-4172-b8da-bec38c43703f)


- Accuracy: The system or model achieves a score of approximately 0.72, indicating a relatively high level of accuracy in providing correct responses.
- Logic: The logic score is around 0.39, suggesting that there is room for improvement in the system's understanding and handling of medical jargon and logical reasoning.
- Informativeness: With a score of approximately 0.36, the system's responses may lack depth and completeness in providing information.
- Comprehension: The comprehension score is about 0.53, indicating a moderate level of clarity and understandability in the system's responses.
- Tone: The system receives a high score of approximately 0.82 for tone, suggesting that it effectively communicates in a suitable and respectful manner.
- Repeated Answer: The score for avoiding repeated answers is around 0.58, indicating that the system generally provides diverse and unique responses.

Overall, while the system performs well in terms of accuracy, tone, and avoiding repeated answers, there are areas such as logic, informativeness, and comprehension that could be further enhanced to improve the overall effectiveness and user experience. This analysis provides insights into the strengths and weaknesses of the system, guiding future iterations and improvements.

![image](https://github.com/myothiha/medicalQA/assets/45217500/f7d68a3a-b609-479b-a100-21c212241c26)

The evaluation data suggests that the system or model being assessed demonstrates relatively high accuracy, logic, informativeness, comprehension, and tone in its responses, with a majority of the responses being rated positively by the doctors. However, there is room for improvement, particularly in addressing repeated answers and potentially refining the comprehensiveness of responses.

The expected output of the project will be one dynamic dashboard and prediction pages. dashboard page includes some useful visuals to help better monitor, analyze, and manage. On the prediction model page, users can see the price prediction by inputting some property characteristics as well as model quality measurements. Figure (3) shows a draft preview of the project’s result.

## Discussions

The project faces several significant challenges that impact its feasibility, implementation, and evaluation. These challenges include:
- Resource Constraints for Model Loading: The inability to load all three models due to GPU limitations severely hampers the functionality and effectiveness of the medical chatbot. This limitation restricts the complexity and size of models that can be utilized, potentially compromising the chatbot's ability to generate accurate responses to users' queries.
- Parameter Availability for Models: The absence of certain model parameters due to resource constraints further exacerbates the limitations of the project. Missing parameters can adversely affect the performance and reliability of the models, leading to suboptimal outcomes and potentially inaccurate responses.
- Difficulty in Finding Medical Professionals for Evaluation: The scarcity of medical professionals available to evaluate the chatbot poses a significant challenge. Without expert input and validation, it becomes challenging to assess the accuracy, relevance, and appropriateness of the chatbot's responses to medical queries. This limitation undermines the credibility and reliability of the chatbot as a source of medical information.
- Limited Availability of Open-Source Datasets: The scarcity of open-source datasets poses a significant hurdle in training and evaluating the chatbot. Access to diverse and comprehensive datasets is crucial for developing robust and effective machine learning models. The absence of suitable datasets restricts the scope and quality of the chatbot's knowledge base, potentially limiting its ability to address a wide range of medical queries accurately.

Despite these challenges, several important insights and hypotheses can be gleaned from the project:
- Insights from Results: The evaluation results provide valuable insights into the performance and effectiveness of the chatbot. Metrics such as accuracy, precision, recall, and user feedback offer valuable indicators of the chatbot's strengths and weaknesses. Analyzing these results can help identify areas for improvement and optimization, guiding future iterations of the chatbot.
- Hypotheses for Improvement: Based on the observed limitations and challenges, several hypotheses for improvement can be formulated. For example, exploring alternative model architectures that require fewer computational resources may help alleviate the GPU constraint issue. Additionally, leveraging transfer learning techniques and pre-trained models can mitigate the need for extensive parameter tuning and resource-intensive training.
- Limitations and Challenges: It's essential to acknowledge and address the limitations and challenges inherent in the project. Lack of access to sufficient computational resources, expertise, and data can hinder the project's progress and impact its overall success. Recognizing these limitations can inform strategic decisions and resource allocation to overcome obstacles and enhance project outcomes.

In summary, while the project faces significant challenges, including resource constraints, model parameter availability, and data scarcity, it also presents valuable insights and hypotheses for improvement. By addressing these challenges and leveraging insights from the results, the project can navigate obstacles effectively and advance toward its goals of developing a robust and reliable medical chatbot.




