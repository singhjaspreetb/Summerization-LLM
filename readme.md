

### [Get started](https://singhjaspreetb.github.io/Summerization-LLM/) 

This is a web app to **Summarize long text documents** and **Ask question** on that document. 

<br> 
Technologies used to built are: 
<br>  

- [Langchain](https://python.langchain.com/en/latest/index.html) framework<br>
- [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) Vector Database<br> 
- [Steamlit](https://docs.streamlit.io/) framework.<br>
- [OpenAi](https://platform.openai.com/docs/introduction) Api<br>

### Setup for running on local machine

Step 1:

- Open your command prompt/bash.
- Fork The Github [Repository](https://github.com/singhjaspreetb/Summerization-LLM).
- Clone the repo from your repository use Command `git clone` repo link.
<br>


Step 2:

- Install [Python](https://youtu.be/0QibxSdnWW4) on your system 
<br>

Step 3:

<br>

- Install [langchain](https://python.langchain.com/en/latest/getting_started/getting_started.html)
<br>

```
pip install langchain 
#or
conda install langchain -c conda-forge

```
<br>
Step 4:


<br>

- Install [openai](https://platform.openai.com/docs/introduction)
```
pip install openai
```
<br>
Step 5:

<br>

- Install [PyPDF2](https://pypdf2.readthedocs.io/en/3.0.0/user/installation.html)
```
pip install PyPDF2
```
<br>
Step 6:


<br>

- Install [faiss-cpu](https://faiss.ai/)
```
#installing through Conda
conda install -c pytorch faiss-cpu

```
<br>
Step 7:


<br>

- Install tiktoken
```
pip install tiktoken
```

<br>
Step 8:


<br>

- Install [streamlit](https://docs.streamlit.io/library/get-started/installation)
```
pip install streamlit
```

### Approach To the Solution
Before going to a solution, we need to understand the limitations:
<br>

•	**Token limitations:** A Chat-gpt3-like model has token limitations of 4096 to be precise.This means that they can only process a certain number of words at a time. This can make it difficult to summarize longer documents.
<br>

•	**Memory issues:** As with other language models, Chat GPT models may have difficulty retaining information over a long conversation or summarization task. This can lead to the model producing a summary that differs from the original text or misses important details.
<br>

•	**Hallucinations:** Chat GPT models have been known to "hallucinate" or generate text that is not directly related to the input. This can lead to inaccurate or misleading summaries.
<br>

• **Limited understanding:** Chat GPT models may struggle with understanding the context and nuances of the text, particularly in cases where the language is highly technical or specialized. This can lead to inaccurate or incomplete summaries.	These LLMs have large amounts of data from different fields and certain terminologies might mean something or a particular field and something else for the other.

### Solution
•	Our goal is to create a Summarization tool that takes the long text and gives concise output. But during summarization we do not want to hit the token limit, for we take the .txt file and **divide it into chunks** so that it fits in the **size of the token** and we embed these chunks with **openai embeddings**.

•	As these models have **memory issues** after embedding the chunks, we make a **semantic index of these embeddings** and store them in a **vector database** this can solve the memory issue.

•	As for **hallucination**, we set the **temperature** of the model at a **minimum** as we do not need much creativity and need only to summarize the given file .

•	For this issue I made the LLMs access to the **knowledge base** of the model only to return output and if some terminology is asked it will reply based on the knowledge base provided. 


### The architecture of the model
 
![](https://github.com/singhjaspreetb/Summerization-LLM/blob/master/Arch.png)     

### Usage
The combination of summarization and question-answering capabilities of the application can have a variety of useful applications across different fields some examples are:

- The model can be used to summarize any long documents and gives concise summary output. It can be in form of bullet points or in paragraphs.

- The model can be used as questions on the data previously stored or the recently given file which was stored in the knowledge base.

- Education: Students can use your tool to quickly summarize and comprehend long textbooks, research papers, or academic articles, and then test their understanding through the generated questions.

- Business and Finance: Financial analysts can use your tool to summarize complex reports and documents, and then quiz themselves to ensure they have a solid understanding of the material.

- Legal: Lawyers and paralegals can use your tool to summarize legal documents and then test their understanding of the content by answering relevant questions.

- Research and Academia: Researchers can use your tool to summarize and comprehend academic articles and research papers, and then quiz themselves to ensure they have fully understood the material.

- News and Media: Journalists and news analysts can use your tool to quickly summarize news articles and then generate questions to ensure they have captured the key information and insights from the piece.


### Limitations 
- Quality of the summarization: While breaking the long text into smaller chunks and embedding them with OpenAI embeddings can help with summarization, there is still a risk that the summarization output may not accurately capture the main points and nuances of the text.

- Accuracy of the question-answering: The generated questions may not always accurately capture the most relevant information from the summarization output, and the question-answering component may not always provide accurate responses. This could potentially lead to misunderstandings or incorrect information being retained by the user.

- Compatibility with different types of text: The summarization tool may not work as well with certain types of text, such as highly technical or scientific documents, which may require a deeper level of understanding than what the tool can provide.

- Resource limitations: Using OpenAI embeddings and a semantic index vector database can be resource-intensive, and may require significant processing power and memory to operate effectively. This could limit the scalability of the tool and make it less accessible to users with less powerful computing resources.

- Language limitations: LLMs may not always be able to accurately summarize or answer questions in languages other than English, which could limit the utility of the tool for non-English speakers.
- Needs more processing power for faster response on longer documents

- Can be more Optimized in terms of fine-tuning and more Domain-specific knowledge can be fed to the model.

### Futher Improvements 

- Pre-processing : Before dividing the text into chunks, useing a pre-processing step to remove unnecessary information, such as headers, footers, and boilerplate text. This can help to reduce the size of the input text and make it easier for the model to summarize.

- More Fine-tune the model: Depending on the type of text we are summarizing, we may be able to fine-tune the Chat GPT model to improve its performance. For example, if we are summarizing scientific papers, we could fine-tune the model on a large corpus of scientific texts to improve its understanding of technical language.

- Ensemble of models: Rather than relying on a single Chat GPT model, use an ensemble of models trained on different datasets or with different hyperparameters. This can help to improve the accuracy of the summaries and reduce the risk of hallucinations.

- Rateing the summaries: After generating the summaries,rateing their quality using metrics such as ROUGE (Recall-Oriented Understudy for Gisting Evaluation) or BLEU (Bilingual Evaluation Understudy). This can help to identify areas for improvement and refine the summarization algorithm over time.

