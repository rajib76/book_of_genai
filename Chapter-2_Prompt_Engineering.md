## Chapter 2 - Prompt Engineering

Prompt engineering is one of the most important component in RAG technique. It is the art of designing 
the instructions(in natural language) for the language models. From our previous discussion in the class, 
remember that language models are just a probability disctribution of words.It is a statistical model of the 
data that we have fed inot it. It is a autoregressive trained model that predicts the next token.

```
remember 1000 tokens ~ 700 tokens
```

### What do we mean when we say that it is a statistical model

So, lets first understand in a very simplistic manner, how the training of the model happened. Please do not
look for technical correctness. The below steps are for conceptual understanding of what does in. For 
technical understanding, Andrej Karpathy has a great video on this. Please go through that for a detailed
understanding

[Building GPT from scratch by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)

The conceptual steps

1. First, imagine we collected lots of text from the internet, like articles, stories, and conversations. 
2. We wanted to teach a computer to understand this text, but computers only understand numbers, not words or meanings. 
3. So, we turned each piece of text into numbers. This is like giving each piece of text a unique code. 
4. These codes represent the meaning of the text. It could be about anything, like emotions or finance. 
5. Now, we have these code numbers, and they are like ingredients for understanding text. 
6. Imagine all these code numbers together make a text, like a list of ingredients for a recipe. 
7. Each number in this list tells us something important about the text's meaning. 
8. Now, we can measure how similar or different two texts are by comparing the numbers in the lists.
10. We put all the texts in different groups in a big space, like sorting them into folders. 
11. When you ask a question, the computer finds the group of texts that are most similar to your question. 
12. It guesses the next word based on what it learned from those similar texts. 
13. Sometimes, it guesses right, and sometimes it doesn't. We check if it's right or wrong. 
14. If it's wrong, we adjust some settings in the computer's brain (we call them "weights") to make it better. 
15. We keep doing this over and over until the computer gets really good at guessing. 
16. Now, we have a smart computer that can help answer questions because it learned from all those texts.

So, in the end, we have a computer that's like a language expert, trained to understand and answer 
questions using numbers and patterns it learned from lots of text.

### The importance of prompt crafting

If you have understood the basic conceots outlined above, you can probably now appreciate the importance of crafting 
the prompt for the LLM. Each word(including the position) in the text determines what it will spit out next. It
heavily depends on how it was trained(with instructions) to predict the next word or token. As users,who do not have
a lot of visibility into the training part, it becomes difficult for us to write a prompt unless we are aware of 
certain prompt patters. And that is what we are going to see next

### Prompt Patterns

There are many prompt patterns and some of the key ones are as below

#### Pattern Name: 
Meta Prompt
#### Description: 
Meta-prompts are used to provide context and specific instructions for generating text. They guide 
the model's response by framing the task or question in a particular way.
#### Example:
_Without meta prompting_
"What's your opinion on the best way to invest in stocks?"

_With meta prompting_
"What's your opinion on the best way to invest in stocks?In your response, provide an 
overview of different stock investment strategies, their risks, and potential rewards. 
Help the reader understand the key principles they should consider when deciding how to invest in stocks."

#### Pattern Name: 
Output Automator
#### Description: 
In this patter, we expect the LLM to generate the response in the form of an executable, it can be a script or an 
automaton artifact that can automatically perform the intended step
#### Example:
I want to copy a file from my local folder to AWS S3. 
Please create a python script to copy the file.

#### Pattern Name: 
Peronsa Pattern
#### Description: 
In this pattern, we give an identity or a role to the LLM while generating the output. This is done
so that the LLM can take a certain point of view or perspective.
#### Example:
You are a helpful math teacher. You help students answer math questions. Please answer the below question
What is 25*56

#### Pattern Name: 
Template Pattern
#### Description: 
In this pattern, we instruct the language model to output the response in a certain format. This may be 
required so that we stadardize the output that can be used by the next process in a deterministic way.
#### Example:
I will give you a text and you will need to extract the person name and profession from the text. You must
output the response in a JSON format.
Here is an example of the output:
{"name":name of the person,"profession":profession of the person}
<text>
Rajib is a software engineer
Answer:

#### Pattern Name: 
Recipe Pattern
#### Description: 
In this pattern, the user knows the end goal and some steps of the end goal but do not know 
the precise answer. Through the prompt, it guides the LLM to take a specific approach to answer the question
#### Example:
I am planning to go for a mountain hike. I know I need boots with good grips, need to carry water. Please
provide me a complete sequence of steps and activities that I need to do to prepare for the hike.

#### Pattern Name: 
Cognitive verifier
#### Description: 
In this pattern, the LLM asks follow up questions based on the user's original question to get more 
precise information from the user to be able to provide a sharper response
#### Example:
I will ask you a question. You need to provide me a very precise and concise answer. Before answering my question,
you ma ask me three additional question that may help you to provide a more accurate answer. Once you get the answers, 
you can combine them to present the final response

#### Pattern Name: 
Alternative Approach
#### Description: 
In this pattern, the LLM is prompted to provide alternative apporaches so that the user is aware of all the
available options to perform a task and not just the approach they are familiar with.
#### Example:
When I ask you to share the AWS service name for a particular task, please also sugegst alternate services available
along with their pros and cons.

#### Pattern Name: 
Refusal breaker
#### Description: 
In this pattern, the LLM suggest an alternate question if it cannot answer the original question based on the 
context provided.
#### Example:
Please answer based on the provided context only. If you cannot answer based on the context, please suggest an alternate question that can be answered based on the context.
<context>
India is one of the largest democracy in the world.

answer:

#### Pattern Name: 
Fact Check List
#### Description: 
In this pattern, the LLM is prompted to output a list of facts that it has used to provide the answer. This
technique is used to reduce hallucinations.
#### Example:
Please answer the question based on the context. When you answer, please also provide
a list of facts that the answer depends on that should be fact checked

#### Pattern Name: 
Flipped Interaction
#### Description: 
In this pattern, the LLM is prompted to ask follow-up questions to provide more accurate answer
#### Example:
You are a helful investment advisor. Before advising the customer, 
let's first understand the input and customer attributes. 
Please make the follow-up questions based *ONLY* on the provided customer attributes in order.
    
person attributes :{"name":"name of the person","income":"income of the person"}
    
* DO NOT make * any additional follow-up questions which does not help in filling out person attributes.

#### Pattern Name: 
Context Manager
#### Description: 
In this pattern, the LLM is instructed to add/remove context while answering the question.
#### Example:
You are a helpful chatbot, you answer questions based on provided context only.
When you answer the question, only consider the answer from documents 
which are from the year 2021 onwards

context:
date: 02/11/2020
content: Ramu works at Apple

date: 02/11/2022
content: Ramu works at facebook

{question}

Answer:


## Reference:

[A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT](https://arxiv.org/pdf/2302.11382.pdf)









