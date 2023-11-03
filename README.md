# Pattern-Recognition-Systems
Enhancing the user's learning experience - Emotion Aware Intelligent Agent - <br>
Masters of Technology in Artificial Intelligence (National University of Singapore - ISS) : Pattern Recognition Systems Practice Module

The Corporate Training market size was valued at USD 145465.91 million in 2022 and is expected to expand at a CAGR of 9.57% during the forecast period, reaching USD 251715.61 million by 2028.
-	https://www.linkedin.com/pulse/corporate-training-market-size-regional/

Online corporate training can be improved in various ways to enhance the learning experience for participants. Among the many strategies and considerations for improving online training, one that stands out as being sorely needed yet difficult to achieve is:
-	Personalised Content: 
  - Tailor the training to the individual needs and learning styles of participants. Offer options for self-paced learning and provide opportunities for learners to choose their own paths within the training.
    
-	Where Personalisation is achieved via:
  - Content Chunking: 
    - Break down content into manageable chunks or modules, making it easier for learners to digest and retain information.
  - Modifying the difficulty of the questions based on the user’s level of proficiency, which we can estimate based on the user’s emotion. 
    - For example, a user who is facing a series of questions that are far in advance of his capabilities would tend to feel frustrated, angry or perhaps sad.

The difficulty lies in the question of how any courseware system knows the needs of the learner. 

The typical method is to have the learner take a test and then based on the results of the test, determine what intervention is required to help the user. 

Instead of such reactive methods, an Intelligent Agent which modifies training content and questions proactively during the training session is likely to have higher levels of learning engagement, better learning experience and learning outcomes.

Hence, the motivation behind this proof-of-concept project, to have an Intelligent Agent assess the emotional status of the learner in real-time, alter the training content into manageable chunks and modify question difficulty such that the learner becomes more motivated to understand the objective of the course because it is personalised for them.

The Emotion Aware AI Agent adopts a chatbot interface to showcase its ability to adapt the questions it asks from random or custom content in accordance with the user’s emotional state.
![image](https://github.com/atsui888/Pattern-Recognition-Systems/assets/18540586/20e90b03-3190-40d7-a3f4-22c196273a91)

The user interacts with the Agent via a Streamlit App using text input. The Agent uses the input and decides on what tool to use. 
![image](https://github.com/atsui888/Pattern-Recognition-Systems/assets/18540586/c1d1eba9-40ec-477f-ae4b-8adc358d5f07)
The Agent will perform Emotion Classification and then take the appropriate next action.

The Emotion Aware Intelligent Agent utilises the ReAct Framework as its Reasoning Engine:
- Synergizing Reasoning and Acting in Language Models (ReAct ) - https://arxiv.org/abs/2210.03629




