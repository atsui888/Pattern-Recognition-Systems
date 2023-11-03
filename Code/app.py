# Emotion Aware Agent - when generating questions

import requests
import os
import streamlit as st
from langchain.agents import (AgentExecutor, load_tools, tool, ZeroShotAgent)  # AgentType, initialize_agent,
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from typing import List
from dataclasses import dataclass

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 600px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.header("📝 Emotion-Aware AI")


# ***********************************************************************************************************
#             Run Docker Container for Emotion Detection using Logistic Regression
# ***********************************************************************************************************
# to start Logistic Regression Emotion Classifier docker container:
# docker run --rm -p 3000:3000 rchai/text-emotion-detection:trained_on_dair-ai_emotion
# curl -X POST -H "accept: text/plain" -H "content-type: text/plain" --data "the cake was mouthwatering delicious! Yummy!" http://127.0.0.1:3000/classifyemotion


# ***********************************************************************************************************
#                                LOAD distilBERT Emotion Classifier
# ***********************************************************************************************************
model_id = "richardchai/distilbert-emotion"
emotion_classifier_distilBERT = pipeline("text-classification", model=model_id)
distilBERT_CLASS_INT2STR = {0: 'sadness', 3: 'anger', 2: 'love', 5: 'surprise', 4: 'fear', 1: 'joy'}
distilBERT_CLASS_STR2INT = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
distilBERT_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


@dataclass
class prediction_dc:
    label_num: int
    label_name: str
    label_score: float

    def __gt__(self, other):
        return self.label_score > other.label_score

    def __eq__(self, other):
        return self.label_score == other.label_score


def distilBERT_classify_emotion(labels_d: dict, preds: list, top_k=-1, desc=True, verbose=False) -> List[prediction_dc]:
    """
    preds is a list of dict (k is 'label', v is 'score')
        format:
            'label': 'LABEL_N':str e.g. LABEL_0 or LABEL_1 ... LABEL_N
            'score': float
    if verbose: print all results
    returns a list of dicts of label_name:score, sorted from highest score to lowest score
    """
    prediction_dc_l = []
    for pred in preds:
        for i in range(len(labels_d)):
            if str(i) in pred.get('label'):
                class_num = i
                class_name = labels_d.get(class_num)
                class_score = pred.get('score')
                # print(class_num, class_name, class_score)
                prediction_dc_l.append(
                    prediction_dc(label_num=class_num, label_name=class_name, label_score=class_score))
    if desc:
        prediction_dc_l = sorted(prediction_dc_l, reverse=True)
    else:
        prediction_dc_l = sorted(prediction_dc_l, reverse=False)

    if verbose:
        for pdc in prediction_dc_l:
            print(pdc)
    if top_k == -1:
        return prediction_dc_l
    else:
        return prediction_dc_l[:top_k]


# ***********************************************************************************************************
#                                       STREAMLIT SIDEBAR
# ***********************************************************************************************************
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    CLASSIFY_EMOTION_METHOD = st.radio(
        "Select method of Emotion Classification",
        ["Logistic Regression (ML)", "DistilBERT (NN)", "GPT3.5 Turbo (LLM)"],
    )
    # if CLASSIFY_EMOTION_METHOD == "Logistic Regression (ML)":
    #     st.write('selected LR')
    # elif CLASSIFY_EMOTION_METHOD == "DistilBERT (NN)":
    #     st.write('selected DistilBERT')
    # else:
    #     st.write("GPT3.5 Turbo (LLM)")
    st.divider()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
else:
    os.environ['OPENAI_API_KEY'] = openai_api_key


# ***********************************************************************************************************
#                                       STREAMLIT RIGHT-SIDE PANE
# ***********************************************************************************************************

uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))

EMOTION = 'inactive'
QUESTION_LEVEL = 'inactive'
st.divider()
# st.caption('Emotion and Sentiment classification is active only when AI is generating a question.')
emotion_lbl = st.text(f'Emotion:\t{EMOTION.title()}')
question_lvl_lbl = st.text(f"Question Level:\t{QUESTION_LEVEL.title()}")
# st.divider()


if uploaded_file and openai_api_key:
    CUSTOM_CONTENT = uploaded_file.read().decode()
    # st.caption(CUSTOM_CONTENT)  # for debugging only
else:
    CUSTOM_CONTENT = ''

# ***********************************************************************************************************
#                                      LANGCHAIN AGENT and TOOLS Definition
# ***********************************************************************************************************
LLM = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)

# tools = load_tools(["ddg-search"])
# agent = initialize_agent(
#     tools, LLM, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )


@tool
def default_chat(text: str) -> str:
    """
    Use this only when no other tool is suitable or when no other tool is available, or you don't know what tool to use.
    Use this to chat with the user.
    Remember, This is the final answer.

    """
    global EMOTION

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

    # emotion-sentiment
    # using USER_RESPONSE because ReACT action_input is not always the same as what the user said.
    emotion = classify_emotion(USER_RESPONSE)
    EMOTION = emotion

    return llm(USER_RESPONSE_VERBATIM).strip()


@tool
def classify_emotion(text: str) -> str:
    """Returns the emotion of a text content it is given.
    Use this for any questions that is about the detection or classification of a piece of text content.
    """

    detected_emotion = None
    if CLASSIFY_EMOTION_METHOD == "Logistic Regression (ML)":
        endpoint = "http://127.0.0.1:3000/classifyemotion"
        response = requests.post(endpoint,
                                 headers={
                                     "accept": "text/plain",
                                     "content-type": "text/plain"
                                     },
                                 data=text)
        detected_emotion = response.text
    elif CLASSIFY_EMOTION_METHOD == "DistilBERT (NN)":
        preds = emotion_classifier_distilBERT(text, return_all_scores=True)

        top_k_results = distilBERT_classify_emotion(distilBERT_CLASS_INT2STR,
                                                    preds[0],
                                                    top_k=1,
                                                    desc=True,
                                                    verbose=False)
        for k in top_k_results:
            # print(f"{k.label_name} ({k.label_num}) \t--> \t{round(k.label_score, 2)}")
            return k.label_name
    elif CLASSIFY_EMOTION_METHOD == "GPT3.5 Turbo (LLM)":
        prompt = f"classify the emotion this content is written in:\n{text}."
        prompt += "\nRemember, ONLY return the emotion you classified and nothing else."
        detected_emotion = LLM(prompt)

    return detected_emotion.strip().lower()


@tool
def classify_emotional_sentiment(text: str) -> str:
    """
    Use this when we want to know the sentiment of a emotion.
    An emotion can be positive, neutral or negative. This function takes in emotion as a piece of content and
    classifies the sentiment of that emotion.
    emotion sentiment is returned as a text string.
    """
    prompt = f"Sentiment can be 'positive', 'neutral' or 'negative'. Given the following emotion:\n{text}. Return the sentiment only."
    sentiment = LLM(prompt)
    return sentiment.strip().lower()


@tool
def give_me_a_question_about(text: str) -> str:
    """
    Use this only when the user says "Ask me a question" or "give me a question".
    Use this only when the user wants you to create a question.
    Exit after this tool is executed. This is the final answer.
    """
    global EMOTION
    global QUESTION_LEVEL

    llm = OpenAI(temperature=0.1)

    use_custom_content = True  # prev was False
    # question = None

    # # what is the topic in the 'text'? shd we use it?
    # prompt = f"what is the topic of the following content:\n{text}. \nReturn the topic only."
    # topic = llm(prompt).strip().lower()
    # non_useful_topics = [
    #     "answering questions",
    #     "ask question",
    #     "asking question",
    #     "asking a question",
    #     "question-asking",
    #     "quiz",
    #     "quizzes"
    # ]
    # for useless_topic in non_useful_topics:
    #     if  useless_topic in topic:
    #         use_custom_content = True
    #         break

    # if text is None or text.lower() == 'nothing' or text.lower() == 'none':
    #     use_custom_content = True

    # We have restricted it so that CUSTOM_CONTENT will always be used.
    prompt = f"what is the topic of the following content:\n{CUSTOM_CONTENT}. \nReturn the topic only."
    topic = llm(prompt).strip().lower()

    if use_custom_content:
        if CUSTOM_CONTENT is not None and len(CUSTOM_CONTENT) > 0:
            use_this_content = CUSTOM_CONTENT
        else:
            use_this_content = ''
    else:
        use_this_content = 'text'

    # emotion-sentiment
    emotion = classify_emotion(USER_RESPONSE)
    EMOTION = emotion
    emotion_sentiment = classify_emotional_sentiment(emotion)

    question_generator_beginner_level_template = f'''You are a {topic} teacher who is really
focused on beginners and explaining complex topics in simple to understand terms. 
You assume no prior knowledge. Please create a question about this content: \n{use_this_content}\n
return only the results provided by the tool used and do not change it.'''

    question_generator_mid_level_template = f'''You are a {topic} teacher who is really
focused on experienced professionals. Please create a question about this content: \n{use_this_content}\n
return only the results provided by the tool used and do not change it.'''

    question_generator_expert_level_template = f'''You are a world expert {topic} professor who explains {topic} topics
to advanced audience members. You can assume anyone you answer has a PhD level understanding of Physics. 
Please create a question about this content: \n{use_this_content}\n
return only the results provided by the tool used and do not change it.'''

    if emotion_sentiment == 'positive':
        question = llm(question_generator_expert_level_template)
        QUESTION_LEVEL = "Creating Advanced level question"
    elif emotion_sentiment == 'negative':
        question = llm(question_generator_beginner_level_template)
        QUESTION_LEVEL = "Creating Beginner level question"
    else:
        # neutral
        question = llm(question_generator_mid_level_template)
        QUESTION_LEVEL = "Creating Mid level question"

    question = question.strip()
    prompt = f"If the following content is not a question, transform it into a question:\n{question}"
    final_question = llm(prompt).strip()

    if final_question != question:
        return final_question
    else:
        return question


# Do NOT give the tool "wikipedia" or "google search" related tools, it tends to supercede "give_me_a_question_about"
# ************** Load Tools *******************
# Langchain provided tools: "wikipedia", "human", "llm-math" (calc tool)
tools = load_tools(["llm-math"], llm=LLM)
# my custom tools
# handle_user_response, classify_emotion, classify_emotional_sentiment
my_custom_tools = "'default_chat' and 'give_me_a_question_about'."
tools = tools + [default_chat, give_me_a_question_about]


prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Do not use tools that you do not have access to. Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

PROMPT = ZeroShotAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=LLM, prompt=PROMPT)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, handle_parsing_errors=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=1,
    early_stopping_method="generate"
)

# ***********************************************************************************************************
#                                       MAIN() - STREAMLIT
# ***********************************************************************************************************

USER_RESPONSE_PREFIX = f"You only have access to the tools:\n{my_custom_tools}. "
if user_response := st.chat_input():
    USER_RESPONSE_VERBATIM = user_response
    st.chat_message("user").write(user_response)
    if "hi there" in user_response.lower():
        user_response = user_response.lower()
        user_response = user_response.replace('hi there', 'hi')
        # st.caption(user_response)
    if (("ask" in user_response and "question" in user_response)
            or ("give" in user_response and "question" in user_response)):
        USER_RESPONSE_PREFIX += " The final answer must be a question."
        USER_RESPONSE = f"{USER_RESPONSE_PREFIX}. Create a question from the following text:\n{user_response}."
    else:
        USER_RESPONSE = f"{USER_RESPONSE_PREFIX}. Respond to the following text:\n{user_response}."
    # st.caption(USER_RESPONSE)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        os.environ['OPENAI_API_KEY'] = openai_api_key

    # Zero out Emotions and Sentiment, for this bot, it's only active when generating a question
    EMOTION = 'inactive'
    QUESTION_LEVEL = 'inactive'
    emotion_lbl.write(f"Emotion:\t{EMOTION.title()}")
    question_lvl_lbl.write(f'Question:\t{QUESTION_LEVEL.title()}')

    with st.sidebar:
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            bot_response = agent_chain.run(USER_RESPONSE, callbacks=[st_callback])
            # st.write(response)

    st.chat_message('assistant').write(bot_response)

    # If Agent was generating a question, the vars EMOTION and SENTIMENT may have been modified
    # Let's display them
    emotion_lbl.write(f"Emotion:\t{EMOTION.title()}")
    question_lvl_lbl.write(f'Question:\t{QUESTION_LEVEL.title()}')
