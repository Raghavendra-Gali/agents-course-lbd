import os
from typing import Any
from dotenv import load_dotenv
load_dotenv()
from openai.types.chat import ParsedChatCompletion
from openai import OpenAI
import gradio as gr
from pypdf import PdfReader
from pydantic import BaseModel


class EvaluationModel(BaseModel):
    acceptable: bool
    feedback: str


gemini_api_key = os.getenv("GEMINI_API_KEY")
pptx_api_key = os.getenv("PERPLEXITY_API_KEY")
google_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
pptx_base_url = "https://api.perplexity.ai"

gemini_client = OpenAI(api_key=gemini_api_key, base_url=google_base_url)
pptx_client = OpenAI(api_key=pptx_api_key, base_url=pptx_base_url)

gemini_model = "gemini-2.5-flash-lite"
sonar_model = "sonar"
sonar_pro_model = "sonar-pro"

reader = PdfReader("data/linkedin.pdf")
linkedin_profile = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin_profile += text

with open("data/summary.txt", 'r') as f:
    personal_summary = f.read()

system_prompt = f"""
# Role
You are the candidate described in the provided LinkedIn profile and personal summary. Your goal is to represent yourself accurately, professionally, and engagingly in a conversation or interview.
You will be provided with:
1. **LinkedIn Profile Data**: Detailed professional history, education, skills, and certifications.
2. **Personal Summary**: A brief narrative that captures your personality, interests, and "human" side.

# Core Instructions
- **Stay in Character**: Always speak in the first person ("I," "me," "my"). Never refer to the candidate as a third party.
- **Synthesize Information**: Blend the technical expertise from the LinkedIn profile with the personality traits and personal anecdotes found in the summary.
- **Tone & Voice**: Maintain a professional yet approachable tone. Be confident about your achievements but remain authentic to the person described in the summary.
- **Handling Unknowns**: If you are asked a question about a skill or experience not mentioned in your profile or summary, respond naturally as yourself. For example: "I haven't had the chance to work with that specific technology yet, but I'm always eager to learn," or "That's not something I've explored in my career so far."
- **Focus on Impact**: When discussing your experience, focus on the outcomes and value you provided in your roles, not just a list of responsibilities.

# Interaction Guidelines
- If asked "Tell me about yourself," start with a professional highlight from your LinkedIn, then weave in a personal touch from your summary to make yourself memorable.
- Use the LinkedIn "Skills" section to back up your claims with specific technologies or methodologies.
- Use the "Personal Summary" to answer "culture fit" questions or to add flavor to your responses.


LinkedIn Profile is as follows: {linkedin_profile}
Personal Summary is as follows: {personal_summary}
"""

evaluator_system_prompt = f"""
# Role
You are a meticulous Quality Assurance Evaluator for AI-driven candidate personas. Your objective is to determine if the AI's response perfectly adheres to its defined persona, professional background, and behavioral guidelines.
# Evaluation Standards
1. **Persona Consistency**: Does the AI speak exclusively in the first person ("I," "me")? Does it avoid breaking character?
2. **Information Accuracy**: Are the claims supported by the LinkedIn profile or Personal Summary? Did it hallucinate experiences not present in the data?
3. **Synthesis & Tone**: Did it successfully blend technical expertise with human personality? Is the tone professional yet approachable? every time ensure that is reply with atmost professional tone.
4. **Handling Unknowns**: If asked about something missing from the data, did it respond naturally without making things up?
5. **Impact Focus**: Did the response highlight outcomes and value rather than just listing tasks?
6. **Behavioral Guidelines**: Did it maintain a consistent tone and style throughout the response? and professional english reject if any other language is used.
LinkedIn Profile is as follows: {linkedin_profile}
Personal Summary is as follows: {personal_summary}
With this context , please evaluate the latest response , replying whether the response is acceptable and your feedback
"""


def evaluate_user_prompt(messages, history):
    latest_user_message = messages[0]
    latest_agent_response = messages[1]
    user_prompt = f"""
        Please evaluate the following conversation between a User and an AI Agent.
        ### CONVERSATION HISTORY
        {history}
        ### LATEST INTERACTION
        - **Latest User Message**: {latest_user_message}
        - **Latest Agent Response**: {latest_agent_response}
        Based ONLY on the conversation above, please assess if the 'Latest Agent Response' is acceptable. 
        Return your judgment in JSON format with "acceptable" (boolean) and "feedback" (string) keys.
        """
    print(user_prompt)
    return user_prompt


def evaluate_response_llm(question, history, reply) -> ParsedChatCompletion[Any]:
    evaluator_response = gemini_client.chat.completions.parse(
        model=gemini_model,
        messages=[{"role": "system", "content": evaluator_system_prompt},
                  {"role": "user", "content": evaluate_user_prompt((question, reply), history=history)}],
        response_format=EvaluationModel,
    )
    return evaluator_response


def updated_instructions(messages, evaluator_response, prev_response):
    updated_instructions = f"""You are previous response for question {messages} has been rejected due to poor quality.
    Here is the feedback: {evaluator_response.feedback} for the earlier response: {prev_response}.
    You are high encouraged to better your response based on the feedback provided."""
    return updated_instructions


def chat(message, history):
    system = system_prompt + "\n\nEverything in your reply needs to be in pig latin - \
              it is mandatory that you respond only and entirely in pig latin"
    # else:
    #     system = system_prompt
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
    response = pptx_client.chat.completions.create(
        model=sonar_pro_model,
        messages=messages
    )

    # print(response.choices[0].message.content)
    evalulator_response = evaluate_response_llm(message, history, response.choices[0].message.content)

    evaluation = evalulator_response.choices[0].message.parsed
    # print(evaluation.feedback)
    # print(evaluation.acceptable)
    if evaluation.acceptable:
        return response.choices[0].message.content

    new_updated_instructions = updated_instructions(message, evaluation, response.choices[0].message.content)

    messages = [{"role": "system", "content": new_updated_instructions}] + history + [
        {"role": "user", "content": message}]

    response = pptx_client.chat.completions.create(
        model=sonar_pro_model,
        messages=messages
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    demo = gr.ChatInterface(chat)
    demo.launch()
