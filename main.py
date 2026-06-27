import sys
from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY")
if not MODEL_NAME:
    raise ValueError("Missing MODEL_NAME")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=OPENROUTER_API_KEY)



def load_prompt(path):
    return Path(path).read_text(encoding="utf-8")

def prompt_chain(project_description):

    def run_prompt(filename, **kwargs):
        template = load_prompt(filename)
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        response = chain.invoke(kwargs)

        return response.content

    interpretation = run_prompt("prompts/1_interpret.txt",
                                 project_description=project_description)
    print (f"""
    Step 1 - Interpretation:
    {interpretation} """)

  
    possible_categories = run_prompt("prompts/2_possible_categories.txt",
        interpretation=interpretation)
    if possible_categories is None:
     raise RuntimeError("Step 2 failed")
    print (f"""
    Step 2 - Possible Categories:
    {possible_categories}""")


    best_category = run_prompt("prompts/3_best_category.txt",
        possible_categories=possible_categories,)
    if best_category is None:
     raise RuntimeError("Step 3 failed")
    print (f"""
    Step 3 - Best Category:
    {best_category}""")
    

    extra_info = run_prompt("prompts/4_extra_info.txt",
        best_category=best_category)
    if extra_info is None:
     raise RuntimeError("Step 4 failed")
    print (f"""
    Step 4 - Extra Info:
    {extra_info}""")

    initial_assessment = run_prompt("prompts/5_initial_assessment.txt",
        project_description=project_description,
        interpretation=interpretation,
        best_category=best_category,
        extra_info=extra_info,                 
        )

    return initial_assessment





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your prompt here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    result = prompt_chain(prompt)

    if result:
        print(f"""
    Response:
    {result}""")
