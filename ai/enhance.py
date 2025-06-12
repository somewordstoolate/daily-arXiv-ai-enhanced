import os
import json
import sys

import dotenv
import argparse

import langchain_core.exceptions
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from structure import Structure
if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    preferable_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)
    preference = os.environ.get('CATEGORIES', 'cs.CV, cs.CL').split(',')
    preference = set(map(lambda x: x.strip(), preference))
    for item in unique_data:
        record_cate = set(item["categories"])
        if len(preference & record_cate) == 0:
            continue
        preferable_data.append(item)
    
        
        
        

    data = preferable_data

    print('Open:', args.data, file=sys.stderr)

    llm = ChatOpenAI(model=model_name)

    output_json_parser = PydanticOutputParser(pydantic_object=Structure)
  
    print('Connect to:', model_name, file=sys.stderr)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template,partial_variables={
    "format_instructions": output_json_parser.get_format_instructions()
})
    ])

    chain = prompt_template | llm | output_json_parser


    def format_outputs(d):
        try:
            response: Structure = chain.invoke({
                "language": language,
                "content": d['summary']
            })
            d['AI'] = response.model_dump()
        except langchain_core.exceptions.OutputParserException as e:
            print(f"{d['id']} has an error: {e}", file=sys.stderr)
            d['AI'] = {
                 "tldr": "Error",
                 "motivation": "Error",
                 "method": "Error",
                 "result": "Error",
                 "conclusion": "Error"
            }
        return d

    formatted_chain = (
        RunnableLambda(
            lambda x: format_outputs(x)
        )
        .with_retry(stop_after_attempt=3)
        .with_config(max_concurrency=512)
    )
    formatted_outputs = formatted_chain.batch(data)
      
    with open(args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl'), "w") as f:
        for d in formatted_outputs:
            f.write(json.dumps(d) + "\n")

    # for idx, d in enumerate(data):
    #     try:
    #         response: Structure = chain.invoke({
    #             "language": language,
    #             "content": d['summary']
    #         })
    #         d['AI'] = response.model_dump()
    #     except langchain_core.exceptions.OutputParserException as e:
    #         print(f"{d['id']} has an error: {e}", file=sys.stderr)
    #         d['AI'] = {
    #              "tldr": "Error",
    #              "motivation": "Error",
    #              "method": "Error",
    #              "result": "Error",
    #              "conclusion": "Error"
    #         }
    #     with open(args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl'), "a") as f:
    #         f.write(json.dumps(d) + "\n")

    #     print(f"Finished {idx+1}/{len(data)}", file=sys.stderr)

if __name__ == "__main__":
    main()
