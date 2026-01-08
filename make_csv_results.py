import pandas as pd
from tqdm.auto import tqdm 

from google.genai import Client
from google.genai import types

import asyncio

input_file = "Planning_테스트 자동화_text.csv"
output_file = "result_csv.csv"

#모델 및 데이터 선택
model = "gemini-2.5-flash"
df = pd.read_csv(f"./data/input/{input_file}")

client = Client(
    vertexai=True,
    project="beha-data",
    location="us-central1",
)

# gemini 실행 함수
# def generate_answer(model, content):
    
#     # 2. 텍스트 생성
#     response = client.models.generate_content(
#         model=model,
#         contents=str(content)
#     )
    
#     return response.text

# gemini 실행 루프
# for index, row in tqdm(df.iterrows()):
#     query = row.get("사용자 질의")
#     template = row.get("프롬프트")
#     prompt = template.replace("{사용자 질의}", str(query))
#     answer = generate_answer(model, prompt)
#     df.loc[index, '결과'] = answer

# gemini 병렬 실행 함수
async def generate_batch_answers(model, prompts):
    async def call_api(contents):
        try:
            # .aio는 비동기 클라이언트를 호출합니다.
            response = await client.aio.models.generate_content(model=model, contents=contents)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    tasks = [call_api(p) for p in prompts]

    return await asyncio.gather(*tasks)
   
async def process_all():

    batch_size = 8
    results = []

    # gemini 병렬 실행 루프
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]

        prompts = []
        for _, row in batch_df.iterrows():
            query = row.get("사용자 질의")
            template = row.get("프롬프트")
            prompt = template.replace("{사용자 질의}", str(query))
            prompts.append(prompt)

        batch_answers = await generate_batch_answers(model, prompts)
        results.extend(batch_answers)

    df['결과'] = results

    df.to_csv(f"./data/output/{output_file}", index=False)

    print("CSV 생성 완료!")

if __name__ == "__main__":
    asyncio.run(process_all())