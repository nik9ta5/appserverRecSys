from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

import torch

import argparse

# -------------------------- parser --------------------------

parser = argparse.ArgumentParser(description="Backend for recoment system prediction diagnose")

#Параметры которые передаются в командную строку
#Отвечают за заполнение векторной базы данных; Удаление векторной базы данных
parser.add_argument('--fullVDB', action='store_true', help="Заполнить векторную базу данных")
parser.add_argument('--delVDB', action='store_true', help="Удалить все данные из коллекции векторной базы данных")

args = parser.parse_args()


# -------------------------- CUDA --------------------------

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


# -------------------------- Импорты из проекта --------------------------

from vectordatabase.embeddingFunction import EmbeddingFunction
from vectordatabase.vectorDatabase import CreateVectorDatabase

from llmmodule.languageModel import LanguageModel

from adaptersdata.adapter_csv import get_csv_data
from adaptersdata.adapter_json import get_json_data


# -------------------------- Векторная база данных --------------------------

embedding_func = EmbeddingFunction(model_name="ai-forever/sbert_large_nlu_ru", device=device)

vector_database = CreateVectorDatabase('medCollection', embedding_function=embedding_func)


if args.delVDB:
    vector_database.delete_collection()

    vector_database = CreateVectorDatabase('medCollection', embedding_function=embedding_func)
    print('Delete data vector data base')


#Заполнение VDB
if args.fullVDB:
    jsonData_doc, jsonData_doc_id = get_json_data('./datafolder/word_base.json')
    csvData_doc, csvData_doc_id = get_csv_data('./datafolder/final_base_disease.csv')

    print('\nFilling in a vector database\n')
    vector_database.add_documents(documents=jsonData_doc, ids=jsonData_doc_id) 
    vector_database.add_documents(documents=csvData_doc, ids=csvData_doc_id) 

print("object in vector DataBase: ", vector_database._collection.count())


# -------------------------- Prompt generating --------------------------
#"content": f"Вы — помощник врача в вопросах постановки диагноза пациенту. Ваши ответы должны быть основаны только на информации, представленной ниже. Используйте только те данные, которые содержатся в запросе. Ответьте одним словом, выбрав диагноз из предложенных. Не добавляйте объяснений, предположений или дополнительной информации. Ответ должен быть исключительно диагнозом. \nЖалобы пациента: {msg} \nВозможные диагнозы и их описания: {str_extraction_vdb}"

def prompt_create(msg : str, extraction_vdb : list):
    # extraction_vdb - массив документов
    #можно поэксперементировать и подобавлять еще деталей (анамнез, рекомендации и др.) НО! проверить, что для данного объекта они существуют
    str_extraction_vdb = "\n".join([f"\n {doc.metadata['diagnosis_details']} — {doc.metadata['objective_status']}" for doc in extraction_vdb])
    return [{
            "role": "system",
            "content": f"""
            Вы — помощник врача, который помогает определить диагноз, основываясь на предоставленном описании симптомов. Твоя задача — выбрать диагноз, который соответствует данному описанию из списка возможных диагнозов. Ответ должен быть точно таким, каким он написан в списке диагнозов, без изменений.
            Вот описание симптомов:
            {msg}

            Вот список диагнозов с их описаниями:
            {str_extraction_vdb}
            
            Ответь **одним словом**, которое является диагнозом, указанным в списке. Ответ должен быть **точным** и **без изменений**.
            """
    }]


# -------------------------- Language model --------------------------

model = LanguageModel(device=device)


# -------------------------- Модели данных --------------------------

class PacientRequestData(BaseModel):
    visits: int
    complaint: str
    anamnesis: str
    objective_status: str
    diagnosis_details: str
    recommendations: str

#Модель для ответа 
class RecomendItem(BaseModel):
    diagnosis_details: str
    objective_status: str

class ResponseModel(BaseModel):
    data: list[RecomendItem]  # Список объектов Item


# -------------------------- Сервер --------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #С какого домена разрешено отправлять запрос
    allow_credentials=True,
    allow_methods=["*"], #Какие методы разрешены
    allow_headers=["*"], #Какие заголовки разрешены
)


#POST
@app.post("/preddiagnose")
async def predict_diagnose(pacient_data: PacientRequestData):
    #pacient_data - данные с клиента
    if pacient_data.complaint is None or pacient_data.anamnesis is None or pacient_data.objective_status is None:
        return {"exceptions" : "not data complaint or anamnesis or objective_status"}
    
    #Поиск по векторной базе данных
    str_for_search = " ".join([pacient_data.complaint, pacient_data.anamnesis, pacient_data.objective_status])
    res_search = vector_database.similarity_search(query=str_for_search, k=10)

    #Создание промпта
    prompt = prompt_create(str_for_search, res_search)
    # print(prompt)
    # print('-----------------------')

    #Генерация ответа моделью
    output_text_model = model.generate(prompt)
    # print(output_text_model)
    # print('-----------------------')
    
    #Формирование для отправки клиенту
    send_client_data = [{"diagnosis_details":item_search.metadata['diagnosis_details'], "objective_status":item_search.metadata["objective_status"]} for item_search in res_search]
    # print(send_client_data)
    # print('-----------------------')

    #Ответ
    return {"response" : "server response", "response_model" : output_text_model, 'vector_search' : send_client_data}



#Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)