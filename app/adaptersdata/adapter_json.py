import json

from langchain_core.documents import Document

#Написать адаптеры для каждого типа документов
#Пока что просто функция, которая возвращает все документы из заданного файла

def get_json_data(path2file):
    ext = path2file.split('.')[-1]
    if ext != 'json':
        return ([], [])
    
    all_data = []
    with open(path2file, 'r', encoding='utf-8') as file:
        cont = file.read()  
        all_data = json.loads(cont) 


    #Добавление пациентов в ChromoDB
    documents = [] #массив документов
    documents_ids = [] #массив id документов
    doc_id = 0
    for item in all_data: #Идем по элементам (пациентам)
        for vis in item['visits']: #Идем по посещениям
            doct_text_arr = []
            metadata = {}

            #Проверка, могут быть None
            #Для page_content
            if vis['complaint'] is not None:
                doct_text_arr.append(vis['complaint'])
                metadata['complaint'] = vis['complaint']
            if vis['anamnesis'] is not None:
                doct_text_arr.append(vis['anamnesis'])
            if vis['objective_status'] is not None:
                doct_text_arr.append(vis['objective_status'])
                metadata['objective_status'] = vis['objective_status']
            
            #Для metadata
            if vis['visit_number'] is not None:
                metadata['visit_number'] = vis['visit_number']
            if vis['diagnosis_details'] is not None:
                metadata['diagnosis_details'] = vis['diagnosis_details']
            if vis['recommendations'] is not None:
                metadata['recommendations'] = vis['recommendations']
            if item['patient_id'] is not None:
                metadata['patient_id'] = item['patient_id']

            
            #формируем текст документа
            doc_text = " ".join(st for st in doct_text_arr)

            documents.append(Document(page_content=doc_text, metadata=metadata))
            documents_ids.append(f'json_docID{doc_id}')
            doc_id += 1    
    return (documents, documents_ids)