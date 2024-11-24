import pandas as pd
from langchain_core.documents import Document

#Написать адаптеры для каждого типа документов
#Пока что просто функция, которая возвращает все документы из заданного файла

def get_csv_data(path2file):
    ext = path2file.split('.')[-1]
    if ext != 'csv':
        return ([], [])

    doc_id = 0
    frame_bol = pd.read_csv(path2file)

    documents_fr2 = [] #массив документов
    documents_ids_fr2 = [] #массив id документов

    for index, row in frame_bol.iterrows():
        document_item = Document(page_content=row['Симптомы'], metadata={
            'diagnosis_details' : row['Index'],
            'objective_status' : row['Симптомы'],
            'mkb_10': row['МКБ_10']
            })
        documents_fr2.append(document_item)
        documents_ids_fr2.append(f'csv_docID{doc_id}')
        
        doc_id += 1
    
    return (documents_fr2, documents_ids_fr2)