import re
from numpy import append
import pandas as pd
import copy
"""
Question has a common format PRB-<problem_number_full_document>  CH.PRB- <chapted_number>.<problem_number_chapterwise>.
Example: PRB-4  CH.PRB- 2.1. 

Solution has a common format SOL-<problem_number_full_document>  CH.SOL- <chapted_number>.<problem_number_chapterwise>.
Example: SOL-4  CH.SOL- 2.1.
"""
class ExtractDf:
    def __init__(self, path) -> None:
        self.path = path
        self.df_columns_template = {
            "problem_number": "",
            "chapterwise_problem_number" : "",
            "PRB" : [],
            "SOL" : []
        }
        self.read_txt()
    
    def convert_to_df(self):
        dataset_list = []
        for _,object in self.problem_sol_dict.items():
            problem_dataset = {}
            problem_dataset['problem_number'] = object["problem_number"]
            problem_dataset['chapterwise_problem_number'] = object['chapterwise_problem_number']
            problem_dataset["PRB"] = " ".join(object['PRB']).replace("-","")  
            problem_dataset["SOL"] = " ".join(object["SOL"]).replace("-","")
            dataset_list.append(problem_dataset)
        df = pd.DataFrame(dataset_list)
        df.to_csv("dataset_chapter_2.csv")
        return df

    def prepare_problem_sol_dict(self):
        self.problem_sol_dict = {}
        for statement in self.text:
            problem_number,chapterwise_problem_number,type = self.detect_question_number(statement)
            if problem_number!= -1:
                self.problem_number,self.chapterwise_problem_number,self.type = problem_number,chapterwise_problem_number,type         
                if self.chapterwise_problem_number not in self.problem_sol_dict.keys():
                    self.problem_sol_dict[self.chapterwise_problem_number] = {}
                    self.problem_sol_dict[self.chapterwise_problem_number]["PRB"] =[]
                    self.problem_sol_dict[self.chapterwise_problem_number]["SOL"] =[]
                self.problem_sol_dict[self.chapterwise_problem_number]["problem_number"] = self.problem_number
                self.problem_sol_dict[self.chapterwise_problem_number]["chapterwise_problem_number"] = self.chapterwise_problem_number
                
            else:
                try:
                    self.problem_sol_dict[self.chapterwise_problem_number][self.type].append(statement)
                except Exception as AttributeError:
                    pass
        
    def detect_question_number(self,statement:str):
        #PRB-22  CH.PRB- 2.19.
        if statement.startswith('PRB-') or statement.startswith('SOL-'):
            first = re.findall(r'[0-9]',statement.split()[0])
            second = re.findall(r'[0-9]*\.[0-9]*\.',statement)
            # problem_number,chapterwise_problem_number,type
            return "".join(first),second[0],statement[:3]
        return -1,-1,-1
    
    def processing(self):
        for idx in range(len(self.text)):
            self.text[idx] = (self.text[idx][:-1] if self.text[idx].endswith('\n') else self.text[idx])
            self.text[idx] = (self.text[idx][1:-1] if self.text[idx].startswith("(") and self.text[idx].endswith(")") else self.text[idx])
            self.text[idx] = self.text[idx].replace("\uf059","")
            self.text[idx] = self.text[idx].replace("\uf14b","")
            self.text[idx] = self.text[idx].replace("\x0c","")
            self.text[idx] = self.text[idx].replace("\x0c","")
            self.text[idx] = self.text[idx].replace("cid:4","")
            self.text[idx] = ("" if self.text[idx].startswith('Chapter') else self.text[idx])
            if self.text[idx].isdigit() or bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', self.text[idx])):
                self.text[idx] = ""
        while "" in self.text:    
            self.text.remove("")
    
    def read_txt(self):
        f = open(self.path,'r',encoding="utf8")
        self.text = f.readlines()
        self.processing()
        self.prepare_problem_sol_dict()
        

if __name__ == '__main__':
    path = 'Sample_Chapter_2_Q&A.txt'
    extract_df_from_text_obj = ExtractDf(path)
    df = extract_df_from_text_obj.convert_to_df()