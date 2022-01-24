import imp


import re
"""
Question has a common format PRB-<problem_number_full_document>  CH.PRB- <chapted_number>.<problem_number_chapterwise>.
Example: PRB-4  CH.PRB- 2.1. 

Solution has a common format SOL-<problem_number_full_document>  CH.SOL- <chapted_number>.<problem_number_chapterwise>.
Example: SOL-4  CH.SOL- 2.1.
"""
class ExtractDf:
    def __init__(self, path) -> None:
        self.path = path
        self.df_columns = {
            "problem_number": "",
            "chapterwise_problem_number" : "",
            "question" : "",
            "solution" : ""
        }
        self.read_txt()
        
    def prepare_problem_sol_dict(self):
        self.problem_sol_dict = {}
        for statement in self.text:
            problem_number,chapterwise_problem_number,type = self.detect_question_number(statement)
            if problem_number!= -1:
                print(problem_number,chapterwise_problem_number,type)
                # TODO
                # self.prepare_problem_sol_dict[chapterwise_problem_number] = 
            else:
                print(statement)
        
    def detect_question_number(self,statement:str):
        if statement.startswith('PRB-') or statement.startswith('SOL-'):
            return statement[4],statement[-4:],statement[:3]
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
    path = 'test.txt'
    extract_df_from_text_obj = ExtractDf(path)
    print(extract_df_from_text_obj.text)