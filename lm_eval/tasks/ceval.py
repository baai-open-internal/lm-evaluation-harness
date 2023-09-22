# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.



TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import MultipleChoiceTask


# TODO: Add the BibTeX citation for the task.
_CITATION = """@article{huang2023ceval,
title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models}, 
author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
journal={arXiv preprint arXiv:2305.08322},
year={2023}
}
"""



SUBJECTS = [
    'tax_accountant', 
    'fire_engineer', 
    'high_school_politics', 
    'middle_school_chemistry', 
    'plant_protection', 
    'computer_architecture', 
    'art_studies', 
    'legal_professional', 
    'clinical_medicine', 
    'ideological_and_moral_cultivation', 
    'high_school_geography', 
    'teacher_qualification', 
    'probability_and_statistics', 
    'high_school_chinese', 
    'sports_science', 
    'high_school_history', 
    'middle_school_geography', 
    'operating_system', 
    'college_programming', 
    'discrete_mathematics', 
    'urban_and_rural_planner', 
    'electrical_engineer', 
    'computer_network', 
    'college_physics', 
    'marxism', 
    'college_economics', 
    'law', 
    'education_science', 
    'middle_school_biology', 
    'civil_servant', 
    'business_administration', 
    'basic_medicine', 
    'metrology_engineer', 
    'advanced_mathematics', 
    'veterinary_medicine', 
    'high_school_physics', 
    'middle_school_politics', 
    'accountant', 
    'college_chemistry', 
    'physician', 
    'modern_chinese_history', 
    'high_school_biology', 
    'middle_school_physics', 
    'mao_zedong_thought', 
    'high_school_mathematics', 
    'high_school_chemistry', 
    'environmental_impact_assessment_engineer', 
    'middle_school_mathematics', 
    'professional_tour_guide', 
    'middle_school_history', 
    'chinese_language_and_literature', 
    'logic'
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"ceval-{sub}": create_task(sub) for sub in SUBJECTS}


def create_task(subject):
    class Ceval(GeneralCevalTask):
        def __init__(self):
            super().__init__(subject)

    return Ceval



class GeneralCevalTask(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = 'csv'
    DATASET_NAME = None

    def __init__(self, subject):
        data_dir = f'custom_dataset/ceval/{subject}'
        self.DATASET_NAME = subject
        super().__init__(data_dir=data_dir)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return map(self._process_doc, self.dataset['train'])

    def validation_docs(self):
        return map(self._process_doc, self.dataset['validation'])

    def test_docs(self):
        return map(self._process_doc, self.dataset['test'])
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description= f"以下是关于{subject}的单项选择题，请直接给出正确答案的选项。"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            答案：
            """

            question = doc["question"].strip()
            choices = "".join(
                [f'{key}. {doc[key]}\n' for key in keys]
            )
            prompt = f"{question}\n{choices}答案："
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": ord(doc["answer"])-ord("A"),
        }

    def doc_to_text(self, doc):
        return doc["query"]

    '''def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["train"]))

        # use the unchanged order of the dev set without sampling,
        return self._fewshot_docs[:k]'''

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]