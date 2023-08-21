# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.



TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import MultipleChoiceTask


# TODO: Add the BibTeX citation for the task.
_CITATION = ''''''



SUBJECTS = [
    'conceptual_physics', 
    'high_school_microeconomics', 
    'high_school_european_history', 
    'college_biology', 
    'abstract_algebra', 
    'electrical_engineering', 
    'clinical_knowledge', 
    'philosophy', 
    'high_school_macroeconomics', 
    'professional_psychology', 
    'professional_accounting', 
    'high_school_geography', 
    'international_law', 
    'management', 
    'anatomy', 
    'college_physics', 
    'virology', 
    'high_school_statistics', 
    'global_facts', 
    'miscellaneous', 
    'human_aging', 
    'moral_disputes', 
    'logical_fallacies', 
    'college_mathematics', 
    'high_school_world_history', 
    'nutrition', 
    'us_foreign_policy', 
    'computer_security', 
    'high_school_physics', 
    'high_school_psychology', 
    'high_school_us_history', 
    'formal_logic', 
    'medical_genetics', 
    'security_studies', 
    'marketing', 
    'college_chemistry', 
    'high_school_computer_science', 
    'machine_learning', 
    'high_school_biology', 
    'professional_medicine', 
    'human_sexuality', 
    'public_relations', 
    'jurisprudence', 
    'college_medicine', 
    'high_school_mathematics', 
    'moral_scenarios', 
    'high_school_chemistry', 
    'elementary_mathematics', 
    'astronomy', 
    'business_ethics', 
    'prehistory', 
    'professional_law', 
    'college_computer_science', 
    'world_religions', 
    'sociology', 
    'high_school_government_and_politics', 
    'econometrics'
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"cmmlu-{sub}": create_task(sub) for sub in SUBJECTS}


def create_task(subject):
    class Cmmlu(GeneralCmmluTask):
        def __init__(self):
            super().__init__(subject)

    return Cmmlu



class GeneralCmmluTask(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = 'csv'
    DATASET_NAME = None

    def __init__(self, subject):
        data_dir = f'custom_dataset/cmmlu/{subject}'
        super().__init__(data_dir=data_dir)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset['train'])

    def validation_docs(self):
        return map(self._process_doc, self.dataset['validation'])

    def test_docs(self):
        return map(self._process_doc, self.dataset['test'])

    def _process_doc(self, doc):
        
        keys = ["A", "B", "C", "D"]
        choices = []
        for key in keys:
            choices.append(doc[key])
        gold = keys.index(doc["answer"])
        
        query = "Question: " + doc["question"] + "\nChoices:\n"
        query += "".join([f"{key}. {doc[key]}\n" for key in keys])
        query += "Answer:"
                
        return {
            "query": query,  # The query prompt.
            "choices": choices,  # The list of choices.
            "gold": gold,  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        return doc["query"]
