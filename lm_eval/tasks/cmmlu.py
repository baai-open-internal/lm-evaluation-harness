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
    DATASET_PATH = 'custom_dataset/cmmlu'
    DATASET_NAME = None
    SUBJECT = None

    def __init__(self, subject):
        self.DATASET_PATH = f'custom_dataset/cmmlu/{subject}'
        self.SUBJECT = subject
        super().__init__()

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
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.SUBJECT
        description= f"以下是关于{subject}的单项选择题，请直接给出正确答案的选项。"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
    '''def _process_doc(self, doc):
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
        keys_content = [doc[key] for key in keys]
        return {
            "query": format_example(doc, keys),
            "choices": keys_content,
            "gold": ord(doc["answer"])-ord("A"),
        }'''

    
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
