from lm_eval.base import MultipleChoiceTask



SUBJECTS = [
    "LLSRC",
    "SLPWC",
    "SLRFC",
    "SLSRC",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"c_sem_v2-{sub}": create_task(sub) for sub in SUBJECTS}


def create_task(subject):
    class C_SEM_V2(C_SEM_V2_Subject):
        def __init__(self):
            super().__init__(subject)

    return C_SEM_V2


class C_SEM_V2_Subject(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = 'custom_dataset/c_sem_v2'
    DATASET_NAME = None
    SUBJECT = None

    def __init__(self, subject):
        self.DATASET_PATH = f'custom_dataset/c_sem_v2/{subject}'
        self.SUBJECT = subject
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.SUBJECT
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
        '''return {
            "query": format_example(doc, keys),
            "choices": [doc[key] for key in keys],
            "gold": ord(doc["Answer"])-ord("A"),
        }'''
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": ord(doc["answer"])-ord("A"),
        }
        
    '''def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["validation"]))

        # use the unchanged order of the dev set without sampling,
        # just as in the original code https://github.com/hendrycks/test/blob/master/evaluate.py#L28
        return self._fewshot_docs[:k]'''

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
