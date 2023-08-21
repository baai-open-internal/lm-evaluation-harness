import os
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time
import requests
import json
import math
from typing import List, Mapping, NewType, Optional, Tuple, Union

def get_response(prompt,
                 url = "http://127.0.0.1:5050/func", 
                 temperature=0.9,
                num_return_sequences=1 ,
                max_new_tokens=0,
                top_p=0.9,
                echo_prompt=False,
                top_k_per_token=100000,
                stop_sequences=[],
                candidate=["A","B","C","D"]):
    
    prompt = prompt[0] if len(prompt) == 1 else prompt
    raw_request = {
            "engine": 'gpt2',
            "prompt": prompt,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,  
            "max_new_tokens": max_new_tokens,
            "top_p":top_p,
            "echo_prompt": echo_prompt,
            "top_k_per_token": top_k_per_token,
            "stop_sequences": stop_sequences,
            # "candidate": candidate
            # "seed": 123
        }
    for i in range(5):  # 尝试5次
        try:
            response = requests.post(url, json=json.dumps(raw_request))
            if response.status_code == 200:
                response = response.json()
                return response
            else:
                raise Exception(f"Request failed with status code {response.status_code}")
        except Exception as e:
            if i < 4:  # 如果不是最后一次尝试，则等待几秒再重试
                print(f"Request failed with exception {e}. Retrying...")
                time.sleep(10 * (i+1))
            else:  # 最后一次尝试如果还是失败，则返回None
                return None
    

# Update the get_result function to handle the new output format
def get_result(response_cc, response_c):
    is_greedy = False
    assert len(response_cc["completions"]) == len(response_c["completions"])
    continuation_logprobs_list = []
    for k in range(len(response_cc["completions"])):
        tokens = response_cc["completions"][k]["tokens"]
        probs = response_cc["completions"][k]["logprobs"]
        continuation_logprobs = 0
        
        continuation_tokens = response_c['completions'][k]['tokens'][1:]
        ans_idx = -1

        for i in range(len(tokens)-len(continuation_tokens) + 1):
            if tokens[len(tokens)-len(continuation_tokens)-i:len(tokens)-i] == continuation_tokens:
                ans_idx = len(tokens)-len(continuation_tokens)-i
                break
            
        if ans_idx == -1:
            print(f'tokens: {tokens}')
            print(f'continuation_tokens: {continuation_tokens}')
            raise Exception("Answer not found")
        
        for i in range(len(continuation_tokens)):
            if probs[i + ans_idx] == 0:
                continuation_logprobs = -float('inf')
                break
            else:
                continuation_logprobs += math.log(probs[i + ans_idx])
        
        continuation_logprobs_list.append((continuation_logprobs,is_greedy))
    

    # continuation_logprobs = sum(logprobs[ctxlen:])
    # tokens = completion["tokens"]
    # top_logprobs_dicts = completion["top_logprobs_dicts"]

    # for i in range(ctxlen, len(tokens)):
    #     token = tokens[i]
    #     top_token = max(top_logprobs_dicts[i].keys(), key=lambda x: top_logprobs_dicts[i][x])
    #     if top_token != token:
    #         is_greedy = False
    #         break

    return continuation_logprobs_list




class AquilaLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, url="http://127.0.0.1:5050/func", truncate=False, batch_size=1):
        """

        :param engine: str
            OpenAI API engine (e.g. davinci)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()


        self.url = url
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.model_info = "aquila"
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size)

        # to make the annoying "Using pad_token, but it is not set yet." error go away
        self.tokenizer.pad_token = "<|endoftext|>"
        assert self.tokenizer.encode("hello\n\nhello") == [31373, 198, 198, 31373]
        self.truncate = truncate
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(
            ["<|endoftext|>"]
        )[0]
    
    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    # def loglikelihood(self, requests):
    #     res = []
        
    #     cnt = 0
    #     for context, continuation in tqdm(requests):
    #         '''print(context)
    #         print()
    #         print(continuation)'''
    #         response_cc = get_response(
    #             url=self.url, 
    #             prompt=context+continuation,
    #         )
    #         response_c = get_response(
    #             url=self.url,
    #             prompt=continuation,
    #         )
    #         if response_cc is not None and response_c is not None:
    #             self.model_info = response_cc["model_info"]
    #             answer = get_result(response_cc, response_c)
    #         else:
    #             cnt += 1
    #             answer = (0, False)
    #         res.append(answer)
    #     print(f'unfaithful numbers: , {cnt}/{len(requests)}')
    #     return res
    
    def loglikelihood(self, requests):
        res = []
        
        cnt = 0
        for chunk in tqdm(utils.chunks(requests, self.batch_size)):
            response_cc = get_response(
                url=self.url, 
                prompt=[c[0]+ c[1] for c in chunk],
            )
            response_c = get_response(
                url=self.url,
                prompt=[c[1] for c in chunk],
            )
            if response_cc is not None and response_c is not None:
                self.model_info = response_cc["model_info"]
                answer = get_result(response_cc, response_c)
            else:
                cnt += 1
                answer = [(0, False)]
            res.extend(answer)
        print(f'unfaithful numbers: , {cnt}/{len(requests)}')
        return res  

    '''def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            inps = []
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)

            response = oa_completion(
                engine=self.engine,
                prompt=inps,
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                logprobs=10,
                stop=until,
            )

            for resp, (context, until_) in zip(response.choices, chunk):
                s = resp["text"]

                for term in until_:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until_), s)

                res.append(s)

        return re_ord.get_original(res)'''
    
    
    def greedy_until(self, requests: List[Tuple[str, Union[List[str], str]]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]
        
        results = []
        reorder = utils.Reorderer(requests, _collate)

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False), self.batch_size
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get('until', None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None
            
            # TODO: Find a better way to handle stop sequences for 0-shot.
            '''if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]'''
            
            #until = stop_sequences

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length
                            
            responses = get_response(prompt=context[0], url=self.url, max_new_tokens=max_tokens, stop_sequences=stop_sequences)
            if responses is not None:
                self.model_info = response["model_info"]
                response = responses['completions'][0]['text'].strip()
                for term in stop_sequences:
                    response = response.split(term)[0]

            
                print(context)
                print(response)
            
            else:
                response = ""
            
            self.cache_hook.add_partial("greedy_until", (context, stop_sequences), response)
            results.append(response)
                            
            '''token_context = self.tok_encode_batch(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())'''
            
            '''for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)'''
                
        return reorder.get_original(results)


    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
