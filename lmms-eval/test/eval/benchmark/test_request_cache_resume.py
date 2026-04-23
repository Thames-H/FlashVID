import tempfile
import unittest
from pathlib import Path
from unittest import mock

import lmms_eval.api.model as model_module
from lmms_eval.api.instance import Instance


def make_request(doc_id: int, task_name: str = "videomme", split: str = "validation") -> Instance:
    return Instance(
        request_type="generate_until",
        arguments=("context", {}, lambda doc: [], doc_id, task_name, split),
        idx=doc_id,
        metadata={"task": task_name, "doc_id": doc_id, "repeats": 1},
    )


class DummyResumeLm(model_module.lmms):
    def __init__(self):
        super().__init__()
        self.task_dict = {"videomme": {"validation": {}}}
        self.model_name = "dummy-resume-model"
        self.generated_doc_ids = []

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        original_requests = list(requests)
        cached_responses, pending_requests = self.split_requests_by_cache(original_requests)
        generated_responses = []
        for request in pending_requests:
            response = f"generated-{request.doc_id}"
            self.generated_doc_ids.append(request.doc_id)
            self.add_request_response_to_cache(request, response)
            generated_responses.append(response)
        return self.merge_cached_and_new_responses(
            original_requests,
            cached_responses,
            pending_requests,
            generated_responses,
        )

    def generate_until_multi_round(self, requests):
        raise NotImplementedError


class RequestCacheResumeTest(unittest.TestCase):
    def test_resume_only_generates_uncached_requests(self):
        requests = [make_request(1), make_request(2), make_request(3)]

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            model_module,
            "LMMS_EVAL_USE_CACHE",
            "True",
        ), mock.patch.object(model_module, "LMMS_EVAL_HOME", temp_dir):
            first_model = DummyResumeLm()
            first_model.add_request_response_to_cache(requests[0], "cached-1")

            resumed_model = DummyResumeLm()
            resumed_outputs = resumed_model.generate_until(requests)

            self.assertEqual(resumed_outputs, ["cached-1", "generated-2", "generated-3"])
            self.assertEqual(resumed_model.generated_doc_ids, [2, 3])

            third_model = DummyResumeLm()
            cached_outputs = third_model.generate_until(requests)

            self.assertEqual(cached_outputs, ["cached-1", "generated-2", "generated-3"])
            self.assertEqual(third_model.generated_doc_ids, [])

            cache_dir = Path(third_model.get_model_cache_dir)
            cache_files = list(cache_dir.glob("*.jsonl"))
            self.assertEqual(len(cache_files), 1)


if __name__ == "__main__":
    unittest.main()
