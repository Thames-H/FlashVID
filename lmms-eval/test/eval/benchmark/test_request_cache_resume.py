import unittest

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


class _DummyCacheModel(lmms):
    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError

    def generate_until_multi_round(self, requests):
        raise NotImplementedError


def _make_request(doc_id: int, idx: int = 0):
    return Instance(
        request_type="generate_until",
        arguments=("ctx", {"max_new_tokens": 16}, None, doc_id, "videomme", "validation"),
        idx=idx,
        metadata={"task": "videomme", "doc_id": doc_id, "repeats": 1},
    )


class RequestCacheResumeTest(unittest.TestCase):
    def test_partition_loaded_cache_requests_splits_cached_and_pending(self):
        model = _DummyCacheModel()
        model.cache_dict["videomme"][1] = "cached-1"

        req_1 = _make_request(1, idx=0)
        req_2 = _make_request(2, idx=1)

        cached, pending = model.partition_loaded_cache_requests([req_1, req_2])

        self.assertEqual(cached, {("videomme", 1): "cached-1"})
        self.assertEqual(pending, [req_2])

    def test_merge_cached_and_generated_responses_restores_original_order(self):
        model = _DummyCacheModel()
        req_1 = _make_request(1, idx=0)
        req_2 = _make_request(2, idx=1)
        req_3 = _make_request(3, idx=2)

        merged = model.merge_cached_and_generated_responses(
            [req_1, req_2, req_3],
            cached_responses={
                ("videomme", 1): "cached-1",
                ("videomme", 3): "cached-3",
            },
            generated_responses={
                ("videomme", 2): "new-2",
            },
        )

        self.assertEqual(merged, ["cached-1", "new-2", "cached-3"])


if __name__ == "__main__":
    unittest.main()
