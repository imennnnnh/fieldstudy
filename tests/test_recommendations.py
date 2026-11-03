"""固定機率條件、四維搜尋與跨次推薦的回歸測試。"""
import unittest
import numpy as np
import pandas as pd
from core import predictor
from core.config import FEATURE_COLUMNS
from core.reverse_search import filter_accurate_candidates, search_best_parameters, normalize_target_labels, build_search_grid
from core.multi_route import recommend_multi_route, _history

class FakeModel:
    classes_ = np.array(['(4,3)', '(1,1)', '未過網', '其他無效'])
    def __init__(self, exact=True): self.exact = exact
    def predict_proba(self, frame):
        result = np.zeros((len(frame), 4))
        good = frame['出球強度'].to_numpy() <= 4
        result[:, 0] = np.where(good, 1 if self.exact else 0.999999999999, 0.7)
        result[:, 2] = 1 - result[:, 0]
        return result

class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(predictor._state)
        predictor._state.update(package={'best_model': FakeModel()}, class_to_index={label: i for i, label in enumerate(FakeModel.classes_)})
        self.ranges = {c: range(1, 6) for c in FEATURE_COLUMNS}
        _history.clear()
    def tearDown(self):
        predictor._state.update(self.old); _history.clear()
    def test_no_rounding_and_no_lowering_threshold(self):
        rows = pd.DataFrame({'命中目標':[1]*5, '目標落點機率':[1,0.999999999999,1,1,0.8], '無效總機率':[0,0,1e-15,0,0], '非目標總機率':[0,1e-12,1e-15,1e-15,0.2]})
        self.assertEqual(filter_accurate_candidates(rows, 0.1, 0.99).index.tolist(), [0])
    def test_all_four_dimensions_are_searched(self):
        result, _, count = search_best_parameters('(4,3)', top_n=100, grid_ranges=self.ranges, chunk_size=17)
        self.assertEqual(count, 625); self.assertEqual(len(result), 100)
        self.assertTrue((result['目標落點機率'] == 1).all()); self.assertTrue((result['無效總機率'] == 0).all())
        self.assertEqual(set(result['出球強度']), {1,2,3,4})
        grid = build_search_grid(self.ranges)
        self.assertTrue(all(grid[c].nunique() == 5 for c in FEATURE_COLUMNS))
    def test_no_fallback_even_if_legacy_arguments_lower_threshold(self):
        predictor._state['package']['best_model'] = FakeModel(exact=False)
        result, status, _ = search_best_parameters('(4,3)', grid_ranges=self.ranges, min_target_probability=0.1, max_invalid_probability=1)
        self.assertTrue(result.empty); self.assertIn('沒有符合條件', status)
        results, _, _ = recommend_multi_route('(4,3)', ['高速球','均衡方案'], grid_ranges=self.ranges, min_target_probability=0, max_invalid_probability=1)
        self.assertTrue(all(df.empty for df in results.values()))
    def test_multi_new_combinations_and_strict_rules(self):
        routes = ['高速球','低速球','多樣化方案']
        first, _, count = recommend_multi_route('(4,3)', routes, n_solutions=5, grid_ranges=self.ranges, random_state=1, chunk_size=57)
        second, _, _ = recommend_multi_route('(4,3)', routes, n_solutions=5, grid_ranges=self.ranges, random_state=2, chunk_size=57)
        self.assertEqual(count, 625)
        for condition in first:
            a,b=first[condition],second[condition]
            self.assertEqual(len(a),5); self.assertEqual(len(b),5)
            self.assertFalse(set(a[FEATURE_COLUMNS].itertuples(index=False,name=None)) & set(b[FEATURE_COLUMNS].itertuples(index=False,name=None)))
            self.assertTrue((b['目標落點機率']==1).all()); self.assertTrue((b['無效總機率']==0).all())
        self.assertGreater(first['高速球']['出球強度'].mean(), first['低速球']['出球強度'].mean())
    def test_single_target_only(self):
        for value in [[],['(4,3)','(4,4)'],['(4,3)','(4,3)'],'(6,2)']:
            with self.assertRaises(ValueError): normalize_target_labels(value)
    def test_reused_only_when_all_candidates_used(self):
        tiny={c:[1] for c in FEATURE_COLUMNS}
        first,_,_=recommend_multi_route('(4,3)',['高速球'],n_solutions=5,grid_ranges=tiny)
        second,status,_=recommend_multi_route('(4,3)',['高速球'],n_solutions=5,grid_ranges=tiny)
        self.assertEqual(len(first['高速球']),1); self.assertEqual(len(second['高速球']),1); self.assertIn('曾在',status['高速球'])
if __name__=='__main__': unittest.main()
