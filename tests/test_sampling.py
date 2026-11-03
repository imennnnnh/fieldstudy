"""合成資料只用於測試；不交付虛構實測結果。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from core.config import FEATURE_COLUMNS, SHOT_COLUMNS, IMPORT_REQUIRED_COLUMNS, PARAMETER_LIMITS
from core.sampling import generate_supplement, load_complete_observations

def complete_data(boundary=True):
    rows=[[14,5,30,50],[15,5,30,50],[1,1,1,1],[28,9,59,99]] if boundary else [[1,1,1,1],[28,9,59,99]]
    data=pd.DataFrame(rows,columns=FEATURE_COLUMNS); data.insert(0,'測試編號',range(1,len(data)+1))
    for col in SHOT_COLUMNS: data[col]='(4,3)'
    if boundary: data.loc[1,SHOT_COLUMNS]='(4,4)'
    return data
class SamplingTests(unittest.TestCase):
    def assert_valid_output(self,source,result):
        self.assertEqual(list(result.columns),IMPORT_REQUIRED_COLUMNS); self.assertEqual(len(result),300)
        self.assertEqual(result['測試編號'].nunique(),300); self.assertEqual(len(result[FEATURE_COLUMNS].drop_duplicates()),300)
        self.assertFalse(set(source[FEATURE_COLUMNS].itertuples(index=False,name=None)) & set(result[FEATURE_COLUMNS].itertuples(index=False,name=None)))
        self.assertTrue((result[SHOT_COLUMNS]=='').all().all())
        for col,(low,high) in PARAMETER_LIMITS.items():
            self.assertTrue(result[col].between(low,high).all()); self.assertTrue((result[col]%1==0).all())
    def test_300_disjoint_integer_settings_and_allocation(self):
        source=complete_data(); result,strategy,s=generate_supplement(source,random_state=17)
        self.assert_valid_output(source,result); self.assertEqual(s['boundary_pair_count'],1); self.assertEqual(s['random_count'],30)
        self.assertEqual(s['spatial_count']+s['boundary_count'],270); self.assertEqual(len(strategy),300)
        self.assertEqual(s['spatial_count'],150+s['boundary_fallback_count']); self.assertLessEqual(s['boundary_count'],120)
    def test_no_boundary_falls_back_to_exploration(self):
        source=complete_data(False); result,_,s=generate_supplement(source,random_state=2)
        self.assert_valid_output(source,result); self.assertEqual(s['spatial_count'],270); self.assertEqual(s['boundary_count'],0); self.assertEqual(s['random_count'],30)
    def test_excel_csv_input_and_alias(self):
        source=complete_data(); source.loc[0,SHOT_COLUMNS[0]]='其他無效'
        with tempfile.TemporaryDirectory() as folder:
            xlsx,csv=Path(folder)/'sample.xlsx',Path(folder)/'sample.csv'
            source.to_excel(xlsx,index=False); source.to_csv(csv,index=False,encoding='utf-8-sig')
            loaded=load_complete_observations([xlsx,csv])
        self.assertEqual(len(loaded),8); self.assertEqual(loaded.loc[0,SHOT_COLUMNS[0]],'碰桌後彈出')
        self.assert_valid_output(source,generate_supplement(loaded,random_state=2)[0])
    def test_missing_unknown_and_invalid_parameters_blocked(self):
        for value in ['', '未知',np.nan]:
            data=complete_data(); data.loc[0,SHOT_COLUMNS[0]]=value
            with self.assertRaises(ValueError): generate_supplement(data)
        for value in [0,29,1.5,np.nan]:
            data=complete_data(); data['仰角角度']=data['仰角角度'].astype(float); data.loc[0,'仰角角度']=value
            with self.assertRaises(ValueError): generate_supplement(data)
        with self.assertRaises(ValueError): load_complete_observations([])
    def test_reproducible_with_seed(self):
        a=generate_supplement(complete_data(),random_state=42)[0]; b=generate_supplement(complete_data(),random_state=42)[0]
        pd.testing.assert_frame_equal(a,b)
if __name__=='__main__': unittest.main()
