"""補測資料必須來自目前模型：快照、舊檔遷移與排除訓練參數。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core import predictor
from core.config import FEATURE_COLUMNS, SHOT_COLUMNS
from core.data_loader import prepare_training_data
from core.sampling import generate_from_training_data


def measured_frame():
    frame = pd.DataFrame([[14,5,30,50],[15,5,30,50]], columns=FEATURE_COLUMNS)
    frame.insert(0,'測試編號',[1,2])
    for col in SHOT_COLUMNS:
        frame[col] = ['(4,3)','(4,4)']
    # 與訓練一致：空白球不會被補成假的落點。
    frame.loc[0,SHOT_COLUMNS[0]] = ''
    return frame


class TrainingSamplingTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(predictor._state)
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'原訓練資料.xlsx'
        measured_frame().to_excel(self.path,index=False)
        self.data = prepare_training_data([self.path])
        self.package = {
            'tested_parameters':self.data['tested_parameters'],
            'training_info':{'total_samples':len(self.data['y'])},
            'class_names':list(self.data['y'].unique()),
        }
        predictor._state['package'] = self.package
        self.folder_patch = patch.object(predictor,'DATA_DIR',Path(self.folder.name))
        self.folder_patch.start(); self.addCleanup(self.folder_patch.stop)
    def tearDown(self):
        predictor._state.update(self.old)

    def test_legacy_model_reuses_matching_training_files(self):
        result = predictor.get_training_observations()
        self.assertEqual(len(result),9)
        self.assertEqual(self.package['training_source_files'], ['原訓練資料.xlsx'])
        self.assertEqual(self.package['sampling_source_kind'],'legacy_files')
        self.path.unlink()
        # 同一服務內已核對並緩存的快照不受後續資料檔變動影響。
        pd.testing.assert_frame_equal(result,predictor.get_training_observations())

    def test_mismatched_legacy_files_block_sampling(self):
        self.package['training_info']['total_samples'] = 500
        with self.assertRaisesRegex(ValueError,'不一致'):
            predictor.get_training_observations()
        self.assertFalse(predictor.get_sampling_info()['ready'])

    def test_missing_legacy_files_block_sampling(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError,'沒有原訓練檔'):
            predictor.get_training_observations()

    def test_snapshot_is_independent_of_current_data_folder(self):
        snapshot=self.data['training_observations']
        self.package['training_observations']=snapshot.copy()
        self.path.write_bytes(b'file changed after model training')
        result=predictor.get_training_observations()
        pd.testing.assert_frame_equal(result,snapshot)
        result.loc[result.index[0],'落點']='未過網'
        pd.testing.assert_frame_equal(self.package['training_observations'],snapshot)

    def test_partial_shots_and_unobserved_uploaded_settings(self):
        snapshot=self.data['training_observations']
        excluded=self.data['tested_parameters'] | {(28,9,59,99)}
        result,_,summary=generate_from_training_data(snapshot,tested_parameters=excluded,random_state=42)
        self.assertEqual(len(result),300)
        self.assertEqual(summary['observed_shots'],9)
        self.assertEqual(summary['excluded_count'],3)
        self.assertFalse(excluded & set(result[FEATURE_COLUMNS].itertuples(index=False,name=None)))
        self.assertEqual(len(result[FEATURE_COLUMNS].drop_duplicates()),300)


if __name__ == '__main__': unittest.main()
