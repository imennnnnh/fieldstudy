"""Excel 資源回歸測試：不依賴垃圾回收，驗證可立即清理暫存檔。"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.config import FEATURE_COLUMNS, SHOT_COLUMNS
from core.data_loader import load_excel_files, prepare_training_data
from core.sampling import load_complete_observations


def observations():
    frame = pd.DataFrame([[14, 5, 30, 50]], columns=FEATURE_COLUMNS)
    frame.insert(0, '測試編號', [1])
    for col in SHOT_COLUMNS:
        frame[col] = '(4,3)'
    return frame


class ExcelResourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / '第1輪_智慧補測推薦_300組_弱落點優先.xlsx'
        self.handles = []
        self.readers = []
        self.real_excel_file = pd.ExcelFile

    def track_reader(self, *args, **kwargs):
        reader = self.real_excel_file(*args, **kwargs)
        self.readers.append(reader)
        self.handles.append(reader._reader.handles.handle)
        self.addCleanup(reader.close)
        return reader

    def assert_all_closed(self):
        self.assertTrue(self.handles, '測試必須實際開啟工作簿')
        self.assertTrue(all(handle.closed for handle in self.handles), '讀檔返回或拋錯前必須關閉檔案')

    def test_multiple_sheets_reuse_one_reader_and_close(self):
        with pd.ExcelWriter(self.path) as writer:
            observations().to_excel(writer, sheet_name='第一組', index=False)
            observations().to_excel(writer, sheet_name='第二組', index=False)
            pd.DataFrame().to_excel(writer, sheet_name='空白', index=False)
        with patch('core.data_loader.pd.ExcelFile', side_effect=self.track_reader):
            frame, report = load_excel_files([self.path])
        self.assertEqual(len(frame), 2)
        self.assertEqual(len(report), 3)
        self.assertEqual(len(self.readers), 1)
        self.assert_all_closed()
        self.path.unlink()  # Windows 上也必須可立即刪除。

    def test_missing_columns_close_before_validation_error(self):
        pd.DataFrame({'不正確欄位': [1]}).to_excel(self.path, index=False)
        with patch('core.data_loader.pd.ExcelFile', side_effect=self.track_reader):
            with self.assertRaises(ValueError):
                load_excel_files([self.path])
        self.assert_all_closed()

    def test_sheet_read_error_closes_workbook(self):
        observations().to_excel(self.path, index=False)
        with patch('core.data_loader.pd.ExcelFile', side_effect=self.track_reader), patch.object(self.real_excel_file, 'parse', side_effect=ValueError('模擬讀取失敗')):
            with self.assertRaises(ValueError):
                load_excel_files([self.path])
        self.assert_all_closed()

    def test_invalid_parameter_does_not_retain_handle_in_traceback(self):
        frame = observations(); frame['仰角角度'] = 29
        frame.to_excel(self.path, index=False)
        with patch('core.data_loader.pd.ExcelFile', side_effect=self.track_reader):
            with self.assertRaises(ValueError):
                prepare_training_data([self.path])
        self.assert_all_closed()

    def test_supplement_reader_closes_before_incomplete_data_error(self):
        frame = observations(); frame[SHOT_COLUMNS[0]] = ''
        frame.to_excel(self.path, index=False)
        with patch('core.sampling.pd.ExcelFile', side_effect=self.track_reader):
            with self.assertRaises(ValueError):
                load_complete_observations([self.path])
        self.assert_all_closed()

    def test_temp_cleanup_under_windows_style_delete_lock(self):
        # POSIX 可刪除開啟中的檔案；此 guard 模擬 Windows 禁止該操作。
        real_unlink = os.unlink
        filename = self.path.name
        def windows_unlink(path, *args, **kwargs):
            if os.path.basename(os.fsdecode(path)) == filename and any(not h.closed for h in self.handles):
                raise PermissionError(32, '模擬 WinError 32：檔案仍被使用', os.fsdecode(path))
            return real_unlink(path, *args, **kwargs)
        with patch('core.data_loader.pd.ExcelFile', side_effect=self.track_reader), patch('os.unlink', side_effect=windows_unlink):
            with tempfile.TemporaryDirectory(prefix='fieldstudy-check-') as folder:
                source = Path(folder) / filename
                observations().to_excel(source, index=False)
                prepare_training_data([source])
            self.assertFalse(Path(folder).exists())
        self.assert_all_closed()


if __name__ == '__main__':
    unittest.main()
