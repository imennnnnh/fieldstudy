"""HTTP 串接測試：背景工作、固定條件、補測上下載、輸入阻擋。"""
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from starlette.testclient import TestClient

import app as server
from core import predictor
from core.config import FEATURE_COLUMNS, SHOT_COLUMNS
from core.reverse_search import build_search_grid


class ApiModel:
    classes_ = np.array(['(4,3)', '未過網'])
    def predict_proba(self, frame):
        probability = np.where(frame['出球強度'].to_numpy() == 1, 1, 0.9)
        return np.column_stack([probability, 1-probability])
    def predict(self, frame):
        return self.classes_[np.argmax(self.predict_proba(frame), axis=1)]


def csv_fixture():
    data = pd.DataFrame([[14,5,30,50], [15,5,30,50]], columns=FEATURE_COLUMNS)
    data.insert(0, '測試編號', [1,2])
    for col in SHOT_COLUMNS: data[col] = ['(4,3)', '(4,4)']
    return data.to_csv(index=False).encode('utf-8-sig')


class ApiTests(unittest.TestCase):
    def setUp(self):
        server.jobs.clear()
        self.old = dict(predictor._state)
        predictor._state.update(package={
            'best_model': ApiModel(), 'best_model_name':'test-only', 'class_names':['(4,3)','未過網'],
            'num_classes':2,'tested_parameter_count':2,'training_info':{},
        }, class_to_index={'(4,3)':0,'未過網':1})
        observed = pd.read_csv(io.BytesIO(csv_fixture())).melt(id_vars=FEATURE_COLUMNS, value_vars=SHOT_COLUMNS, value_name='落點')
        observed.loc[observed['落點'] == '(4,4)', '落點'] = '未過網'
        predictor._state['package'].update(training_observations=observed,
            tested_parameters={(14,5,30,50), (15,5,30,50)}, training_source_files=['模型訓練資料.xlsx'])
        self.temp = tempfile.TemporaryDirectory()
        self.output_patch = patch.object(server, 'OUTPUT_DIR', Path(self.temp.name)); self.output_patch.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        predictor._state.update(self.old); self.output_patch.stop(); self.temp.cleanup(); server.jobs.clear()

    def wait_job(self, job_id):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            response = self.client.get(f'/api/job/{job_id}'); self.assertEqual(response.status_code,200)
            job = response.json()['job']
            if job['status'] != 'running':
                self.assertEqual(job['status'],'done',job.get('error')); return job['result']
            time.sleep(0.01)
        self.fail('背景工作逾時')

    def test_page_static_model_and_prediction(self):
        self.assertIn('推薦補測資料',self.client.get('/').text)
        for path in ['/static/app.js','/static/styles.css']:
            self.assertEqual(self.client.get(path).status_code,200)
        self.assertTrue(self.client.get('/api/model-info').json()['info']['has_model'])
        data=self.client.post('/api/predict',json={'elevation':14,'horizontal':5,'spin':30,'power':1}).json()
        self.assertEqual(data['prediction'],'(4,3)'); self.assertEqual(data['top'][0]['預測機率'],1)
        for body in [[],{'elevation':True}, {'elevation':1.5,'horizontal':5,'spin':30,'power':1}]:
            self.assertEqual(self.client.post('/api/predict',json=body).status_code,400)

    def test_api_cannot_lower_threshold(self):
        ranges={c:[1,2] for c in FEATURE_COLUMNS}
        grid=build_search_grid(ranges)
        with patch('core.reverse_search.build_search_grid',return_value=grid), patch('core.multi_route.build_search_grid',return_value=grid):
            for route,extra in [('/api/reverse-search', {'top_n':100}),('/api/multi-route',{'routes':['高速球'],'n_solutions':20})]:
                response=self.client.post(route,json={'targets':['(4,3)'],'min_target_probability':0,'max_invalid_probability':1,**extra})
                self.assertEqual(response.status_code,200)
                result=self.wait_job(response.json()['job_id'])
                rows=result['rows'] if 'rows' in result else result['routes'][0]['rows']
                self.assertTrue(rows); self.assertTrue(all(row['目標落點機率']==1 and row['無效總機率']==0 and row['出球強度']==1 for row in rows))
        for count in [0,1.5,101,True]:
            self.assertEqual(self.client.post('/api/reverse-search',json={'targets':['(4,3)'],'top_n':count}).status_code,400)
        self.assertEqual(self.client.post('/api/reverse-search',json={'targets':['(4,3)','(4,4)']}).status_code,400)

    def test_sampling_uses_model_training_data_and_downloads_300(self):
        self.assertTrue(self.client.post('/api/sampling',json={'generate':False}).json()['skipped'])
        # 頁面不再上傳檔案，API 也不能指定其他資料來源。
        self.assertEqual(self.client.post('/api/sampling',json={'generate':True,'files':['other.xlsx']}).status_code,400)
        with patch.dict(predictor._state, {'package':None}):
            self.assertEqual(self.client.post('/api/sampling',json={'generate':True}).status_code,400)
        info=self.client.get('/api/model-info').json()['info']['sampling']
        self.assertTrue(info['ready']); self.assertEqual(info['source_files'], ['模型訓練資料.xlsx'])
        response=self.client.post('/api/sampling',json={'generate':True})
        self.assertEqual(response.status_code,200)
        result=self.wait_job(response.json()['job_id'])
        self.assertEqual(len(result['rows']),300)
        self.assertEqual(result['summary']['source_files'], ['模型訓練資料.xlsx'])
        csv=self.client.get(result['downloads']['csv']); xlsx=self.client.get(result['downloads']['xlsx'])
        self.assertEqual(csv.status_code,200); self.assertEqual(xlsx.status_code,200)
        csv_df=pd.read_csv(io.BytesIO(csv.content)); excel_df=pd.read_excel(io.BytesIO(xlsx.content),sheet_name='推薦補測300組')
        self.assertEqual(len(csv_df),300); pd.testing.assert_frame_equal(csv_df,excel_df)
        proposed=set(csv_df[FEATURE_COLUMNS].itertuples(index=False,name=None))
        self.assertFalse(proposed & predictor.get_tested_parameters())
        self.assertTrue(csv_df[SHOT_COLUMNS].isna().all().all())
        self.assertEqual(self.client.get('/api/sampling/unknown/download/csv').status_code,404)
        self.assertEqual(self.client.get(result['downloads']['csv'].replace('/csv','/txt')).status_code,404)

    def test_upload_train_full_pipeline(self):
        import train_model
        folder = Path(self.temp.name)
        data_dir = folder / 'data'; data_dir.mkdir()
        model_path = folder / 'model.pkl'
        rows = pd.DataFrame([[i,5,30,i*5] for i in range(1,11)], columns=FEATURE_COLUMNS)
        rows.insert(0, '測試編號', range(1,11))
        for col in SHOT_COLUMNS:
            rows[col] = ['(4,3)']*5 + ['未過網']*5
        buffer = io.BytesIO(); rows.to_excel(buffer, index=False)
        with patch.object(server, 'DATA_DIR', data_dir), patch.object(train_model, 'MODEL_PATH', model_path), patch.object(train_model, 'OUTPUT_DIR', folder), patch.object(predictor, 'MODEL_PATH', model_path):
            response = self.client.post('/api/upload-train', data={'replace_old':'true'}, files={'files':('synthetic.xlsx',buffer.getvalue(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
            self.assertEqual(response.status_code, 200)
            result = self.wait_job(response.json()['job_id'])
            self.assertEqual(result['total_samples'], 50)
            self.assertEqual(result['total_groups'], 10)
            self.assertEqual(predictor.get_best_model().n_estimators, 500)
            self.assertTrue(model_path.exists())
            self.assertTrue((folder / 'cv_summary.csv').exists())
            self.assertTrue(self.client.get('/api/model-info').json()['info']['has_model'])
            old = (data_dir/'synthetic.xlsx').read_bytes()
            bad = self.client.post('/api/upload-train', data={'replace_old':'true'}, files={'files':('broken.xlsx',b'not excel','application/octet-stream')})
            self.assertEqual(bad.status_code,400)
            self.assertEqual((data_dir/'synthetic.xlsx').read_bytes(), old)
            # 新模型保存訓練快照；原始 Excel 移走並重新載入模型後也能產生補測。
            (data_dir/'synthetic.xlsx').unlink()
            self.assertTrue(predictor.load_model_from_disk())
            self.assertEqual(len(predictor.get_training_observations()), 50)
            response=self.client.post('/api/sampling',json={'generate':True})
            self.assertEqual(response.status_code,200)
            result=self.wait_job(response.json()['job_id'])
            self.assertEqual(len(result['rows']),300)
            self.assertEqual(result['summary']['excluded_count'],10)

    def test_job_lock(self):
        job=server.create_job('test')
        response=self.client.post('/api/reverse-search',json={'targets':['(4,3)']})
        self.assertEqual(response.status_code,409)
        server.update_job(job,status='done',finished_at=time.time())

if __name__=='__main__': unittest.main()
