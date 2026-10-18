"""Synthetic technical fixtures only; never evidence of human pilot collection."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
import numpy as np
from training.drawai.private_data import (audit,build,canonical,dump,plan_content,read_plan,schedule,validate_session,new_output,ROOT)


def session(pid='P001',count=2):
    stage='pilot' if pid.startswith('P') else 'main';sid='session_'+str(uuid4())
    now=(datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat()
    result=dict(schema_version='drawai-private-session-v1',protocol_version='drawai-private-protocol-v1',
        session_id=sid,participant_id=pid,stage=stage,input_device='mouse',started_at=now,
        consent=dict(version='drawai-consent-v1',record_ref='C-'+pid,confirmed_at=now,training=True,evaluation=True,kaggle_publication=True),samples=[])
    for i,task in enumerate(schedule(pid,stage)[:count]):
        strokes=[[dict(x=30+i,y=50),dict(x=130,y=150+i),dict(x=220+i,y=270)]]
        result['samples'].append(dict(schema_version='drawai-private-sample-v1',protocol_version=result['protocol_version'],
            sample_id='sample_'+str(uuid4()),participant_id=pid,session_id=sid,stage=stage,
            label=task['label'],repetition=task['repetition'],input_device='mouse',actual_pointer_types=['mouse'],
            source_kind='human-collected',collected_at=now,
            drawing=dict(schema_version='drawai-strokes-v1',coordinate_system='canvas-y-down',canvas=dict(width=400,height=400),
                strokes=strokes,reference_label=task['label'],source=dict(kind='user'))))
    return result


class PrivateDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.inbox=self.root/'inbox';self.inbox.mkdir();self.plan=plan_content()
    def put(self,value,name='session.json'):dump(self.inbox/name,value)
    def test_plan_writer_disjoint_quota_and_hash_guard(self):
        people=self.plan['participants'];self.assertEqual(len(people),35)
        self.assertEqual({split:sum(p['split']==split for p in people) for split in ['pilot-dev','train','validation','test']},
                         {'pilot-dev':5,'train':20,'validation':5,'test':5})
        p=self.root/'plan.json';dump(p,self.plan);self.assertEqual(read_plan(p),self.plan)
        bad=deepcopy(self.plan);bad['participants'][-1]['split']='train';dump(p,bad)
        with self.assertRaisesRegex(ValueError,'locked P3'):read_plan(p)
    def test_schedule_all_digits_per_repetition_and_reproducible(self):
        for pid,stage,repeats in [('P001','pilot',5),('M030','main',20)]:
            tasks=schedule(pid,stage);self.assertEqual(len(tasks),repeats*10)
            self.assertEqual(tasks,schedule(pid,stage))
            for repetition in range(1,repeats+1):self.assertEqual(sorted(t['label'] for t in tasks if t['repetition']==repetition),list(range(10)))
        self.assertNotEqual(schedule('P001','pilot')[:10],schedule('P005','pilot')[:10])
    def test_valid_partial_build_has_same_sample_ids_and_hashes(self):
        value=session();self.put(value);out=self.root/'build';report=build(self.inbox,self.plan,'pilot',out)
        self.assertEqual(report['accepted_samples'],2);self.assertFalse(report['quota_complete']);self.assertFalse(report['dataset_locked'])
        import csv
        with (out/'manifest.csv').open(encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
        for row,original in zip(rows,value['samples']):
            self.assertEqual(row['sample_id'],original['sample_id']);self.assertEqual(row['split'],'pilot-dev')
            for field,hashkey in [('raw_json','raw_sha256'),('png','png_sha256'),('swin_npy','swin_sha256')]:
                self.assertEqual(hashlib.sha256((out/row[field]).read_bytes()).hexdigest(),row[hashkey])
            self.assertEqual(np.load(out/row['swin_npy'],allow_pickle=False).shape,(3,128))
        self.assertNotIn('record_ref',(out/rows[0]['raw_json']).read_text())
    def test_empty_inbox_no_fake_pilot_and_incomplete_build_rejected(self):
        report,_,_=audit(self.inbox,self.plan,'pilot');self.assertEqual(report['accepted_samples'],0)
        self.assertFalse(report['quota_complete']);self.assertEqual(report['warnings'][0]['code'],'NO_HUMAN_DATA')
        with self.assertRaisesRegex(ValueError,'No samples'):build(self.inbox,self.plan,'pilot',self.root/'empty')
        self.put(session())
        with self.assertRaisesRegex(ValueError,'Quota incomplete'):build(self.inbox,self.plan,'pilot',self.root/'quota',True)
    def test_unknown_pii_consent_label_device_time_and_nonfinite_rejected(self):
        original=session( count=1)
        changes=[lambda v:v.update(email='unwanted@example.test'),lambda v:v['consent'].update(training=False),
            lambda v:v['samples'][0].update(label=(v['samples'][0]['label']+1)%10),
            lambda v:v['samples'][0].update(actual_pointer_types=['pen']),
            lambda v:v['samples'][0].update(collected_at='2999-01-01T00:00:00Z'),
            lambda v:v['samples'][0]['drawing']['strokes'][0][0].update(x=float('nan')),
            lambda v:v['samples'][0]['drawing']['strokes'][0][0].update(t=0)]
        for change in changes:
            with self.subTest(change=change):
                value=deepcopy(original);change(value)
                with self.assertRaises((ValueError,TypeError)):validate_session(value,self.plan)
    def test_duplicate_sample_strokes_and_snapshot_are_errors(self):
        original=session();original['samples'][1]['drawing']['strokes']=deepcopy(original['samples'][0]['drawing']['strokes']);self.put(original)
        report,_,_=audit(self.inbox,self.plan,'pilot');self.assertIn('Exact duplicate',report['errors'][0]['error'])
        self.assertEqual(report['accepted_samples'],0)
        self.put(session(count=1));self.put(json.loads((self.inbox/'session.json').read_text()),'snapshot.json')
        report,_,_=audit(self.inbox,self.plan,'pilot');self.assertTrue(any('Duplicate session' in e['error'] for e in report['errors']))
    def test_cross_writer_exact_duplicates_cannot_hide_in_other_split(self):
        a=session('M001',1);b=session('M026',1)
        b['samples'][0]['drawing']['strokes']=deepcopy(a['samples'][0]['drawing']['strokes'])
        self.put(a,'train.json');self.put(b,'test.json');report,_,_=audit(self.inbox,self.plan,'main')
        self.assertTrue(any('Exact duplicate' in e['error'] for e in report['errors']))
    def test_pilot_main_not_mix_and_no_quota_reset(self):
        self.put(session('M001',1));report,_,_=audit(self.inbox,self.plan,'pilot');self.assertTrue(report['errors'])
        self.put(session('P001',1));self.put(session('P001',1),'other-session.json')
        report,_,_=audit(self.inbox,self.plan,'pilot');self.assertTrue(any('Multiple sessions' in e['error'] for e in report['errors']))
    def test_near_duplicate_is_review_flag_not_model_based_exclusion(self):
        a=session(count=2);a['samples'][1]['drawing']['strokes']=[[dict(x=p['x']+1,y=p['y']+1) for p in s] for s in a['samples'][0]['drawing']['strokes']]
        self.put(a);report,_,_=audit(self.inbox,self.plan,'pilot')
        self.assertFalse(report['errors']);self.assertEqual(report['accepted_samples'],2)
        self.assertTrue(report['near_duplicates']);self.assertTrue(report['warnings'])
    def test_outputs_never_overwrite_or_enter_public(self):
        out=self.root/'build';out.mkdir();(out/'keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError,'never overwrite'):new_output(out)
        with self.assertRaisesRegex(ValueError,'public/dist'):new_output(ROOT/'web/public/private-test-forbidden')

if __name__=='__main__':unittest.main()
