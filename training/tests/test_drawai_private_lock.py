"""P4 tests use a miniature synthetic policy in temporary folders only.

Production read_plan still rejects this reduced policy; no pilot/main data claim.
"""
from copy import deepcopy
from datetime import datetime,timezone
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid5,NAMESPACE_URL
import zipfile

from training.drawai import private_dataset as ds
from training.drawai.private_data import plan_content,canonical,sha,dump,load,audit
from training.tests.test_drawai_private import session


class PrivateLockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.pilot=self.root/'pilot';self.main=self.root/'main';self.pilot.mkdir();self.main.mkdir()
        self.policy=plan_content();self.policy['participants']=[p for p in self.policy['participants'] if p['participant_id'] in ['P001','M001','M021','M026']]
        for p in self.policy['participants']:p['samples_per_class']=1
        self.policy['target_pilot_samples']=10;self.policy['target_main_samples']=30
        self.policy.pop('plan_sha256');self.policy['plan_sha256']=sha(canonical(self.policy))
        patched=patch('training.drawai.private_dataset.read_plan',return_value=self.policy);patched.start();self.addCleanup(patched.stop)
        self.serial=0
    def capture(self,pid,count=10):
        value=session(pid,count);seed=int(pid[1:])+(500 if pid.startswith('M') else 0)
        for i,s in enumerate(value['samples']):
            s['drawing']['strokes']=[[dict(x=20+seed%50+i*.17,y=30+(seed%70)*.2),
                dict(x=100+i*.3+seed*.02,y=130+seed*.03+i*.23),dict(x=250,y=300+seed*.01)]]
        dump((self.pilot if pid.startswith('P') else self.main)/(pid+'.json'),value)
        return value
    def signed(self,stage,approval=None):
        self.serial+=1;out=self.root/f'review-{self.serial}'
        ds.review_template(self.pilot if stage=='pilot' else self.main,stage,out,pilot_approval=approval)
        review=load(out/'review.json');review['reviewer_ref']='R-SYNTHETIC-TEST';review['reviewed_at']=datetime.now(timezone.utc).isoformat()
        review['confirmations']={key:True for key in ds.CONFIRMATIONS}
        for person in review['participants']:
            person.update(person_token='person_'+uuid5(NAMESPACE_URL,'SYNTHETIC-P4-'+person['participant_id']).hex,
                          human_origin_verified=True,consent_record_verified=True)
        for group in review['near_groups']:group.update(decision='keep_distinct_verified',reason='Synthetic fixture review for automated technical test only.')
        dump(out/'review.json',review);return out/'review.json'
    def approved(self):
        self.capture('P001');review=self.signed('pilot');out=self.root/'approval'
        ds.approve_pilot(self.pilot,review,out);return out/'pilot-approval.json'
    def ready(self):
        approval=self.approved()
        for pid in ['M001','M021','M026']:self.capture(pid)
        return self.signed('main',approval),approval
    def test_empty_inbox_generates_unsigned_review_and_cannot_lock(self):
        out=self.root/'empty-review';v=ds.review_template(self.main,'main',out)
        self.assertTrue(all(flag is False for flag in v['confirmations'].values()))
        self.assertFalse(load(out/'readiness.json')['dataset_locked'])
        with self.assertRaisesRegex(ValueError,'Full planned quota'):ds.lock(self.main,out/'review.json',self.root/'absent',self.root/'lock')
        self.assertFalse((self.root/'lock').exists())
    def test_positive_lock_has_integrity_sealed_test_and_common_representations(self):
        review,approval=self.ready();out=self.root/'locked';record=ds.lock(self.main,review,approval,out)
        verified,rows=ds.verify_lock(out,record['lock_sha256']);self.assertEqual(verified,record)
        self.assertEqual(len(rows),20);self.assertEqual({r['split'] for r in rows},{'train','validation'})
        with zipfile.ZipFile(out/'heldout-test.zip') as archive:
            tests=list(csv.DictReader(io.StringIO(archive.read('manifest.csv').decode())))
            self.assertEqual(len(tests),10);self.assertEqual({r['split'] for r in tests},{'test'})
            self.assertFalse({r['sample_id'] for r in rows}&{r['sample_id'] for r in tests})
            for r in tests:
                for key,hkey in [('raw_json','raw_sha256'),('png','png_sha256'),('swin_npy','swin_sha256')]:self.assertEqual(sha(archive.read(r[key])),r[hkey])
        raw=load(out/'development'/rows[0]['raw_json'])
        self.assertNotIn('collected_at',raw);self.assertNotIn('session_id',raw);self.assertNotIn('consent',raw)
        self.assertFalse(record['release_ready']);self.assertFalse(record['published'])
    def test_loader_reads_development_only_without_opening_test_zip(self):
        review,approval=self.ready();out=self.root/'locked';record=ds.lock(self.main,review,approval,out)
        with patch('zipfile.ZipFile',side_effect=AssertionError('Development must not open held-out labels')):
            x,y,rows=ds.load_development(out,'train',record['lock_sha256'])
            swin=list(ds.iter_development(out,'train',record['lock_sha256'],'swin1d'))
            dino=list(ds.iter_development(out,'train',record['lock_sha256'],'dinov3'))
        self.assertEqual(x.shape,(10,3,128));self.assertEqual(sorted(y.tolist()),list(range(10)))
        self.assertEqual([r[2]['sample_id'] for r in swin],[r[2]['sample_id'] for r in dino])
        self.assertEqual(dino[0][0].shape,(3,224,224));self.assertEqual(swin[0][1],dino[0][1])
        with self.assertRaisesRegex(ValueError,'Test remains sealed'):ds.load_development(out,'test',record['lock_sha256'])
    def test_unsigned_review_missing_confirmations_and_unresolved_near_groups_rejected(self):
        review,approval=self.ready();original=load(review)
        variants=[]
        value=deepcopy(original);value['confirmations']['human_origin_and_consent_verified']=False;variants.append(value)
        value=deepcopy(original);value['reviewed_at']=None;variants.append(value)
        self.assertTrue(original['near_groups'])
        value=deepcopy(original);value['near_groups'][0]['decision']='pending';variants.append(value)
        value=deepcopy(original);value['near_groups']=[];variants.append(value)
        for i,v in enumerate(variants):
            dump(review,v)
            with self.subTest(i=i),self.assertRaises(ValueError):ds.lock(self.main,review,approval,self.root/f'bad-{i}')
    def test_stale_source_snapshot_invalidates_review(self):
        review,approval=self.ready();path=self.main/'M001.json';value=load(path);value['samples'][0]['drawing']['strokes'][0][0]['x']+=.01;dump(path,value)
        with self.assertRaisesRegex(ValueError,'Stale review'):ds.lock(self.main,review,approval,self.root/'bad')
    def test_same_person_across_codes_or_pilot_cannot_enter_main(self):
        review,approval=self.ready();value=load(review);value['participants'][1]['person_token']=value['participants'][0]['person_token'];dump(review,value)
        with self.assertRaisesRegex(ValueError,'Same actual person'):ds.lock(self.main,review,approval,self.root/'bad1')
        value=load(review);value['participants'][1]['person_token']='person_'+uuid5(NAMESPACE_URL,'SYNTHETIC-P4-P001').hex;dump(review,value)
        with self.assertRaisesRegex(ValueError,'Pilot participant reused'):ds.lock(self.main,review,approval,self.root/'bad2')
    def test_pilot_geometry_copied_into_main_rejected_after_metadata_changes(self):
        approval=self.approved()
        for pid in ['M001','M021','M026']:self.capture(pid)
        v=load(self.main/'M001.json');v['samples'][0]['drawing']['strokes']=load(self.pilot/'P001.json')['samples'][0]['drawing']['strokes'];dump(self.main/'M001.json',v)
        review=self.signed('main',approval)
        with self.assertRaisesRegex(ValueError,'Pilot strokes copied'):ds.lock(self.main,review,approval,self.root/'bad')
    def test_partial_quota_withdrawn_participant_or_no_pilot_rejected(self):
        approval=self.approved()
        for pid in ['M001','M021','M026']:self.capture(pid)
        review=self.signed('main',approval);value=load(review);value['participants'][0]['withdrawn']=True;dump(review,value)
        with self.assertRaisesRegex(ValueError,'withdrawn'):ds.lock(self.main,review,approval,self.root/'bad1')
        review=self.signed('main',approval)
        with self.assertRaisesRegex(ValueError,'Approved human pilot'):ds.lock(self.main,review,None,self.root/'bad2')
        self.capture('M026',9)
        with self.assertRaisesRegex(ValueError,'Full planned quota'):ds.lock(self.main,review,approval,self.root/'bad3')
    def test_approval_tamper_unknown_review_fields_and_prediction_exclusion_rejected(self):
        review,approval=self.ready();v=load(approval);v['sample_count']=9;dump(approval,v)
        with self.assertRaisesRegex(ValueError,'approval hash'):ds.lock(self.main,review,approval,self.root/'bad1')
        value=load(review);value['email']='not-permitted@example.test';dump(review,value)
        with self.assertRaisesRegex(ValueError,'unexpected/missing'):ds.lock(self.main,review,approval,self.root/'bad2')
        value.pop('email');value['exclusion_log']=[dict(session_id='session_'+'a'*36,original_file_sha256='a'*64,reason='model_predicted_wrong',replacement_session_id='session_'+'b'*36)];dump(review,value)
        with self.assertRaisesRegex(ValueError,'model prediction'):ds.lock(self.main,review,approval,self.root/'bad3')
    def test_changed_payload_metadata_or_extra_file_invalidates_lock(self):
        review,approval=self.ready();out=self.root/'locked';record=ds.lock(self.main,review,approval,out)
        testzip=out/'heldout-test.zip';original=testzip.read_bytes();testzip.write_bytes(original+b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):ds.verify_lock(out,record['lock_sha256'])
        testzip.write_bytes(original);(out/'extra.txt').write_text('unexpected')
        with self.assertRaisesRegex(ValueError,'Unexpected/missing'):ds.verify_lock(out,record['lock_sha256'])
        with self.assertRaisesRegex(ValueError,'metadata hash'):ds.verify_lock(out,'0'*64)
    def test_overlapping_or_existing_or_public_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'non-overlapping'):ds.review_template(self.main,'main',self.main/'review')
        with self.assertRaisesRegex(ValueError,'public/dist'):ds.separate_output(ds.ROOT/'web/public/private-lock',self.main)
        out=self.root/'exists';out.mkdir()
        with self.assertRaisesRegex(ValueError,'must be new'):ds.review_template(self.main,'main',out)
    def test_duplicate_json_keys_and_truncated_near_pairs_keep_complete_groups(self):
        bad=self.root/'duplicate.json';bad.write_text('{"label":1,"label":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):load(bad)
        value=self.capture('M001',110)
        # Distinct raw coordinates, identical normalized glyph: 5,995 pairs
        # exceed the P3 pair-report cap while the P4 group must stay complete.
        for i,sample in enumerate(value['samples']):
            sample['drawing']['strokes']=[[dict(x=x+i*.001,y=y+i*.001) for x,y in [(20,30),(100,130),(250,305)]]]
        dump(self.main/'M001.json',value);report,_,_=audit(self.main,self.policy,'main')
        self.assertTrue(report['near_groups_include_all_pairs']);self.assertTrue(report['near_duplicate_groups'])
        self.assertTrue(report['near_duplicate_report_truncated']);self.assertEqual(report['near_duplicate_pairs_total'],5995)
        group_members=set().union(*(set(g['sample_ids']) for g in report['near_duplicate_groups']))
        self.assertEqual(len(group_members),110)
        for pair in report['near_duplicates']:
            self.assertIn(pair['sample_id'],group_members);self.assertIn(pair['other_sample_id'],group_members)

if __name__=='__main__':unittest.main()
