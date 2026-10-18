"""P5 input, cache, training and final-test gates, isolated synthetic fixtures."""
import ast
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from training.drawai import paint_train as p
from training.drawai.models import build_model, swin_image_config, swin_config
from training.drawai.private_data import load, dump, sha


class StubBackbone(nn.Module):
    provenance={'pretrained_loaded':True,'test_stub_only':True}
    def forward_features(self,x):return x.mean((1,2,3)).unsqueeze(1).repeat(1,768)


class PaintTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        p.seed_all(42,2)

    def test_new_factory_preserves_legacy_input(self):
        self.assertEqual(swin_config(64)['in_chans'],3)
        self.assertEqual(swin_image_config()['in_chans'],64)
        model=build_model('swin1d_image');self.assertEqual(tuple(model(torch.zeros(2,64,64)).shape),(2,10))

    def test_new_factory_rejects_legacy_length_and_weights(self):
        for kwargs in ({'length':128},{'weights_dir':self.root},{'smoke_only':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):build_model('swin1d_image',**kwargs)

    def test_fixed_plan_no_test_selection(self):
        v=p.plan();self.assertEqual(v['seeds'],[42,43,44]);self.assertEqual(v['lock_sha256'],p.LOCK_HASH)
        self.assertFalse(v['augmentation']);self.assertTrue(v['dinov3']['backbone_frozen'])
        self.assertFalse(v['similarity_independence_verified'])

    def test_plan_drift_rejected(self):
        p.prepare(self.root);value=load(self.root/'experiment-plan.json');value['seeds']=[99]
        dump(self.root/'experiment-plan.json',value)
        with self.assertRaisesRegex(ValueError,'plan changed'):p.read_plan(self.root)

    def test_existing_public_output_rejected(self):
        with patch.object(p,'ROOT',self.root):
            folder=self.root/'web/public/existing';folder.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'public'):p.prepare(folder)
            self.assertEqual(list(folder.iterdir()),[])

    def test_output_does_not_modify_locked_dataset(self):
        dataset=self.root/'locked';dataset.mkdir()
        with self.assertRaisesRegex(ValueError,'overlap'):
            p.train_experiment(dataset,dataset/'run',self.root/'weights',self.root/'cache')
        self.assertEqual(list(dataset.iterdir()),[])

    def test_validation_ranking(self):
        def row(f1,loss,seed):return dict(validation={'macro_f1':f1,'loss':loss},seed=seed,family='swin1d-image')
        winner=min([row(.8,.2,42),row(.9,.3,44),row(.9,.2,43),row(.9,.2,42)],key=p.rank)
        self.assertEqual(winner['seed'],42);self.assertEqual(winner['validation']['macro_f1'],.9)

    def test_predict_metrics_recomputed_from_logits(self):
        model=nn.Identity();x=np.eye(10,dtype=np.float32)*10;y=np.arange(10,dtype=np.int64)
        metrics,logits=p.predict(model,x,y,batch_size=3)
        self.assertEqual(metrics['accuracy'],1);self.assertEqual(metrics['macro_f1'],1)
        self.assertTrue(np.array_equal(logits,x))

    def test_invalid_logits_rejected(self):
        with self.assertRaisesRegex(ValueError,'logits'):
            p.predict(nn.Identity(),np.full((2,10),np.nan,dtype=np.float32),np.array([0,1]))

    def test_writer_metrics_not_device_inference(self):
        y=np.array([0,1]);logits=np.eye(10,dtype=np.float32)[:2]
        metrics=p.grouped_metrics(y,logits,[{'person_id':26},{'person_id':27}])
        self.assertEqual(set(metrics),{'26','27'});self.assertEqual(metrics['26']['samples'],1)

    def cache_fixture(self):
        pixels=np.full((2,64,64),255,dtype=np.uint8);pixels[:,12:40,30]=0
        rows=[dict(sample_id=f'synthetic_{i}',label=str(i),image_sha256=sha(v.tobytes())) for i,v in enumerate(pixels)]
        return pixels,rows

    def test_cache_hash_and_order_binding(self):
        pixels,rows=self.cache_fixture();folder=self.root/'cache'
        features,meta=p.frozen_features(StubBackbone(),pixels,rows,folder,'development')
        self.assertEqual(features.shape,(2,768));self.assertEqual(meta['key']['partition'],'development')
        with patch.object(StubBackbone,'forward_features',side_effect=AssertionError('Recomputed valid cache')):
            loaded,_=p.frozen_features(StubBackbone(),pixels,rows,folder,'development')
        self.assertTrue(np.array_equal(loaded,features))
        with self.assertRaisesRegex(ValueError,'cache'):
            p.frozen_features(StubBackbone(),pixels[::-1],rows[::-1],folder,'development')

    def test_corrupt_cache_rejected(self):
        pixels,rows=self.cache_fixture();folder=self.root/'cache'
        p.frozen_features(StubBackbone(),pixels,rows,folder,'development')
        (folder/'development.npy').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'cache'):p.frozen_features(StubBackbone(),pixels,rows,folder,'development')

    def test_test_reader_needs_selection_before_zip(self):
        with patch.object(p.zipfile,'ZipFile',side_effect=AssertionError('Opened held-out before selection')):
            with self.assertRaises(FileNotFoundError):p.test_images(self.root,self.root)

    def test_training_function_has_no_test_reader_call(self):
        # Static guard supplements actual runtime audit which blocks ZipFile.
        tree=ast.parse(Path(p.__file__).read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_experiment')
        calls={n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertNotIn('test_images',calls);self.assertNotIn('evaluate',calls)

    def test_real_optimizer_checkpoint_roundtrip(self):
        original=p.plan;value=copy.deepcopy(original());value['swin']['epochs']=2;value['swin']['warmup_epochs']=0
        rows=[dict(sample_id=f'fixture_{i}',split='train' if i<20 else 'validation',person_id=6 if i<20 else 26) for i in range(30)]
        x=np.random.default_rng(42).random((30,64,64),dtype=np.float32);y=np.arange(30,dtype=np.int64)%10
        with patch.object(p,'plan',return_value=value):
            result=p.train_one('swin1d-image',42,x,y,rows,self.root/'run','unit-test-plan')
        self.assertTrue(result['trained']);self.assertFalse(result['release_ready']);self.assertEqual(result['epochs_completed'],2)
        saved=torch.load(self.root/'run'/result['checkpoint'],weights_only=True)
        model=build_model('swin1d_image');model.load_state_dict(saved['model_state_dict'],strict=True)
        model.eval();metrics,logits=p.predict(model,x[20:],y[20:])
        self.assertEqual(metrics,result['validation'])
        with np.load(self.root/'run/validation-predictions.npz',allow_pickle=False) as z:
            self.assertTrue(np.array_equal(z['logits'],logits))

    def test_learning_rate_schedule_bounded(self):
        c=p.plan()['swin'];rates=[p.learning_rate(c,e) for e in range(c['epochs'])]
        self.assertGreater(min(rates),0);self.assertLessEqual(max(rates),c['lr'])

    def test_image_training_resume_replays_uninterrupted_weights(self):
        config=copy.deepcopy(p.plan());config['swin']['epochs']=2;config['swin']['warmup_epochs']=0
        rows=[dict(sample_id=f'fixture_{i}',split='train' if i<10 else 'validation',person_id=6 if i<10 else 26) for i in range(20)]
        x=np.random.default_rng(7).random((20,64,64),dtype=np.float32);y=np.arange(20,dtype=np.int64)%10
        with patch.object(p,'plan',return_value=config):
            full=p.train_one('swin1d-image',42,x,y,rows,self.root/'full','same-plan')
        short=copy.deepcopy(config);short['swin']['epochs']=1
        with patch.object(p,'plan',return_value=short):
            p.train_one('swin1d-image',42,x,y,rows,self.root/'partial','same-plan')
        (self.root/'resumed').mkdir()
        for name in ('last.pth','best-trainable.pth'):
            shutil.copyfile(self.root/'partial'/name,self.root/'resumed'/name)
        with patch.object(p,'plan',return_value=config):
            resumed=p.train_one('swin1d-image',42,x,y,rows,self.root/'resumed','same-plan')
        a=torch.load(self.root/'full'/full['checkpoint'],weights_only=True)['model_state_dict']
        b=torch.load(self.root/'resumed'/resumed['checkpoint'],weights_only=True)['model_state_dict']
        self.assertEqual(full['best'],resumed['best'])
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_names_separate_from_legacy(self):
        a=p.checkpoint_name('swin1d-image',42);b=p.checkpoint_name('dinov3-vits16',42)
        self.assertIn('rows64',a);self.assertIn('rgb224',b);self.assertNotEqual(a,b)


if __name__=='__main__':unittest.main()
