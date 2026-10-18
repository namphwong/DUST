"""D3 leakage prevention, numerical metrics, augmentation and exact resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from training.drawai.experiment import config_for, ranked
from training.drawai.metrics import classification_metrics
from training.drawai.prepare_data import write_json
from training.drawai.prepare_data import digest
from training.drawai.preprocessing import contract
from training.drawai.train import augment, learning_rate, load_split, tensor_hash, train


def fake_data(folder):
    rng=np.random.default_rng(3)
    entries={}
    for split,n in [('train',40),('validation',20)]:
        X=rng.normal(0,0.2,(n,3,64)).astype(np.float32); X[:,2]=0; X[:,2,0]=1
        y=np.arange(n,dtype=np.int64)%10
        ids=np.asarray([f'{split}-{i}' for i in range(n)])
        np.savez(folder/f'{split}.npz',X=X,y=y,sample_ids=ids)
        entries[split]=dict(samples=n,tensor_sha256=tensor_hash(X,y,ids),entries=[dict(id=i,label=int(label)) for i,label in zip(ids,y)])
    write_json(folder/'manifest.json',dict(splits=entries))
    write_json(folder/'preprocessing.json',contract(64))
    write_json(folder/'labels.json',dict(classes=[dict(id=i,name=str(i)) for i in range(10)]))


class TrainingTests(unittest.TestCase):
    def test_archived_family_cannot_enter_new_training(self):
        from training.drawai.models import build_model
        from training.drawai.train import validate_config
        config = config_for('swin',64,False,42,False)
        config['model'] = 'cnn'
        with self.assertRaises(ValueError): validate_config(config)
        with self.assertRaises(ValueError): config_for('cnn',64,False,42,False)
        with self.assertRaises(ValueError): build_model('cnn',64)

    def test_selected_swin_audit_rejects_broken_chain_before_reading_weights(self):
        from training.drawai.verify_training import verify_selected_swin
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); (folder/'evaluation').mkdir()
            write_json(folder/'experiment-plan.json',dict(version='drawai-experiment-v1'))
            write_json(folder/'configuration-lock.json',dict(plan_sha256='broken'))
            write_json(folder/'selection.json',dict(plan_sha256='broken'))
            write_json(folder/'evaluation/evaluation.json',{})
            with patch('torch.load',side_effect=AssertionError('No weights may be loaded')):
                with self.assertRaisesRegex(ValueError,'provenance chain'):
                    verify_selected_swin(folder)

    def test_metrics_match_hand_calculation(self):
        result=classification_metrics([0,0,1,1],[0,1,1,1])
        self.assertEqual(result['accuracy'],0.75)
        self.assertAlmostEqual(result['macro_f1'],((2/3)+0.8)/10)
        self.assertEqual(result['confusion_matrix'][0][:2],[1,1])
        self.assertEqual(result['per_class'][1]['recall'],1)
        with self.assertRaises(ValueError): classification_metrics([10],[0])

    def test_augmentation_preserves_marker_and_input(self):
        torch.manual_seed(42)
        X=torch.zeros(20,3,64); X[:,0]=torch.linspace(-0.5,0.5,64); X[:,2,0]=1
        before=X.clone(); output=augment(X)
        torch.testing.assert_close(X,before,atol=0,rtol=0)
        torch.testing.assert_close(output[:,2],X[:,2],atol=0,rtol=0)
        span=torch.linalg.vector_norm(output[:,:2,-1]-output[:,:2,0],dim=1)
        self.assertTrue(((span>=0.9)&(span<=1.1)).all())
        self.assertTrue((output[:,0,-1]>output[:,0,0]).all())

    def test_scheduler_and_selection_use_validation(self):
        config=config_for('swin',64,False,42,True)
        self.assertAlmostEqual(learning_rate(config,0),config['lr']/3)
        self.assertAlmostEqual(learning_rate(config,49),config['lr']*0.05)
        a=dict(config=config,best=dict(macro_f1=0.9,loss=0.2),run_id='a')
        b=copy.deepcopy(a); b['best']['macro_f1']=0.91
        self.assertLess(ranked(b),ranked(a))

    def test_train_never_reads_test_and_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'data'; data.mkdir(); fake_data(data)
            config=config_for('swin',64,True,42,False)
            config.update(epochs=3,warmup_epochs=1,patience=3,batch_size=20)
            original=np.load
            def guard(path,*args,**kwargs):
                self.assertNotEqual(Path(path).name,'test.npz')
                return original(path,*args,**kwargs)
            with patch('training.drawai.train.np.load',side_effect=guard):
                full=train(config,data,root/'full','cpu',2)
                first=train(config,data,root/'resumed','cpu',2,stop_after=1)
                self.assertEqual(first['status'],'interrupted')
                resumed=train(config,data,root/'resumed','cpu',2,resume=True)
            self.assertEqual(full['best'],resumed['best'])
            a=torch.load(root/'full/last.pth',weights_only=True)
            b=torch.load(root/'resumed/last.pth',weights_only=True)
            for name,value in a['model_state_dict'].items():
                torch.testing.assert_close(value,b['model_state_dict'][name],atol=0,rtol=0)
            self.assertEqual([r['validation'] for r in a['history']],[r['validation'] for r in b['history']])
            with self.assertRaises(ValueError): train(config,data,root/'full','cpu',2)
            changed=copy.deepcopy(config); changed['lr']=0.002
            with self.assertRaises(ValueError): train(changed,data,root/'resumed','cpu',2,resume=True)
            with self.assertRaisesRegex(ValueError,'thread count'):
                train(config,data,root/'resumed','cpu',4,resume=True)

    def test_tampered_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); fake_data(folder)
            with np.load(folder/'train.npz') as data:
                X,y,ids=data['X'].copy(),data['y'],data['sample_ids']
            X[0,0,0]+=0.01
            np.savez(folder/'train.npz',X=X,y=y,sample_ids=ids)
            with self.assertRaises(ValueError): load_split(folder,'train',64)

    def test_early_stopping_keeps_first_tied_best_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); data=folder/'data'; data.mkdir(); fake_data(data)
            config=config_for('swin',64,False,42,False)
            config.update(epochs=6,warmup_epochs=1,patience=2,batch_size=40)
            metrics=dict(macro_f1=0.5,accuracy=0.5,loss=1.0)
            with patch('training.drawai.train.predict',return_value=(metrics,np.zeros((20,10),np.float32))):
                summary=train(config,data,folder/'run','cpu',2)
            self.assertTrue(summary['stopped_early'])
            self.assertEqual(summary['epochs_completed'],3)
            self.assertEqual(summary['best']['epoch'],1)

    def test_evaluation_cannot_open_test_without_three_locked_runs(self):
        from training.drawai.evaluate import evaluate
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            from training.drawai.experiment import PLAN
            write_json(folder/'experiment-plan.json',PLAN)
            write_json(folder/'configuration-lock.json',{})
            write_json(folder/'selection.json',dict(plan_sha256=digest(folder/'experiment-plan.json'),
                       configuration_lock_sha256=digest(folder/'configuration-lock.json'),runs=[]))
            with patch('training.drawai.evaluate.load_split',side_effect=AssertionError('No test may be opened')):
                with self.assertRaisesRegex(ValueError,'three official'):
                    evaluate(folder,2)
            self.assertFalse((folder/'test-access.json').exists())


if __name__=='__main__': unittest.main()
