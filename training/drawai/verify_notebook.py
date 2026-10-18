"""Execute the current Kaggle notebook in explicit local CPU smoke mode."""
import asyncio
import os
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

from .prepare_data import ROOT, digest, write_json


def main():
    if sys.platform=='win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    out=ROOT/'artifacts/drawai/notebook-smoke'
    kernel=out/'kernels/drawai-local'
    write_json(kernel/'kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                       display_name='DrawAI local CPU',language='python'))
    values=dict(DRAWAI_NOTEBOOK_MODE='smoke',DRAWAI_REQUIRE_CUDA='0',DRAWAI_SOURCE_ROOT=str(ROOT),
                DRAWAI_NOTEBOOK_OUTPUT=str(ROOT/'artifacts/drawai/swin-notebook-smoke-v2'))
    old={key:os.environ.get(key) for key in values}
    os.environ.update(values)
    source=ROOT/'notebooks/kaggle_train_drawai.ipynb'
    notebook=nbformat.read(source,as_version=4)
    nbformat.validate(notebook)
    manager=AsyncKernelManager(kernel_name='drawai-local',
                               kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(out/'kernels')]))
    try:
        client=NotebookClient(notebook,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        code=[c for c in notebook.cells if c.cell_type=='code']
        if not all(c.execution_count is not None for c in code) or any(o.output_type=='error' for c in code for o in c.outputs):
            raise ValueError('Notebook cell failed/not executed')
        nbformat.write(notebook,out/'executed.ipynb')
        write_json(out/'verification.json',dict(status='passed',source_sha256=digest(source),
                   executed_sha256=digest(out/'executed.ipynb'),code_cells_executed=len(code),
                   mode='smoke',device='cpu',full_branch_executed=False,kaggle_gpu_executed=False,
                   test_opened=False,interpreter=sys.executable))
        print('PASS: current notebook executed in CPU smoke; full/GPU branch not executed.')
    finally:
        if manager.has_kernel:
            asyncio.run(manager.shutdown_kernel(now=True))
        for key,value in old.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


if __name__=='__main__': main()
