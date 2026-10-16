"""Differentiable CPU bridge to isolated FEniCS using its exact tangent VJP."""
from pathlib import Path
import subprocess,os,json,tempfile
import numpy as np
import torch

class FEMBridge:
    def __init__(self,root,threads=None):
        self.root=Path(root);self.temp=Path(tempfile.mkdtemp(prefix='fem-',dir=self.root/'experiments/logs'))
        env=os.environ.copy();env.update(XDG_CACHE_HOME=str(self.root/'.cache'),MAMBA_ROOT_PREFIX=str(self.root/'.mamba'),DIJITSO_CACHE_DIR=str(self.root/'.fenics-cache-activated'))
        if threads is not None:env.update(OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads))
        self.log=open(self.temp/'service.log','w')
        cmd=[str(self.root/'.tools/bin/micromamba'),'run','--attach','','-p',str(self.root/'.fenics'),'python',str(self.root/'scripts/fem_service.py')]
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,bufsize=1)
        line=self.process.stdout.readline()
        if not line or not json.loads(line).get('ready'):raise RuntimeError('FEniCS assembly service failed: '+str(self.temp/'service.log'))
        self.metadata=json.loads(line)
    def evaluate(self,u,pressure,gradient=True):
        request=self.temp/'request.npz';response=self.temp/'response.npz';np.savez(request,u=np.asarray(u,dtype='float64'),pressure=float(pressure))
        self.process.stdin.write(json.dumps({'request':str(request),'response':str(response),'gradient':gradient})+'\n');self.process.stdin.flush();line=self.process.stdout.readline()
        if not line:raise RuntimeError('FEniCS assembly service terminated; see '+str(self.temp/'service.log'))
        info=json.loads(line)
        if not info['ok']:raise RuntimeError(info['error'])
        with np.load(response) as a:return {k:a[k].copy() for k in a.files}
    def solve(self,pressure):
        response=self.temp/'solve_response.npz'
        self.process.stdin.write(json.dumps({'solve':float(pressure),'response':str(response)})+'\n');self.process.stdin.flush()
        info=json.loads(self.process.stdout.readline())
        if not info['ok']:raise RuntimeError(info['error'])
        return info['record']
    def loss(self,u,pressure):return _ResidualLoss.apply(u,float(pressure),self)
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"quit":true}\n');self.process.stdin.flush()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=5)
        self.log.close()

class _ResidualLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,pressure,bridge):
        result=bridge.evaluate(u.detach().cpu().numpy(),pressure)
        gradient=torch.as_tensor(result['gradient'],dtype=u.dtype,device=u.device);ctx.save_for_backward(gradient)
        return u.new_tensor(float(result['loss']))
    @staticmethod
    def backward(ctx,grad_output):
        gradient,=ctx.saved_tensors
        return grad_output*gradient,None,None
