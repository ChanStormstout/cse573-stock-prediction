"""Kernel runtime recovery only; the registered model/candidates remain unchanged."""
import models as m
from kernel_fast import DenseLinearSVC
m.SVC=DenseLinearSVC
import runpy
runpy.run_path(str(m.Path(__file__).with_name('run_stage.py')),run_name='__main__')
