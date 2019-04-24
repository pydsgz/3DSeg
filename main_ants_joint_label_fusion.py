# imports
import os
from nipype.interfaces import fsl
import utils
import subprocess

def set_env_vars():
    # add /usr/lib/ants to system path
    os.getenv('PATH')
    os.environ['PATH'] = '/usr/lib/ants:'+os.getenv('PATH')
    print(os.getenv('PATH'))

    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    os.environ['FSLDIR']='/usr/share/fsl/5.0'
    os.system('. /usr/share/fsl/5.0/etc/fslconf/fsl.sh')
    os.environ['PATH'] = '/usr/share/fsl/5.0/bin:'+os.getenv('PATH')
    if os.getenv('LD_LIBRARY_PATH') is not None:
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/fsl/5.0:' + os.getenv(
            'LD_LIBRARY_PATH')
    else:
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/fsl/5.0:'

if __name__ == '__main__':
    # Set environment variables.
    set_env_vars()

    """
    Perform ANTs joint label fusion on files at
        pnSRC = './vols_all_%s_%s' % (mfs_, seq_name)
    
    and output will be save at:
        pnBase = './vols_all_%s_%s'  % (mfs_, seq_name)
    
    Select:
    seq = 2 for T2, 1 for T1, and 0 for SWI.
    mfs = 0 for 7T, 1 for 3T
    """
    utils.main_ants_joint_label_fusion(seq=2, mfs=0)