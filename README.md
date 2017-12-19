Prerequisites
-------------
If you have conda you can create a dedicated environment with
``` sh
conda create -n pysit python=2.7
source activate pysit
```

Then install the dependencies
``` sh
conda install numpy
pip install pysit --pre
pip uninstall pysit # Keeps the dependencies we need
pip install matplotlib==1.4.3 obspy==0.10.2 ipython==5.0.0
```


Installation
------------

Clone the repository and
``` sh
python setup.py build
python setup.py install
```

You can checked wether the installation is ok by running examples.

Troubleshooting
---------------
 We experienced troubles on OSX Yosemite at building time, openMP was not supported by clang. To solve that problem, we installed gcc5 with Homebrew.  
