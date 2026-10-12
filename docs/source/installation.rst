Installation
============

The MultiGlue package has been developed based on the pytorch framework and can be implemented with both GPU and CPU. Using GPU acceleration can significantly speed up the integration progress. So we recommend running the package with GPU first.

To ensure smooth installation and execution of the package, it’s important to have PyTorch and CUDA (including CUDNN) installed correctly. These dependencies are necessary for GPU acceleration. Make sure you have the appropriate GPU drivers installed on your system as well.

To run the MultiGlue package, two methods are provided. Note that the installation of all packages included in file ‘requirement.txt’ are needed.

Create environment
------------------

We recommend creating a dedicated Conda environment.

.. code-block:: bash
#Create an environment called MulitGlue

conda create -n MultiGlue python=3.9

#Activate your environment

conda activate MultiGlue

#Install MultiGlue (MultiGlue-3M) package
------------------

.. code-block:: bash

   git clone https://github.com/CSQxiaogongju/MultiGlue.git
   cd MultiGlue


Install dependencies
--------------------

.. code-block:: bash

   pip install -r requirements.txt
