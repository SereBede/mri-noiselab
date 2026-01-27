Tutorial
========
These tutorials introduce the core ideas behind the noise model and the correction strategy implemented in mri-noiselab, using synthetic images where ground truth is known.

The examples are designed to be simple and controlled, so that the effect of Rayleigh noise estimation and subtraction can be clearly observed and quantified. They progressively introduce:

.. toctree::
   :maxdepth: 1

   uniform_image
   three_levels
   three_rois

Together, these tutorials help build intuition before applying the method to real MRI data.

**Requirement: numpy, matplotlib, mri-noiselab**

The code can be executed in an IDE or from the terminal.

When matplotlib.pyplot.show() is called, execution pauses until the figure window is closed; figures can be saved to file before closing the window to continue execution.
