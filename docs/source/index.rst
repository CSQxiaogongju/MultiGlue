.. MultiGlue documentation master file, created by
   sphinx-quickstart on Mon Sep  7 10:32:06 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MultiGlue's documentation
=====================================

MultiGlue
_________

MultiGlue is a computational framework for integrative and generative analysis of spatial multi-omics data.

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   

       
   installation
   tutorials/Tutorial1_Mouse_Thymus
   tutorials/Tutorial2_P22_Mouse_Brain


Overview
________

Recent advances in spatial multi-omics enable the simultaneous profiling of multiple molecular layers within the same tissue, offering new opportunities to characterize tissue organization and cross-omics regulation. However, integrative analysis remains challenging due to heterogeneous feature spaces across modalities. Moreover, existing approaches primarily focus on multi-omics integration, with limited ability to reconstruct biologically coherent profiles across modalities. Here, we introduce MultiGlue, a spatially aware graph contrastive learning framework for integrative and generative analysis of spatial multi-omics data. MultiGlue first employs graph contrastive learning to capture spatial context while learning modality-specific representations, and then adaptively integrates complementary information across modalities into a unified representation. Beyond integration, MultiGlue incorporates prior molecular knowledge to guide cross-modal generation, enabling the reconstruction of spatially and biologically coherent omics profiles across molecular layers. Systematic benchmarking on 17 simulated and real-world datasets across diverse tissues, platforms and modality combinations showed that MultiGlue consistently outperformed 11 state-of-the-art methods in integration accuracy and robustness, while maintaining computational scalability across spatial transcriptome–proteome, transcriptome–epigenome and transcriptome–epigenome–proteome data. Notably, when applied to complex mouse thymus and brain samples, MultiGlue resolved fine-grained anatomical structures and cortical layers at higher spatial resolution that were not able to be clearly delineated by competing methods and showed stronger concordance with known tissue anatomy and molecular markers. Importantly, MultiGlue extended spatial multi-omics analysis beyond multi-omics integration by accurately generating unobserved molecular profiles across modalities. In spatial tri-omics mouse brain data, the generated transcriptomic, epigenomic and proteomic profiles preserved spatial organization and recapitulated biologically meaningful cross-modal relationships, revealing concordant transcriptome–epigenome–proteome patterns. Collectively, MultiGlue established an accurate, robust, scalable computational framework for integrative and generative analysis of spatial multi-omics data.


Citation
________

Siqi Chen et al. Spatially aware integration and biologically informed generation of spatial multi-omics with MultiGlue.
