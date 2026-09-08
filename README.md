[README.md](https://github.com/user-attachments/files/31932945/README.md)
# Spatially aware integration and biologically informed generation of spatial multi-omics with MultiGlue

<img width="5106" height="3009" alt="Figure_1" src="https://github.com/user-attachments/assets/1ce0f281-d4ff-4929-a3d5-69f763fa916e" />

# Overview
Recent advances in spatial multi-omics enable the simultaneous profiling of multiple molecular layers within the same tissue, offering new opportunities to characterize tissue organization and cross-omics regulation. However, integrative analysis remains challenging due to heterogeneous feature spaces across modalities. Moreover, existing approaches primarily focus on multi-omics integration, with limited ability to reconstruct biologically coherent profiles across modalities. Here, we introduce MultiGlue, a spatially aware graph contrastive learning framework for integrative and generative analysis of spatial multi-omics data. MultiGlue first employs graph contrastive learning to capture spatial context while learning modality-specific representations, and then adaptively integrates complementary information across modalities into a unified
representation. Beyond integration, MultiGlue incorporates prior molecular knowledge to guide cross-modal generation, enabling the reconstruction of spatially and biologically coherent omics profiles across molecular layers. Systematic benchmarking on 17 simulated and real-world datasets across diverse tissues, platforms and modality combinations showed that MultiGlue consistently outperformed 11 state-of-the-art methods in integration accuracy and robustness, while maintaining computational scalability across spatial transcriptome–proteome, transcriptome–epigenome and transcriptome–epigenome–proteome data. Notably, when applied to complex mouse thymus and brain samples, MultiGlue resolved fine-grained anatomical structures and cortical layers at higher spatial resolution that were not able to be clearly delineated by competing methods and showed stronger concordance with known tissue anatomy and molecular markers. Importantly, MultiGlue extended spatial multi-omics analysis beyond multi-omics integration by accurately generating unobserved molecular profiles across modalities. In spatial tri-omics mouse brain data, the generated transcriptomic, epigenomic and proteomic profiles preserved spatial organization and recapitulated biologically meaningful cross-modal relationships, revealing concordant transcriptome–epigenome–proteome patterns. Collectively, MultiGlue established an accurate, robust, scalable computational framework for integrative and generative analysis of spatial multi-omics data.

# Requirements
python==3.8
torch>=1.8.0
cudnn>=10.2
numpy==1.22.3
scanpy==1.9.1
anndata==0.8.0
rpy2==3.4.1
pandas==1.4.2
scipy==1.8.1
scikit-learn==1.1.1
scikit-misc==0.2.0
tqdm==4.64.0
matplotlib==3.4.2
R==4.0.3

# Tutorial
For the step-by-step tutorial, please refer to: https://multiglue-tutorials.readthedocs.io/en/latest/

# Data
The data used as input to the methods tested in this study, inclusive of the simulated and real-world datasets have been uploaded to Zenodo and is freely available at https://zenodo.org/records/22143859

# Citation
Siqi Chen et al. Spatially aware integration and biologically informed generation of spatial multi-omics with MultiGlue.
