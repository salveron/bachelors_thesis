### Computational Auditory Scene Analysis (CASA) for Separating Monophonic Music

*Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022*

---

### Structure

```
│   README.md .................. [installation guide and folder structure, this file]
│   requirements.txt ........... [required Python packages]
│   assignment.md .............. [informal assignment of the thesis]
│
└───src ........................ [Python source codes and examples for the practical part]
│   │   main_example.ipynb ..... [Jupyter notebook with an example of the system usage]
│   │   experiments.ipynb ...... [Jupyter notebook with experiments]
│   │   misc.ipynb ............. [Jupyter notebook with library tests and other miscellaneous stuff]
│   │
│   └───scripts ................ [Python source codes]
│   
└───text ....................... [LaTeX source codes and the generated PDF with text]
│   │   main.pdf ............... [PDF-formatted text of the thesis]
│   │   cover.pdf .............. [PDF-formatted front cover of the thesis]
│   │   cover.pdf .............. [PDF-formatted front cover of the thesis]
│   │
│   └───src .................... [LaTeX source codes for the text]
│       │   ctufit-thesis.cls .. [thesis template design file]
│       │   ctufit-cover.cls ... [cover template design file]
│       │   main.tex ........... [main TeX source file]
│       │   cover.tex .......... [TeX source file for the cover]
│       │
│       └───chapters ........... [TeX files for distinct chapters in the text]
│       └───include ............ [images, PDF files and bibliography to include in the text]
│
└───data ....................... [all input and output files for the system]
    │   masked_data.npy ........ [precomputed dataset of masked cochleagrams for the classifier]
    │   masked_labels.npy ...... [precomputed labels for the masked cochleagrams for the classifier]
    │   unmasked_data.npy ...... [precomputed dataset of unmasked cochleagrams for the classifier]
    │   unmasked_labels.npy .... [precomputed labels for the unmasked cochleagrams for the classifier]
    │
    └───aac_source ............. [input AAC monophonic piano sound files]    
    └───target_sounds .......... [input WAV monophonic piano sound files]
    └───background_sounds ...... [input WAV background sound files]
    └───masks .................. [precomputed ideal binary masks for the clean target sounds]
    └───output ................. [outputs from the Jupyter notebooks]

```
---

### Installation

To install the dependencies to your virtual environment, run this:

`pip install -r requirements.txt`

If you want to run Jupyter notebooks, you need to install *Jupyter*. If you want to play sounds from notebooks (played by default), you need *Pygame* as well:

`pip install jupyter pygame`
