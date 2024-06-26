<h1 align="center"> <strong> Eddo's PhD repository</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=0aYqoMMAAAAJ&hl=en">
    <img alt="Google Scholar" src="https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white">
  </a>
  <a href="https://www.linkedin.com/in/eddo-wesselink-1a106089/?originalSubdomain=nl">
    <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
  </a>
  <a href="https://orcid.org/0000-0002-2024-6986">
    <img alt="ORCID" src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white">
  </a>
  <a href="https://twitter.com/EddoWesselink">
    <img alt="X" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white">
  </a>
</p>

<p align="center">
    <img src="assets/image_github_readme.png" height="300" width="900"/>
</p>

## <strong> Content 

In this repository I will share my PhD codes for supervised and non-supervised machine learning models for the quantification of lumbar paraspinal muscle health using conventional T<sub>2</sub>-weighted MRI. The repository will contain programming codes (Python) for:

- Convolutional Neural Networks for the automatic segmentation of the lumbar paraspinal muscles
  
  Link to paper: https://www.nature.com/articles/s41598-022-16710-5

- Quantifying lumbar paraspinal intramuscular fat from clinical MRI 

  Link to paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10869289/


### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- SciPy
- scikit-learn
- nibabel

You can install all the dependencies by running:

```bash
pip install -r requirements.txt
```
This will install all the required packages listed in the `requirements.txt` file. Make sure you have `pip` installed and configured on your system.

### Usage 

To use the code, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Eddowesselink/PhD.git
```
2. Navigate to the code directory where you stored the repository

```bash
cd `/path/to/your/repository`
```
### Thresholding 
3. Run the script main_thresholding.py with the required arguments:
```bash
python main_thresholding.py --data_dir /path/to/your/data --kmeans --gmm
```
Replace `/path/to/your/data` with the path to the directory containing your MRI data. You can specify either `--kmeans` or `--gmm` to choose between KMeans or Gaussian Mixture Model clustering for segmentation.

### CNN
3. Run the script main_CNN.py with the required arguments:
```bash
python main_thresholding.py --data_dir /path/to/your/data --model_dir /path/to/your/data 
```
Replace `/path/to/your/data` in -- data_dir with the path to the directory containing your MRI data. 
Replace `/path/to/your/data` in --model_dir with the path to the directory containing the model parameters. 