#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jetscape-ml-tensorflow-nn-dataset-builder-single-file-analyzer.ipynb

python jetscape-ml-tensorflow-nn-dataset-builder-single-file-analyzer.py -i finalStateHadrons-Matter-1k.dat -d 1000 -y MVAC -o jetscape-ml-benchmark-dataset-1k-matter.pkl

python jetscape-ml-tensorflow-nn-dataset-builder-single-file-analyzer.py -i finalStateHadrons-MatterLbt-1k.dat -d 1000 -y MLBT -o jetscape-ml-benchmark-dataset-1k-lbt.pkl