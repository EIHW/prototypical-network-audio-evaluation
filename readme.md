## A Prototypical Network Approach for Evaluating Generated Emotional Speech

This repository contains code for the INTERSPEECH 2021 paper :page_facing_up: "A Prototypical Network Approach for Evaluating Generated Emotional Speech", by Alice Baird, Silvan Mertes, Manuel Milling, Lukas Stappen, Thomas Wiest, Elisabeth André, and Björn W. Schuller.

The prototypical network applied is adapated based on the Py-torch implementation from <a href="https://github.com/jsalbert/prototypical-networks">Snell et al</a>.

Here we include code for the adapted prototypical network, and embedding-space evaluation. We also include `augmentation_options.py` for the data augmentation methods applied. More detail on SpecAug found <a href="https://github.com/DemisEom/SpecAugment"> here</a>. For audio generation <a href="https://github.com/chrisdonahue/wavegan">WaveGAN</a> was applied. 

Any questions feel free to reach out! :e-mail: alicebaird@ieee.org

### Setup :gear:	 

1. Install virtualenv, create and activate new enviroment

```
pip3 install virtualenv 
virtualenv .protonets 
source .protonets/bin/activate
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Due to data sharing limitations, we share only the WaveGAN generated spectrogram images (based on the original training set). To test the code you can <a href="https://github.com/EIHW/prototypical-network-audio-evaluation/tree/main/data">unzip</a>, the archive included here. If you would like the GEMEP sub-set used in the publication get in touch. 

### Train and Test :steam_locomotive: 

This version of the code has been adapted to run without a GPU.   

1. train network with original spectrograms. 

`bash train.sh model_name`

2. test the network utilising the generated data. 

`bash test_spec_gen.sh model_name`

### Visualise prototypes and embeddings space :eyes:

We also include a script (`tsne-plot.py`) to visualise more easily the embedding space. 

<img src="https://github.com/EIHW/prototypical-network-audio-evaluation/blob/main/plot_ex.png" width="350" />


### Pair-wise embedding space diversity

1. To download the emeddings from all experiemnts of the <i>Baird et al</i>, you can <a href="https://drive.google.com/file/d/1UYchZpFJfiL8fBj9JazGqEz3rfO9shOZ/view?usp=sharing">download</a> and place these under embeddings/.

2. Run `embedding_diversity_analysis.py` to calculate a average pair-wise distance between two points from source samples and different augmentation techniques for each emotion.

3. Results are stored as csv-file (with French emotion labels) and as heatmaps (with English emotion label) in the folder `result_pairwise_distance/`.



### Citation and Contributors

If you use the code from this repositroy please add the following citation to your paper:

A.Baird, S. Mertes, M. Milling, L. Stappen, T. Wiest, E. André, and B. W. Schuller, “A Prototypical Network Approach for Evaluating Generated Emotional Speech” in Proc. INTERSPEECH 2021. Brno, Czech Republic: ISCA, Sep. 2021, p. [to appear]
```bibtex
@inproceedings{baird2021interspeech,
    title={{A Prototypical Network Approach for Evaluating Generated Emotional Speech}},
    author={Baird, Alice and Mertes, Silvan and Milling,Manuel and Stappen,Lukas and Wiest, Thomas and Andr\'{e}, Elisabeth and Schuller, Bj\"{o}rn W.},
    address={Brno, Czech Republic},
    booktitle={Proc. INTERSPEECH 2021},
    organization={ISCA},
    year={Sep. 2021},
    pages={[to appear]}
}
```

Thanks to the contributers of this repository :smiling_face_with_three_hearts:.

<table>
  <tr>
    <td align="center">
<a href="https://github.com/aliceebaird"><img src="https://avatars.githubusercontent.com/u/10690171?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alice</b></sub></a><br /><td align="center">
<a href="https://github.com/millinma"><img src="https://avatars.githubusercontent.com/u/16241688?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Manuel</b></sub></a><br /><td align="center">
<a href="https://github.com/TheThow"><img src="https://avatars.githubusercontent.com/u/5088879?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Thomas</b></sub></a><br />
  </tr>
</table>


