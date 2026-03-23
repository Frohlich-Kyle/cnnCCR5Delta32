Kyle Frohlich

__Overview__
This project serves as a simple tool to build the understanding, knowledge, and usage of convolutional neural networks
in the context of biological applications. This is a starter project I undertook to apply the knowledge I have learned
about developing deep learning solutions. 

__Problem Statement__
Particularly, this project seeks to create a model that can learn and predict whether or not someone has a resistance
to the Human Immunodeficiency Virus (HIV) through checking for a particular mutation. The mechanism of which is a widely
known through the CCR5-Delta32 indel, where the nucleotide typically found at position 32 of the CCR5 gene is deleted. 
This mutation causes a malformation of the protein resulting in greatly reduced binding efficacy of HIV. However this 
mutation can be easily detected with deterministic methods so this project is primarily a learning exercise of using 
modern deep learning based bioinformatics workflow.

__Structure and Constraints__
The architecture for this project is meant to resemble the situations and environment related to medical applications
of deep learning technology. Primarily this includes the major constraint of limited data. Because of this constraint
the structure will be as follows.

1.) K-fold Cross Validation
K-folds will be used with 5 total folds. One fold is used for validation while the remaining four folds are used for
training. This will result in a total of 5 models being trained. These models can be tested for cross validation tests
to reduce the occurance that rote memorization occurs. 

2.) Ensemble Prediction Method
Following in suite to the multiple models made, the cross validation process will be incorporated the final model that
utilizes all fold models created in tandem. By showing each models prediction as well as an probabilistic output between
their predictions, a smoother and more stable final prediction can be made.

3.) Holdout Testing
While K-fold is used due to lack of data, in order to have a viable testing set size without making 6 folds, sequences will
be chosen at random from the entirity of the data. This is to construct a pseudo "new" dataset given the limited 
number of samples.

__Files__
1.) main.py
Primary structure where the model will be trained, validated, tested, and evaluated
2.) prepData.py
File designed just to pull data from the appropriate files and put them into a easy to use format
3.) model.py
Architecture class for the model
4.) fetchData.py
Pulls data from NCBI through Entrez
5.) augmentData.py
Takes real Delta32 CCR5 sequences and fabricates new ones with minor but realistic changes in the form of Single Nucleotide
Polymorphisms (SNPs). 

__Data Represenation__
DNA sequences are represented using the base nucleotides Adenine (A), Cytosine (C), Guanine (G), and Thymine (T). As DNA
is already widely represented as a singular one dimensional sequence in many publications and storage within NCBI, the
sequences will be directly converted to a one dimensional tensor as below.
(4, sequenceLength)
This representation of data allows for the use of 1D convolutional layers to detect local sequence patterns or motifs.

__Data Source__
DNA sequences for the CCR5 sequence were sourced from the National Center for Biotechnology Information's (NCBI)
National Library of Medicine (NIH) through Entrez. This data is written to CCR5sequences.csv
**IMPORTANT**
The Delta32 indel of CCR5 is a relatively rare mutation that is not frequently sequenced due to the circumstances of
the "problem" being solved. This means no further sequencing is required. As such Delta32 CCR5 sequences are particularly uncommon. 
To introduce a more comprehensive environment, synthetic augmented data was created to provide more training examples for the model. 
The file augmentData.py takes the known and real CCR5 sequences, fetched from NCBI, and introduces realistic and sporadic single nucleotide 
polymorphisms (SNP) to prevent the model from rote memorization. While under real circumstances this is unacceptable as a proper training
method, this project serves more as a proof of concept rather than an attempt at making a working tool.

__Model Architecture__
Convolutional 1D Layer
ReLU Activation
Global Max Pooling
Adam as Optimizer
BCEWithLogitsLoss as the Loss function
Fully Connected Output Layer

__Results__
"CCR5 CNN Percentage Accuracy Results Table.xlsx" includes 5 iterative runs of the model run at the kernel sizes 1 through 8.
Included with the accuracy per model is an average across all models as well as the filter which had to highest activation.
At kernel size 8 and above the model achieves a stable 100% accuracy at detecting the CCR5-Delta32 mutation. Through kernel
sizes 4 through 7 average accuracy remains relatively high to the point of having the ability to deem them workable for the
problem at hand. Kernel size 3 shows a noticable drop in relative average accuracy but is still retains acceptable results.

At kernel sizes 1 and 2 there is a much more significant drop in accuracy relative to the average accuracy of other kernel
sizes. It is very important to note that there is about an 80% split between Delta32 CCR5 sequences and Wildtype. This means
that any accuracies around the 80% range are most likely the model "cheating" and simply choosing to mark all sequences as
Delta32 mutations without making meaningful choices.


__Reproducibility__
Python 3.12.3
Pytorch 2.10.0+cu128
NCBI Email Account

__Run Files__
python3 fetchData.py
1.) This will gather data from NCBI and compile it into a .csv.
2.) Requires the user to enter a valid email for NCBI.

python3 augmentData.py
1.) Will take the files from the compiled .csv and modify them slightly.
2.) The modified sequences will be saved into a new .csv to distinctly separate them.

python3 main.py
1.) Load and preprocess sequence data
2.) Perform 5-fold cross-validation training
3.) Evaluate model performance
4.) Save trained models for ensemble prediction