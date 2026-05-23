#Author: Kyle Frohlich
#Name: CNN of CCR5 Delta-32
#Date: 06Mar2026
#Purpose: Main driver for the codebase

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

from model import CCR5CNN
import prepData

#needed check for usage of gpu instead of cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#functions to set up from prepData.py
sequences, labels = prepData.load_all_sequences()
test_data, remaining_data = prepData.create_testing_data(sequences, labels, prepData.test_ratio)
folds = prepData.create_kfolds(*zip(*remaining_data))

#this is the number of folds we are using total
kFolds = 5

allModels = []
allValLosses = []

#loop for making all cross validation models
for k, (train_data, val_data) in enumerate(folds):

    #new model for each loop
    newModel = CCR5CNN().to(device)
    lossFunc = nn.BCEWithLogitsLoss()
    #refresh optimizer each time a new model is made
    optimFunc = optim.Adam(newModel.parameters(), lr=0.01)

    #begin training of the new model
    newModel.train()

    train_sequences, train_labels = zip(*train_data)
    val_sequences, val_labels = zip(*val_data)

    train_loader = DataLoader(prepData.CCR5Dataset(list(train_sequences), list(train_labels)), batch_size=8, shuffle=True)
    val_loader = DataLoader(prepData.CCR5Dataset(list(val_sequences), list(val_labels)), batch_size=8)

    for epoch in range(100):

        for batch in train_loader:

            #import the data of the current batch to train on
            x, y = batch[0].to(device), batch[1].to(device)

            output = newModel(x)
            loss = lossFunc(output, y)

            optimFunc.zero_grad()
            loss.backward()
            optimFunc.step()

        print(f"Fold {k+1}, Epoch {epoch+1}, Loss: {loss.item():.4f}")

    allModels.append(newModel)

    newModel.eval()
    with torch.no_grad():
        
        for batch in val_loader:

            x, y = batch[0].to(device), batch[1].to(device)
            output = newModel(x)
            val_loss = lossFunc(output, y)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        allValLosses.append(val_loss.item())
        print(f"Fold {k+1} Val Loss: {val_loss.item():.4f}")



#testing of all the models
with torch.no_grad():

    all_predictions = []
    true_labels = []
    test_sequences, test_labels = zip(*test_data)
    test_loader = DataLoader(prepData.CCR5Dataset(list(test_sequences), list(test_labels)), batch_size=8, shuffle=False)

    #to avoid multiple collections, just gets all the labels first
    for batch in test_loader:

        x, y = batch[0].to(device), batch[1].to(device)
        true_labels.append(y)

    #brings all labels together for later
    true_labels = torch.cat(true_labels)

    #goes through each model for testing all of them individually
    for model in allModels:

        model_predictions = []

        #goes per batch for each model
        for batch in test_loader:

            x, y = batch[0].to(device), batch[1].to(device)
            output = model(x)
            model_predictions.append(output)

        #brings all predictions made together for later
        all_predictions.append(torch.cat(model_predictions))


    #go through each model and calculate accuracy
    for i, model_preds in enumerate(all_predictions):

        probs = torch.sigmoid(model_preds)
        predicted = (probs > 0.5).float()
        correct = (predicted == true_labels).sum().item()
        accuracy = correct / len(true_labels) * 100
        print(f"Model {i+1} Accuracy: {accuracy:.1f}%")

    #set up ensemble data
    prediction_stack = torch.stack(all_predictions)
    ensemble_avg = prediction_stack.mean(dim=0)

    #calculate ensemble prediction
    ensemble_probability = torch.sigmoid(ensemble_avg)
    ensemble_predicted = (ensemble_probability > 0.5).float()
    correct = (ensemble_predicted == true_labels).sum().item()
    ensemble_accuracy = correct / len(true_labels) * 100

    print(f"Ensemble Accuracy: {ensemble_accuracy:.1f}%")




#motif analysis - find what k-mers the best model's filters respond to
#has to match model.py
kernel_size = 8 
best_model_idx = allValLosses.index(min(allValLosses))
best_model = allModels[best_model_idx]
best_model.eval()

delta32_kmers = Counter()
wildtype_kmers = Counter()

with torch.no_grad():

    for seq_str, label in zip(test_sequences, test_labels):

        # shape: [1, 4, seq_len]
        x = prepData.one_hot_encode(seq_str).unsqueeze(0).to(device)

        # shape: [1, 16, seq_len - kernel_size + 1]
        conv_out = F.relu(best_model.convolution1(x))

        # for each filter, get the position of its strongest activation
        max_activations, max_positions = conv_out[0].max(dim=1)

        # pick the single filter that fired the hardest
        best_filter = max_activations.argmax().item()
        pos = max_positions[best_filter].item()

        kmer = seq_str[pos:pos + kernel_size]

        if label == 1:

            delta32_kmers[kmer] += 1

        else:

            wildtype_kmers[kmer] += 1

print("\nTop motifs in Delta32 sequences:")
for kmer, count in delta32_kmers.most_common(5):

    print(f"{kmer}: {count}")

print("\nTop motifs in Wildtype sequences:")
for kmer, count in wildtype_kmers.most_common(5):

    print(f"{kmer}: {count}")






