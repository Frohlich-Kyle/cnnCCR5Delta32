#Author: Kyle Frohlich
#Name: CNN of CCR5 Delta-32
#Date: 10Mar2026
#Purpose: Augment real CCR5-Delta32 sequences with realistic SNP variants

import csv
import random

INPUT_FILE = "CCR5sequences.csv"
OUTPUT_FILE = "augmentedCCR5sequences.csv"

SNP_RATE = 0.005
#number of augmented Delta32 sequences to generate
AUGMENT_TARGET = 100  

NUCLEOTIDES = ["A", "C", "G", "T"]


def introduce_snps(seq, snp_rate):

    seq = list(seq)
    for i in range(len(seq)):

        if random.random() < snp_rate and seq[i] != "N":

            alternatives = [n for n in NUCLEOTIDES if n != seq[i]]
            seq[i] = random.choice(alternatives)

    return "".join(seq)


def main():

    #set seed for reproducibility
    random.seed(42)

    #load only real Delta32 sequences from the real data file
    d32_seqs = []
    with open(INPUT_FILE, newline="") as f:

        for row in csv.reader(f):

            if len(row) == 2 and row[1].strip() == "1":

                d32_seqs.append(row[0].strip())

    if not d32_seqs:

        print("No Delta32 sequences found in input file. Run fetchData.py first.")
        return

    print(f"Found {len(d32_seqs)} real Delta32 sequences to augment from.")

    written = 0
    seen = set(d32_seqs)  # avoid duplicating real sequences

    with open(OUTPUT_FILE, "w", newline="") as f:
        
        writer = csv.writer(f)

        while written < AUGMENT_TARGET:

            base = random.choice(d32_seqs)
            variant = introduce_snps(base, SNP_RATE)

            if variant in seen:

                continue

            seen.add(variant)
            writer.writerow([variant, 1])
            written += 1

    print(f"Done. Wrote {written} augmented Delta32 sequences to {OUTPUT_FILE}")
    print("Note: These are synthetic variants for training only. Real human sequences are in CCR5sequences.csv")


if __name__ == "__main__":
    
    main()
