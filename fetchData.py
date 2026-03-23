#Author: Kyle Frohlich
#Name: CNN of CCR5 Delta-32
#Date: 10Mar2026
#Purpose: Fetch CCR5 wildtype and Delta-32 sequences from NCBI and write to CSV

from Bio import Entrez, SeqIO
import csv
import time

#NCBI requires an email to use Entrez
try:
    
    user_name = input("Enter your NCBI email: ")
    if not user_name:
        raise ValueError("Email cannot be empty")
    
    Entrez.email = user_name
    
except ValueError as error:
    print(f"Error: {error}")
        
#bp to extract around the deletion site
WINDOW = 200        
OUTPUT_FILE = "CCR5sequences.csv"
#max per class
MAX_SEQS = 300

#in wildtype, not in Delta32
DELETION_SIGNATURE = "GTCAGTATCAATTCTGGAAGAATTTCCAGAC"
#used to anchor sequence window
FLANK_ANCHOR = "CCATACA"


def fetch_ids(query, max_results):

    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_sequences(id_list):

    ids = ",".join(id_list)
    handle = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()
    return records


def find_window_center(seq):

    #anchor on the flanking sequence
    idx = seq.find(FLANK_ANCHOR)
    if idx != -1:
        return idx + len(FLANK_ANCHOR)
    #center of sequence
    return len(seq) // 2


def is_wildtype(seq):
    #wildtype CCR5 contains the 32bp region that Delta32 is missing
    return DELETION_SIGNATURE in seq


def is_delta32(seq):
    #delta32 should not contain the deleted region, but must contain the CCR5 flank anchor
    return DELETION_SIGNATURE not in seq and FLANK_ANCHOR in seq


def extract_window(seq, center, window):

    half = window // 2
    start = max(0, center - half)
    end = start + window

    if end > len(seq):

        end = len(seq)
        start = max(0, end - window)

    region = str(seq[start:end])
    #pad if shorter than window
    region = region.ljust(window, "N")
    return region


def is_valid(seq, window):

    seq = str(seq).upper()

    if len(seq) < window:

        return False
    
    if any(c not in "ACGTN" for c in seq):

        return False
    
    return True


def main():

    print("Fetching wildtype CCR5 sequences...")
    wt_query = 'CCR5[Gene] AND Homo sapiens[Organism] AND "complete cds"[Title]'
    wt_ids = fetch_ids(wt_query, MAX_SEQS)
    print(f"  Found {len(wt_ids)} wildtype IDs")

    time.sleep(1)

    print("Fetching CCR5-Delta32 sequences...")
    d32_query = '(CCR5 delta32 OR CCR5-delta32 OR "CCR5 delta 32") AND Homo sapiens[Organism]'
    d32_ids = fetch_ids(d32_query, MAX_SEQS)
    print(f"  Found {len(d32_ids)} Delta-32 IDs")

    time.sleep(1)

    print("Downloading wildtype sequences...")
    wt_records = fetch_sequences(wt_ids) if wt_ids else []

    time.sleep(1)

    print("Downloading Delta-32 sequences...")
    d32_records = fetch_sequences(d32_ids) if d32_ids else []

    written_wt = 0
    written_d32 = 0
    #for handeling duplication
    seen = set()  

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        for record in wt_records:

            seq = str(record.seq).upper()

            if not is_valid(seq, WINDOW):

                continue

            if not is_wildtype(seq):

                print(f"  Skipping mislabeled wildtype: {record.id}")
                continue

            center = find_window_center(seq)
            region = extract_window(seq, center, WINDOW)

            if region in seen:

                continue

            seen.add(region)

            writer.writerow([region, 0])
            written_wt += 1

        for record in d32_records:

            seq = str(record.seq).upper()

            if not is_valid(seq, WINDOW):

                continue

            if not is_delta32(seq):
                
                print(f"  Skipping mislabeled Delta32 (contains WT signature): {record.id}")
                continue

            center = find_window_center(seq)
            region = extract_window(seq, center, WINDOW)

            if region in seen:

                continue
            
            seen.add(region)

            writer.writerow([region, 1])
            written_d32 += 1

    print(f"Done. Wrote {written_wt + written_d32} sequences to {OUTPUT_FILE}")
    print(f"  Wildtype: {written_wt}, Delta-32: {written_d32}")


if __name__ == "__main__":
    main()
