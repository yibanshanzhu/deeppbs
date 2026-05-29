from Bio.PDB import NeighborSearch
import numpy as np
import sys

def getProteinNeighborSearch(protein, pdb):
    try:
        cid = pdb.split("_")[1].split(".")[0]
        return NeighborSearch(list(protein[cid].get_atoms()))
    except:
        return NeighborSearch(list(protein.get_atoms()))

def computeContactMask(protein, pdb, V_dna, cutoff=5):
    ns = getProteinNeighborSearch(protein, pdb)
    contact_counts = np.zeros(V_dna.shape[0], dtype=int)

    for i in range(V_dna.shape[0]):
        for item in V_dna[i,:,:]:
            if not np.isfinite(item).all():
                continue
            out = ns.search(item, cutoff)
            contact_counts[i] += len(out)

    return contact_counts > 0, contact_counts

def countContacts(protein, pdb, V_dna, dna_mask):
    ns = getProteinNeighborSearch(protein, pdb)
    V_dna = V_dna[dna_mask,:,:].reshape(-1,3)
    count = 0
    for item in V_dna:
        if not np.isfinite(item).all():
            continue
        out = ns.search(item, 5)
        count += len(out)

    return np.array([count])
