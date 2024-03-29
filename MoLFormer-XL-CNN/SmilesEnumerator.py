import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
        return normalized
    except:
        print("Failed to normalize {} !".format(smi))
        return np.nan
    
def randomize_smiles(smiles, isomericSmiles=False):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=isomericSmiles)


def augment_smiles(smiles, n=10, isomericSmiles=False):
    """Augment a SMILES string by randomizing it n times"""
    return [randomize_smiles(smiles, isomericSmiles=isomericSmiles) for _ in range(n)]

def main():
    path = r"/home/HYJ/174/molformer/data/DIR/train_src.csv"
    target_path = r"/home/HYJ/174/molformer/data/DIR/train.csv"
    augment_num = 10
    df = pd.read_csv(path)
    len_df = len(df)
    lst = []
    
    df_info = df.drop('canonical_smiles', axis=1)
    
    df_result = df.copy()
    for i in tqdm(range(len(df))):
        smiles = df.loc[i, "canonical_smiles"]
            
        lst.append(smiles)
        # if len(smiles) <= 10:
        #     continue
        # lst.extend(augment_smiles(smiles, augment_num))
        lst = augment_smiles(smiles, augment_num)
        df_smiles = pd.DataFrame(lst, columns=["canonical_smiles"])
        
        for column in df_info.columns:
            df_smiles[column] = df_info.loc[i, column]
        
        df_result = pd.concat([df_result, df_smiles]).drop_duplicates(
            subset=["canonical_smiles"])
    print("Augmented from {0} to {1} smiles".format(len_df, len(df_result)))
    df_result.to_csv(target_path, index=False)

if __name__ == "__main__":
    main()
