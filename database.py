import pandas as pd

def filter_hyposmia(df, hyposmia_cutoff):
    df = df.copy()
    #filter both for positive HYPOSMIA
    if hyposmia_cutoff == '2023':
        print("Using 2023 article cutoff")
        df = df[df['HYPOSMIA'] == 1]
        #for target == 1, only include PATNOs from data/hc_subset_patnos.csv
        
    elif hyposmia_cutoff == 'ppmi':
        print("Using PPMI cutoff")
        df = df[df['HYPOSMIA_PPMI'] == 1]
    #check if it is a number
    elif hyposmia_cutoff.isnumeric():
        print(f"Using cutoff: {hyposmia_cutoff}")
        df = df[df['TOTAL_CORRECT'] <= int(hyposmia_cutoff)]
    else:
        print("Cutoff invalid, using whole dataset")
        
    return df

class Database:
    def __init__(self, clinical_path, remote_path, log=True):
        self.dataClinical = pd.read_csv(clinical_path, low_memory=False)
        self.dataRemote = pd.read_csv(remote_path, low_memory=False)
        self.data_all = pd.DataFrame()
        self.log = log
        print("Data loaded")
        if self.log: self.log_shapes()
    
    def prepare_data(self,log=True):
        #self.dataPest = self.dataPest[COLUMNS_TO_KEEP]
        #if self.log: print(self.dataDemogr)
        self.dataClinical.reset_index(drop=True, inplace=True)
        self.dataRemote.reset_index(drop=True, inplace=True)
        
        if self.log: self.log_shapes()
        
        #create target; 0 for remote, 1 for clinical
        self.dataClinical['target'] = 1
        self.dataRemote['target'] = 0
        if self.log: self.log_shapes()

        common_columns = list(set(self.dataClinical.columns).intersection(set(self.dataRemote.columns)))
        
        #drop different columns
        self.dataClinical = self.dataClinical[common_columns]
        self.dataRemote = self.dataRemote[common_columns]
        
        #concat
        self.data_all = pd.concat([self.dataClinical, self.dataRemote], ignore_index=True)
            
        if self.log: self.log_shapes()
        
    def get_data(self,):
        
        return self.data_all
        
    
    def log_shapes(self,):
        print("PPMI Clinical shape: ", self.dataClinical.shape)
        print("PPMI Remote shape: ", self.dataRemote.shape)
        print("All Data shape: ", self.data_all.shape)
        
if __name__ == "__main__":
    clinical_path = "data/clinical_data.csv"
    remote_path = "data/remote_data.csv"
    data = Database(clinical_path, remote_path)
    data.prepare_data()
    dataset = data.get_data()
    