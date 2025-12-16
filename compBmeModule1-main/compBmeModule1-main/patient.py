import pandas as pd

class Patient:
        all_patients = []

        def __init__(self, donor_id, atherosclerosis, THAL, ABeta40, ABeta42, CASI):
                self.donor_id = donor_id
                self.atherosclerosis = atherosclerosis
                self.THAL = THAL
                self.ABeta40 = ABeta40
                self.ABeta42 = ABeta42
                self.CASI = CASI
                Patient.all_patients.append(self)
                
        def __repr__(self):
                return f"Patient(ID: {self.donor_id}, Atherosclerosis: {self.atherosclerosis}, THAL: {self.THAL}, ABeta40: {self.ABeta40}, ABeta42: {self.ABeta42})"

        def get_id(self):
                return self.donor_id
        def get_atherosclerosis(self):
                return self.atherosclerosis
        def get_THAL(self):
                return self.THAL
        def get_ABeta40(self):
                return self.ABeta40
        def get_ABeta42(self):
                return self.ABeta42
        def get_CASI(self):
                return self.CASI

        @classmethod
        def combine_and_instantiate(cls):
                with open("data/UpdatedLuminex.csv") as f:
                        luminex = pd.read_csv(f)

                with open("data/UpdatedMetaData.csv") as f:
                        metadata = pd.read_csv(f)
                
                # merge on id
                merged_df = pd.merge(luminex, metadata, on='Donor ID')

                for index, row in merged_df.iterrows():
                        cls(
                                donor_id = row['Donor ID'],
                                atherosclerosis = row['Atherosclerosis'],
                                THAL = row['Thal'],
                                ABeta40 = row['ABeta40 pg/ug'],
                                ABeta42 = row['ABeta42 pg/ug'],
                                CASI = row['Last CASI Score']
                        )

                cls.all_patients.sort(key = cls.get_id)