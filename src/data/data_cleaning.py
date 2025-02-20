import pandas as pd
import os
from typing import Final

GLUCOSE_TO_CARBON_CONST: Final = (12.01*6.)/180.156
ISOLUCINE_TO_CARBON_CONST: Final = (12.01 * 6.) / 131.17


def Load_Clean_BTPH(load_clean: bool = False, to_csv: bool = False)->pd.DataFrame:

    clean_path = fr'data\processed\Clean_BTPH.csv'
    if load_clean:
        if os.path.exists(clean_path):
            return pd.read_csv(clean_path)
        else:
            if not os.path.exists(clean_path):
                raise FileNotFoundError(fr"The specified data file or directory '{clean_path}' does not exist.")

    data_path= fr'Data\Full_BTPH.xlsx'

    if not os.path.exists(data_path):
        raise FileNotFoundError(fr"The specified data file or directory '{data_path}' does not exist.")
    
    data_excel = pd.ExcelFile(data_path)
    clean_data = pd.DataFrame()

    drop_columns = ['Fed_g','Time_h','Carbon_Glucose', 'Carbon_Acetate', 'Carbon','Time_OUR', 
    'Induction_OUR', 'OUR', 'OUR_offset','CPR_div_Biomass', 'CPR_div_BiomassEst', 'dt',
    'int f(OUR)', 'int beta(t)/alfa(t) dt', 'Xkr @AgeB_on', 'Age_online',
    'X*_online', 'in Xdt', 'index', 'Average X', 'Error', 'Error OUR',
    'OUR_est', 'miu', 'Ysx', 'SUM OUR error', 'Bioreactor weight_kg',
    'Feedg_min', 'Real Feed rate', 'S', 'Current_gluc', 'Mju_teorinis',
    'Glucose_cons_max', 'Possible_mju', 'Current_cunsumption','Feedg', 'dFeedg',
    'Current_mju', 'index.1', 'GlucoseEst', 'Error_GlucoseEst','GlucFeed_div_BiomassEst',
    'dGlucFeed_div_BiomassEst', 'Feed_div_BiomassEst','Inter_BioWeight', 'OUR g/h', 'Beta_maintenance', 'Beta_product']


    for sheet_name in data_excel.sheet_names:
        if 'BTPH' in sheet_name:
            print(sheet_name)
            sheet = data_excel.parse(sheet_name=sheet_name)

            sheet = sheet.loc[:,~sheet.columns.str.match("Unnamed")]
            sheet.dropna(axis=0, subset = ['Time'], inplace=True)
            sheet.drop(columns=drop_columns, inplace=True, errors='ignore')
            sheet['CumulativeAge'] = sheet['AgemultBiomass']
            sheet.drop(columns='AgemultBiomass', inplace=True)

            sheet['Carbon'] = sheet['Glucose'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet['Carbon0'] = sheet['Carbon'].loc[0]
            sheet['Carbon_Feed'] = sheet['Glucose_Feed'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet['dCarbon_Feed'] = sheet['dGlucose_Feed'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet.drop(columns=['Glucose', 'Glucose0', 'Glucose_Feed', 'dGlucose_Feed'], inplace=True)

            sheet.dropna(axis=0, inplace=True)

            sheet['Sheet'] = sheet_name

            clean_data = pd.concat([clean_data, sheet], axis=0, ignore_index=True)
        else:
            continue

    if to_csv:
        clean_data.to_csv(clean_path, index=False)

    return clean_data

def Load_Clean_FERM(load_clean: bool = False, to_csv: bool = False)->pd.DataFrame:

    clean_path = fr'data\processed\Clean_FERM.csv'
    if load_clean:
        if os.path.exists(clean_path):
            return pd.read_csv(clean_path)
        else:
            if not os.path.exists(clean_path):
                raise FileNotFoundError(fr"The specified data file or directory '{clean_path}' does not exist.")
        
    data_path= fr'Data\Full_ferm.xlsx'
    if not os.path.exists(data_path):
        raise FileNotFoundError(fr"The specified data file or directory '{data_path}' does not exist.")
    
    data_excel = pd.ExcelFile(data_path)
    clean_data = pd.DataFrame()

    drop_columns = ['Fed_g','Time_h','Carbon_Glucose', 'Carbon_Acetate', 'Carbon','Time_OUR', 
    'Induction_OUR', 'OUR', 'OUR_offset','CPR_div_Biomass', 'CPR_div_BiomassEst', 'dt',
    'int f(OUR)', 'int beta(t)/alfa(t) dt', 'Xkr @AgeB_on', 'Age_online',
    'X*_online', 'in Xdt', 'index', 'Average X', 'Error', 'Error OUR',
    'OUR_est', 'miu', 'Ysx', 'SUM OUR error', 'Bioreactor weight_kg',
    'Feedg_min', 'Real Feed rate', 'S', 'Current_gluc', 'Mju_teorinis',
    'Glucose_cons_max', 'Possible_mju', 'Current_cunsumption','Feedg', 'dFeedg',
    'Current_mju', 'index.1', 'GlucoseEst', 'Error_GlucoseEst','GlucFeed_div_BiomassEst',
    'dGlucFeed_div_BiomassEst', 'Feed_div_BiomassEst','Inter_BioWeight', 'OUR g/h', 'Beta_maintenance', 'Beta_product']

    for sheet_name in data_excel.sheet_names:
        if 'Ferm' in sheet_name:
            print(sheet_name)
            sheet = data_excel.parse(sheet_name=sheet_name)

            sheet = sheet.loc[:,~sheet.columns.str.match("Unnamed")]
            sheet.dropna(axis=0, subset = ['Time'], inplace=True)
            sheet.drop(columns=drop_columns, inplace=True, errors='ignore')
            sheet['CumulativeAge'] = sheet['Age'].multiply(sheet['BiomassEst'])
            sheet['Biomass0'] = sheet['BiomassEst'].loc[0]
            sheet['miuSimp'] = sheet['dBiomassEst'].divide(sheet['BiomassEst'].multiply(sheet['dTime']))
            sheet['miuSimp'].loc[0] = 0.

            sheet['Carbon'] = sheet['Glucose'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet['Carbon0'] = sheet['Carbon'].loc[0]
            sheet['Carbon_Feed'] = sheet['Glucose_Feed'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet['dCarbon_Feed'] = sheet['dGlucose_Feed'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet.drop(columns=['Glucose', 'Glucose0', 'Glucose_Feed', 'dGlucose_Feed'], inplace=True)

            sheet['Sheet'] = sheet_name

            clean_data = pd.concat([clean_data, sheet], axis=0, ignore_index=True)
        else:
            continue

    if to_csv:
        clean_data.to_csv(clean_path, index=False)

    return clean_data

def Load_Clean_GSK(load_clean: bool = False, to_csv: bool = False)->pd.DataFrame:

    clean_path = fr'data\processed\Clean_GSK.csv'
    if load_clean:
        if os.path.exists(clean_path):
            return pd.read_csv(clean_path)
        else:
            if not os.path.exists(clean_path):
                raise FileNotFoundError(fr"The specified data file or directory '{clean_path}' does not exist.")
        
    data_path= fr'Data\Full_GSK.xlsx'
    if not os.path.exists(data_path):
        raise FileNotFoundError(fr"The specified data file or directory '{data_path}' does not exist.")
    
    data_excel = pd.ExcelFile(data_path)
    clean_data = pd.DataFrame()

    drop_columns = ['Fed_g','Time_h','Carbon_Glucose', 'Carbon_Acetate', 'Carbon','Time_OUR', 
    'Induction_OUR', 'OUR', 'OUR_offset','CPR_div_Biomass', 'CPR_div_BiomassEst', 'dt',
    'int f(OUR)', 'int beta(t)/alfa(t) dt', 'Xkr @AgeB_on', 'Age_online',
    'X*_online', 'in Xdt', 'index', 'Average X', 'Error', 'Error OUR',
    'OUR_est', 'miu', 'Ysx', 'SUM OUR error', 'Bioreactor weight_kg',
    'Feedg_min', 'Real Feed rate', 'S', 'Current_gluc', 'Mju_teorinis',
    'Glucose_cons_max', 'Possible_mju', 'Current_cunsumption','Feedg', 'dFeedg',
    'Current_mju', 'index.1', 'GlucoseEst', 'Error_GlucoseEst','GlucFeed_div_BiomassEst',
    'dGlucFeed_div_BiomassEst', 'Feed_div_BiomassEst','Inter_BioWeight', 'OUR g/h', 'Beta_maintenance', 'Beta_product', 'NH4']

    for sheet_name in data_excel.sheet_names:
        if 'MME' in sheet_name:
            print(sheet_name)
            sheet = data_excel.parse(sheet_name=sheet_name)

            sheet = sheet.loc[:,~sheet.columns.str.match("Unnamed")]
            sheet.dropna(axis=0, subset = ['Time'], inplace=True)
            sheet.drop(columns=drop_columns, inplace=True, errors='ignore')
            sheet['Protein'] = sheet['Protein g/L']
            sheet.drop(columns=['Protein g/L', 'Time.1'], inplace=True)
            sheet['CumulativeAge'] = sheet['AgemultBiomass']
            sheet.drop(columns='AgemultBiomass', inplace=True)

            sheet['Carbon'] = sheet['Glucose'].multiply(GLUCOSE_TO_CARBON_CONST)
            sheet['Carbon0'] = sheet['Carbon'].loc[0]
            sheet['Carbon_Feed'] = sheet['Glucose_Feed'].multiply(GLUCOSE_TO_CARBON_CONST).add(sheet['L-Isoleucine'].multiply(ISOLUCINE_TO_CARBON_CONST))
            sheet['dCarbon_Feed'] = sheet['Carbon_Feed'].subtract(sheet['Carbon_Feed'].shift())
            sheet['dCarbon_Feed'].loc[0] = sheet['Carbon_Feed'].loc[0]
            sheet.drop(columns=['Glucose', 'Glucose0', 'Glucose_Feed', 'dGlucose_Feed', 'L-Isoleucine'], inplace=True)

            sheet['Sheet'] = sheet_name

            clean_data = pd.concat([clean_data, sheet], axis=0, ignore_index=True)
        else:
            continue
    if to_csv:
        clean_data.to_csv(clean_path, index=False)

    return clean_data
