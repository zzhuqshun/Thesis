# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:49:36 2023

@author: Florian
"""

import pandas as pd
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

def read_parquet_files(path):
    files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    df_list = []
    
    time_offset = 0

    for file in tqdm(files, desc="Dateien einlesen"):
        file_path = os.path.join(path, file)
        df_temp = pd.read_parquet(file_path)
        
        # Aktualisieren der 'Testtime[s]' Spalte, um sie fortlaufend zu machen
        if 'Testtime[s]' in df_temp.columns:
            df_temp['Testtime[s]'] += time_offset
        
            # Aktualisieren des time_offset für die nächste Datei
            time_offset = df_temp['Testtime[s]'].iloc[-1]
        
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    
    # Konvertieren von 'Absolute_Time[yyyy-mm-dd hh:mm:ss]' in das datetime-Format, wenn die Spalte im DataFrame vorhanden ist
    if 'Absolute_Time[yyyy-mm-dd hh:mm:ss]' in df.columns:
        df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        
    return df

def calculate_capacity_soh_qsum_efc(df):
    global first_df_c_actual

    # Identifizieren der Kapazitätstestintervalle
    capacity_test_intervals = df['Schedule_Step_ID'].isin(range(5, 8))

    # Berechnen der Kapazität für jedes Intervall
    df['delta_time_h'] = df['Testtime[s]'].diff() / 3600
    df.loc[df['delta_time_h'].abs() > 0.5, 'delta_time_h'] = 0
    df['delta_time_h'].fillna(0, inplace=True)
    df['Q_abs'] = abs(df['delta_time_h'] * df['Current[A]'])
    df['Q'] = df['delta_time_h'] * df['Current[A]']   
    
    df['Q_m'] = df['Q'].cumsum()
    df['Q_sum'] = df['Q_abs'].cumsum()
    
    # Initialisieren der neuen Spalten
    df['Capacity[Ah]'] = np.nan
    df['SOH'] = np.nan

    # Resetting the index to avoid non-unique index error
    df.reset_index(drop=True, inplace=True)
    
    # Iteration über jedes Kapazitätstestintervall und Berechnung der Kapazität und des SOH
    start_idx = None
    for idx, in_test in enumerate(capacity_test_intervals):
        if in_test and start_idx is None:
            start_idx = idx
        elif not in_test and start_idx is not None:
            end_idx = idx
            current_capacity = df.loc[start_idx:end_idx, 'Q_abs'].sum()
            df.loc[start_idx:end_idx, 'Capacity[Ah]'] = current_capacity
            
            # Setzen der anfänglichen Kapazität beim ersten Kapazitätstest
            if first_df_c_actual is None:
                first_df_c_actual = current_capacity
                
            df.loc[start_idx:end_idx, 'SOH'] = current_capacity / first_df_c_actual
            start_idx = None

    # 

    # Initialisieren der neuen Spalte Q_c
    df['Q_c'] = 0.0

    # Variablen für die Zustandsüberwachung
    Q_c = 0.0
    V_min = 2.5
    V_max = 3.65
    
    # Fortlaufende Berechnung von Q_c
    print('SOC Berechnung')
    for i in tqdm(range(len(df)), desc="Processing data", unit="row"):
        Q_c += df.loc[i, 'Q']
        if df.loc[i, 'Voltage[V]'] <= V_min:
            Q_c = -first_df_c_actual
        elif df.loc[i, 'Voltage[V]'] >= V_max:
            Q_c = 0
        
        df.loc[i, 'Q_c'] = Q_c
       
    # Berechnung der EFC
    df['EFC'] = df['Q_sum'] / first_df_c_actual
    df.drop(columns=['Q', 'Q_abs'], inplace=True)
    
    return df

def plot_df(df):
    if 'Absolute_Time[yyyy-mm-dd hh:mm:ss]' in df.columns:
        df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        time_data = df['Absolute_Time[yyyy-mm-dd hh:mm:ss]']
    else:
        time_data = df.index
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
    ax1.plot(time_data, df['Voltage[V]'], color='b', label='Voltage[V]')
    ax1.set_ylabel('Voltage[V]', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc='upper left')
    ax2.plot(time_data, df['Schedule_Step_ID'], color='r', label='Schedule_Step_ID')
    ax2.set_ylabel('Schedule_Step_ID', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper left')
    ax3.plot(time_data, df['Current[A]'], color='g', label='Current[A]')
    ax3.set_ylabel('Current[A]', color='g')
    ax3.set_xlabel('Absolute_Time[yyyy-mm-dd hh:mm:ss]')
    ax3.tick_params('y', colors='g')
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def extract_profiles(df, initial_step_ids, subsequent_step_ids):
    """Extrahiert Daten basierend auf Listen von Schedule_Step_ID Werten."""
    df_list = []

    # Extrahieren des ersten Profils
    mask_initial = df['Schedule_Step_ID'].isin(initial_step_ids)
    mask_groups_initial = (mask_initial != mask_initial.shift()).cumsum() * mask_initial
    if (mask_groups_initial == 1).any():
        df_list.append(df[mask_groups_initial == 1].copy())
        # Entfernen des ersten Profils und aller vorherigen Daten
        df = df[df.index > df[mask_groups_initial == 1].index[-1]]

    # Extrahieren der nachfolgenden Profile
    mask_subsequent = df['Schedule_Step_ID'].isin(subsequent_step_ids)
    mask_groups_subsequent = (mask_subsequent != mask_subsequent.shift()).cumsum() * mask_subsequent
    for group in mask_groups_subsequent.unique():
        if group != 0:
            df_list.append(df[mask_groups_subsequent == group].copy())

    return df_list

def extract_capacity_test(df, step_ids):
    """Extrahiert Daten basierend auf gegebenen Schedule_Step_ID Werten."""
    mask = df['Schedule_Step_ID'].isin(step_ids)
    mask_groups = (mask != mask.shift()).cumsum() * mask
    df_list = [df[mask_groups == group].copy() for group in mask_groups.unique() if group != 0]
    
    return df_list

def extract_checkups(df):
    # DataFrame-Index zurücksetzen
    df = df.reset_index(drop=True)
    
    # Finden Sie alle Start- und Endpunkte des Musters
    starts = df[df['Schedule_Step_ID'] == 5].index
    ends = df[df['Schedule_Step_ID'] == 61].index
    
    df_list = []
    for start, end in zip(starts, ends):
        # Überprüfen Sie, ob der Startindex wirklich vor dem Endindex liegt
        if start < end:
            df_list.append(df.loc[start:end].copy())
    
    df_list = process_dataframe(df)
    return df_list

def extract_cycles(df):
    # DataFrame-Index zurücksetzen
    df = df.reset_index(drop=True)
    
    # Finden Sie alle potenziellen Startpunkte des Musters, bei denen von einem Wert 57, 61 oder 7 auf 1 gewechselt wird
    potential_starts = df[(df['Schedule_Step_ID'] == 1) & (df['Schedule_Step_ID'].shift(1).isin([57, 61, 7]))].index

    # Für jeden potenziellen Startpunkt, finde den echten Startpunkt, der der letzte Punkt ist, an dem Schedule_Step_ID 1 ist, bevor es auf 2 wechselt
    starts = [df.loc[idx:df.loc[idx:].query('Schedule_Step_ID == 2').index[0]].query('Schedule_Step_ID == 1').index[-1] for idx in potential_starts]

    ends = []
    for start in starts:
        # Wählen Sie den ersten Punkt nach 'start', bei dem Schedule_Step_ID den Wert 4 hat
        potential_end = df[(df.index > start) & (df['Schedule_Step_ID'] == 4)].index.min()
        
        # Wenn ein potenzielles Ende gefunden wurde
        if not np.isnan(potential_end):
            # Finden Sie den Punkt, an dem Schedule_Step_ID von 4 auf 5 wechselt, nach potential_end
            next_change = df[(df.index > potential_end) & (df['Schedule_Step_ID'] == 5)].index.min()
            
            # Ermitteln Sie den maximalen Spannungswert innerhalb des Bereichs zwischen 'potential_end' und 'next_change'
            V_max = df.loc[potential_end:next_change]['Voltage[V]'].max()
            
            # Finden Sie den ersten Punkt, an dem die Spannung V_max erreicht, nach dem Startpunkt
            end = df[(df.index > potential_end) & (df.index < next_change) & (df['Voltage[V]'] == V_max)].index.min()
            
            # Wenn ein Endpunkt gefunden wurde, fügen Sie ihn zur Liste hinzu
            if not np.isnan(end):
                ends.append(int(end))
    
    df_list = []

    for start, end in zip(starts, ends):
        # Extrahiere den DataFrame-Abschnitt zwischen Start und Ende
        df_cycle = df.loc[start:end].copy()
        df_list.append(df_cycle)
        
    return df_list

def extract_checkups(df):
    # DataFrame-Index zurücksetzen
    df = df.reset_index(drop=True)
    # Finden Sie alle Startpunkte des Musters, bei denen Schedule_Step_ID das erste Mal den Wert 5 hat
    potential_starts = df[(df['Schedule_Step_ID'] == 5) & (df['Schedule_Step_ID'].shift(1) != 5)].index

    # Finden Sie alle Endpunkte des Musters, bei denen Schedule_Step_ID das erste Mal den Wert 61 hat
    ends = df[(df['Schedule_Step_ID'] == 61) & (df['Schedule_Step_ID'].shift(1) != 61)].index

    # Für jeden Endpunkt, finde den letzten Startpunkt vor diesem Endpunkt
    starts = [potential_starts[potential_starts < end].max() for end in ends]

    # Aufteilen des DataFrames df basierend auf den Starts und Ends
    df_list = [df.loc[start:end].copy() for start, end in zip(starts, ends)]

    return df_list


def plot_combined_dfs(dfs):
    combined_df = pd.concat(dfs)
    if 'Absolute_Time[yyyy-mm-dd hh:mm:ss]' in combined_df.columns:
        combined_df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(combined_df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        time_data = combined_df['Absolute_Time[yyyy-mm-dd hh:mm:ss]']
    else:
        time_data = combined_df.index
    
    
first_df_c_actual = None

def process_dataframe(df):
    global first_df_c_actual
    
    # Sicherstellen, dass die Zeitdaten als datetime interpretiert werden
    df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])

    # Berechnen des Zeitdeltas in Stunden
    df['delta_time_h'] = df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'].diff().dt.total_seconds() / 3600

    # Setzen Sie das erste Delta auf 0, da es NaN ist
    df.loc[df.index[0], 'delta_time_h'] = 0

    # Berechnung von I_cum
    df['I_cum'] = abs((df['delta_time_h'] * df['Current[A]']).cumsum())
    
    # Berechnung von C_actual
    c_actual = df['I_cum'].iloc[-1]
    df['C_actual'] = c_actual

    # Speichern des C_actual des ersten DataFrames
    if first_df_c_actual is None:
        first_df_c_actual = c_actual

    # Berechnung von SOH
    df['SOH'] = abs(c_actual / first_df_c_actual)

    # Entfernen der Hilfsspalte delta_time_h
    #df.drop(columns=['delta_time_h'], inplace=True)
    
    return df

def update_testtime(df_list):
    """
    Aktualisiert die 'Testtime[s]' Spalte eines DataFrames basierend auf der letzten 'Testtime[s]' des vorherigen DataFrames.
    """
    last_testtime = 0  # Startwert für die erste DataFrame in der Liste

    for df in df_list:
        # Date to Datetime
        df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = pd.to_datetime(df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'])
        # Berechnen des Zeitdeltas in Sekunden
        df['delta_time_s'] = df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'].diff().dt.total_seconds()

        # Setzen Sie das erste Delta auf 0, da es NaN ist
        df.loc[df.index[0], 'delta_time_s'] = 0

        # Aktualisieren der 'Testtime[s]' Spalte
        df['Testtime[s]'] = last_testtime + df['delta_time_s'].cumsum()

        # Aktualisieren des letzten Testtime-Werts für den nächsten DataFrame und Hinzufügen von 1 Sekunde
        last_testtime = df['Testtime[s]'].iloc[-1] + 1

        # Entfernen der Hilfsspalte delta_time_s
        df.drop(columns=['delta_time_s'], inplace=True)

    return df_list

def plot_testtime(dfs):
    """
    Plottet die DataFrames über 'Testtime[s]'.
    """
    plt.figure(figsize=(15, 7))
    
    # Kombinieren der DataFrames
    combined_df = pd.concat(dfs)
    
    # Plot der 'Voltage[V]' über 'Testtime[s]'
    plt.plot(combined_df['Testtime[s]'], combined_df['Voltage[V]'], label='Voltage[V]', color='blue')
    
    plt.xlabel('Testtime[s]')
    plt.ylabel('Voltage[V]')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cy_SOH(dfs):
    """
    Plottet die DataFrames über 'Testtime[s]'.
    """
    plt.figure(figsize=(15, 7))
    
    # Kombinieren der DataFrames
    combined_df = pd.concat(dfs)
    
    # Plot der 'Voltage[V]' über 'Testtime[s]'
    plt.plot(combined_df['EFC'], combined_df['SOH'], label='SOH', color='blue')
    
    plt.xlabel('EFC')
    plt.ylabel('SOH')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def interpolate_soh(df_cy_list, df_ct_list):
    """
    Interpoliert den SOH für jeden df_cy basierend auf den SOH-Werten in df_ct.
    """
    # Sicherstellen, dass die Anzahl der DataFrames in df_ct und df_cy gleich ist
    assert len(df_cy_list) == len(df_ct_list), "Die Anzahl der DataFrames in df_cy und df_ct muss gleich sein."

    # Für jeden df_cy (außer dem letzten) interpolieren
    for i in range(len(df_cy_list) - 1):
        start_soh = df_ct_list[i]['SOH'].iloc[0]  # SOH-Wert des aktuellen df_ct
        end_soh = df_ct_list[i + 1]['SOH'].iloc[0]  # SOH-Wert des nächsten df_ct

        # Interpolation über den gesamten df_cy DataFrame
        df_cy_list[i]['SOH'] = np.linspace(start_soh, end_soh, len(df_cy_list[i]))

    return df_cy_list

def calculate_efc(df_cy_list):
    """
    Berechnet EFC für jeden df_cy basierend auf first_df_c_actual und Q_sum.
    """
    global first_df_c_actual  # Verwenden des globalen Wertes first_df_c_actual

    Q_sum_cumulative = 0  # Startwert für die kumulative Q_sum

    for df in df_cy_list:
        # Berechnen des Zeitdeltas in Stunden
        df['delta_time_h'] = df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'].diff().dt.total_seconds() / 3600

        # Setzen Sie das erste Delta auf 0, da es NaN ist
        df.loc[df.index[0], 'delta_time_h'] = 0

        # Berechnung von Q für den aktuellen DataFrame
        df['Q'] = abs(df['delta_time_h'] * df['Current[A]'])

        # Aktualisieren des kumulativen Q_sum für den aktuellen DataFrame
        df['Q_sum'] = Q_sum_cumulative + df['Q'].cumsum()

        # Aktualisieren des kumulativen Q_sum für den nächsten DataFrame
        Q_sum_cumulative = df['Q_sum'].iloc[-1]

        # Berechnung von EFC für den aktuellen DataFrame
        df['EFC'] =  df['Q_sum']/first_df_c_actual

        # Entfernen der Hilfsspalten delta_time_h, Q und Q_sum
        #df.drop(columns=['delta_time_h', 'Q', 'Q_sum'], inplace=True)

    return df_cy_list

def assign_soh(df_ct, df_cu, df_profiles):
    """
    Ordnet den SOH-Wert aus df_ct den DataFrames in df_cu und df_profiles zu.
    """
    # Extrahieren des SOH-Werts aus jedem fünften DataFrame in df_ct
    soh_values = [df['SOH'].iloc[0] for i, df in enumerate(df_ct) if (i % 5) == 4]

    # Zuordnen des SOH-Werts zu den DataFrames in df_cu
    for i, df in enumerate(df_cu):
        df['SOH'] = soh_values[i]

    # Zuordnen des SOH-Werts zu den DataFrames in df_profiles (ab dem zweiten DataFrame)
    for i, df in enumerate(df_profiles[1:]):
        df['SOH'] = soh_values[i]

    return df_cu, df_profiles

def interpolate_dataframe(df):
    # 'Absolute_Time[yyyy-mm-dd hh:mm:ss]' als Index setzen
    df.set_index('Absolute_Time[yyyy-mm-dd hh:mm:ss]', inplace=True)

    # Duplikate basierend auf dem Index entfernen und den ersten Eintrag behalten
    df = df[~df.index.duplicated(keep='first')]

    df.index = pd.to_datetime(df.index)

    # Einen neuen Index mit 1-Sekunden-Intervallen erstellen
    new_index = pd.date_range(df.index[0], df.index[-1], freq='S')
    df_interpolated = pd.DataFrame(index=new_index)

    # Erneutes Reindizieren und Interpolieren für jede Spalte einzeln
    for column in df.columns:
        df_temp = df[column].reindex(new_index).interpolate(method='linear')
        df_interpolated[column] = df_temp

    # Index von df_interpolated zurücksetzen
    df_interpolated.reset_index(inplace=True)
    df_interpolated.rename(columns={"index": "Absolute_Time[yyyy-mm-dd hh:mm:ss]"}, inplace=True)

    return df_interpolated

import os

# Ursprungspfad
#source_path = r"c:\Users\Florian\Synology\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_parquet"
source_path = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_parquet'
# Zielverzeichnis
#destination_path = r"c:\Users\Florian\Synology\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_Dataframes"
destination_path = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_Dataframes'

# Liste der Unterordner im Ursprungspfad
subfolders = [f.name for f in os.scandir(source_path) if f.is_dir()]

# tqdm wird in die for-Schleife eingebaut, um den Fortschritt anzuzeigen
for subfolder in tqdm(subfolders, desc="Verarbeiten der Unterordner"):
    # Zielordner für den aktuellen Unterordner
    destination_subfolder_path = os.path.join(destination_path, subfolder)

    # Überprüfen, ob der Zielordner bereits existiert und ob die DataFrames darin gespeichert sind
    if os.path.exists(destination_subfolder_path) and all(os.path.exists(os.path.join(destination_subfolder_path, df_name + ".parquet")) for df_name in ["df", "df_profiles", "df_ct", "df_cu", "df_cy"]):
        print(destination_subfolder_path, 'already existing')
        continue  # Überspringen des aktuellen Unterordners, wenn die Bedingungen erfüllt sind

    # Pfad zum aktuellen Unterordner
    current_path = os.path.join(source_path, subfolder)
    
    # Verarbeiten Sie die Daten im aktuellen Unterordner (hier wird der zuvor erstellte Code verwendet)
    df = read_parquet_files(current_path)
    # Finden Sie den Index, bei dem Schedule_Test_ID zum ersten Mal den Wert 3 erreicht
    start_idx = df[df['Schedule_Step_ID'] == 3].index[0]
    
    # Beschneiden des DataFrames, um nur die Zeilen nach dem ersten Auftreten von Schedule_Step_ID=3 zu behalten
    df = df.loc[start_idx:].copy()
    
    # Initialisierung der globalen Variable
    first_df_c_actual = None
    
    # Anwendung der calculate_capacity_soh_qsum_efc Funktion
    df = calculate_capacity_soh_qsum_efc(df)
    
    # Weiterverarbeitung der Daten
    df['SOH'] = df['SOH'].interpolate(method='linear')
    df['Capacity[Ah]'] = df['Capacity[Ah]'].interpolate(method='linear')
    
    # Berechnung von SOC_c und SOC_m
    df['SOC_c'] = df['Q_c']/first_df_c_actual + 1
    df['SOC_m'] = df['Q_m']/first_df_c_actual + 1
    df = df[df['Schedule_Step_ID'].eq(52).cumsum().astype(bool)]
    print('df ready')
    
    df_ct = extract_capacity_test(df, range(5, 8))
    df_ct = [process_dataframe(dataframe) for dataframe in df_ct]
    df_ct = [interpolate_dataframe(dataframe) for dataframe in df_ct]  # Interpolation für df_ct
    #print(df_ct.columns)
    print('df_ct ready')
    
    df_cy = extract_cycles(df)
    df_cy = [interpolate_dataframe(dataframe) for dataframe in df_cy]  # Interpolation für df_cy
    
    df_profiles = extract_profiles(df, range(52, 60), range(57, 63))
    df_profiles = [interpolate_dataframe(dataframe) for dataframe in df_profiles]  # Interpolation für df_profiles
    
    df_cu = extract_checkups(df)
    df_cu = [interpolate_dataframe(dataframe) for dataframe in df_cu]  # Interpolation für df_cu


    df_ct = update_testtime(df_ct)
    print('df_ct ready')
    df_cy = update_testtime(df_cy)
    df_profiles = update_testtime(df_profiles)
    df_cu = update_testtime(df_cu)
    df_cy = interpolate_soh(df_cy, df_ct)
    df_cy = calculate_efc(df_cy)
    print('df_cy ready')
    df_cu, df_profiles = assign_soh(df_ct, df_cu, df_profiles)
    print('df_cu ready')
    print('df_profiles ready')
    print('saving dataframes ...')
    
    # Erstellen Sie den entsprechenden Unterordner im Zielverzeichnis
    destination_subfolder_path = os.path.join(destination_path, subfolder)
    os.makedirs(destination_subfolder_path, exist_ok=True)
    
    # Speichern Sie die verarbeiteten DataFrames im entsprechenden Unterordner
    df.to_parquet(os.path.join(destination_subfolder_path, "df.parquet"))
    pd.concat(df_profiles).to_parquet(os.path.join(destination_subfolder_path, "df_profiles.parquet"))
    pd.concat(df_ct).to_parquet(os.path.join(destination_subfolder_path, "df_ct.parquet"))
    pd.concat(df_cu).to_parquet(os.path.join(destination_subfolder_path, "df_cu.parquet"))
    pd.concat(df_cy).to_parquet(os.path.join(destination_subfolder_path, "df_cy.parquet"))
    print(subfolder, 'complete')


# # Check if all is correct and all Timeintervalls are in 1 second: 
# # Dateipfad
# file_path = r"C:\Users\Florian\Florian\3_Projekte\MGFarm_privat\State_Estimation\LFP Daten\Versuch_18650_standart\MGFarm_18650_Dataframes\MGFarm_18650_C01\df_profiles.parquet"
# 
# # DataFrame laden
# df = pd.read_parquet(file_path)
# 
# # Überprüfen der Differenzen in der Spalte 'Testtime'
# time_diffs = df['Testtime[s]'].diff().dropna()
# non_one_second_intervals = time_diffs[time_diffs != 1]
# 
# if non_one_second_intervals.empty:
#     print("Alle Zeitintervalle betragen genau 1 Sekunde.")
# else:
#     print("Es gibt einige Zeitintervalle, die nicht genau 1 Sekunde betragen in den Zeilen:")
#     print(non_one_second_intervals.index.tolist())
