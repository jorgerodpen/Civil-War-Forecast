#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This document contains all basic code to run the Civil War Forecasting algorithm. 
Every function is described with its specific inputs and outputs. 
"""

import zipfile
import pandas as pd
from io import BytesIO
import numpy as np
from country_converter import CountryConverter

def read_events_year(zipfilepath, year):
    """
    Reads all the events in the selected year by extracting them from the selected folder.

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    year : int
        Year of the data we want to load as a dataframe.
        The data ranges from 1995 to 2020.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe containing all events of the selected year.

    """
    
    # Basic variables for the algorithm
    year = str(year)
    isinzip = []
    
    # Opening the zip file
    with zipfile.ZipFile(zipfilepath) as mainfile:
        
        # Extracting all the files in it
        listfiles = mainfile.namelist()
        for file in listfiles:
            isinzip.append(year in file)
            
        # Select the zip file that corresponds to our selected year
        filename = [i for indx,i in enumerate(listfiles) if isinzip[indx] == True][0]
        zfilename = BytesIO(mainfile.read(filename))
        
        # Opening the zip file of the selected year
        with zipfile.ZipFile(zfilename) as zfile:
            filename2 = zfile.namelist()[0]
            
            # Saving the data to a new dataframe
            with zfile.open(filename2) as file:
                df = pd.read_table(file, encoding='ISO-8859-1', low_memory=False, converters={'CAMEO Code':str})
    
    return df

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def internal_events_year(zipfilepath, year):
    """
    Returns a dataframe containing events of the selected year.
    Selected events have the same source and target country. 
    Unnecessary columns "Story ID", "Sentence Number", "Publisher" are dropped.
    "Year" and "Month" columns are also added in this step. 

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    year : int
        Year of the data we want to load as a dataframe.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe containing all events of the selected year with same source 
        and target country. 
        "Year" and "Month" columns are added.. 

    """
    
    # Reading the file
    df = read_events_year(zipfilepath, year)
    
    # Selecting only events with same source and target country
    df = df[df["Source Country"] == df["Target Country"]]
    
    # Adding month and year columns
    try:
        df["Event Date"]=pd.to_datetime(df["Event Date"], format="%Y-%m-%d")
    except:
        df["Event Date"]=pd.to_datetime(df["Event Date"], format="%m/%d/%Y")
    df["Year_Month"] = df["Event Date"].dt.to_period('M')
    
    return df

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def sector_filter(variable):
    """
    Cleans a selected variable assigning it to one of the four groups defined:
        Government
        Opposition
        Insurgents
        People

    Parameters
    ----------
    variable : str
        Containing information about source or target sector

    Returns
    -------
    variable : str
        With values "Government", "Opposition", "Insurgents" or "People".
        Depends on its original value

    """
    # List of words associated with Government
    gov = ["Executive", 
           "Government Major Party (In Government)", 
           "Government Minor Party (In Government)",
           "Government Provincial Party (In Government)", 
           "Government Municipal Party (In Government)",
           "State-Owned Enterprises", 
           "State-Owned", 
           "Legislative", 
           "Judicial", 
           "Government Religious",
           "State", 
           "Ministry",
           "National",
           "Local", 
           "Regional",
           "Region",
           "Provincial", 
           "Province",
           "Municipal",
           "Municipality",
           "Elite,Government", 
           "Government,Elite", 
           "Police", 
           "Military", 
           "Government,Police", 
           "Government,Police", 
           "Military,Government", 
           "Government,Military",
           "Upper House",
           "Congress",
           "Diplomatic"]
    
    # List of words associated with Opposition
    opp = ["Opposition", 
           "Opposition Major Party (Out Of Government)", 
           "Opposition Minor Party (Out Of Government)",
           "Opposition Provincial Party (Out Of Government)", 
           "Opposition Municipal Party (Out Of Government)",
           "Mobs", 
           "Protestors", 
           "Popular Opposition",
           "Parties",
           "Union",
           "Activists"]
    
    # List of words associated with Insurgents
    ins = ["Insurgents", 
           "Banned Parties", 
           "Gangs", 
           "Anarchist", 
           "Rebel", 
           "Organized Violent", 
           "Radicals", 
           "Separatists", 
           "Criminals", 
           "Fundamentalists", 
           "Extremists", 
           "Exiles", 
           "Dissident",
           "Violent",
           "Unidentified Forces"]
    
    # List of words associated with People
    ppl = ["Civilian", 
           "Education", 
           "Labor", 
           "Refugees", 
           "Displaced", 
           "National Ethnic", 
           "Agricultural",
           "Population",
           "People",
           "Bussines",
           "Citizen",
           "Villager",
           "Lawyer",
           "Militia",
           "Medical",
           "Legal",
           "Social"]
    
    # Checking if any of those words are in the column
    if type(variable) != str: # skipping NaNs
        pass
    elif any(word_gov in variable for word_gov in gov):
        variable = "Gov"
    elif variable == "Government":
        variable = "Gov"
    elif any(word_opp in variable for word_opp in opp):
        variable = "Opp"
    elif any(word_ins in variable for word_ins in ins):
        variable = "Ins"
    elif any(word_ppl in variable for word_ppl in ppl):
        variable = "Peo"
    else:
        variable = np.nan
    
    return variable

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def read_filtered_data(zipfilepath, year):
    """
    Returns a dataframe containing events of the selected year.
    Selected events have the same source and target country. 
    "Year" and "Month" columns are added in this step. 
    The Source and Target Sectors are classified as Government, Opposition,
    Insurgents or People. 

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    year : int
        Year of the data we want to load as a dataframe.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe containing all events of the selected year with same source 
        and target country. 
        Unnecessary columns "Story ID", "Sentence Number", "Publisher" are dropped.
        "Year" and "Month" columns are also included. 
        The Source and Target Sectors are classified as Government, Opposition,
        Insurgents or People. 

    """
    # Reading dataframe
    df = internal_events_year(zipfilepath, year)
    
    # Transforming sectors
    df["Source Sectors"] = df["Source Sectors"].apply(sector_filter)
    df["Target Sectors"] = df["Target Sectors"].apply(sector_filter)
    
    # Transforming Sectors classified as country to Government
    df.loc[df["Source Name"] == df["Source Country"],"Source Sectors"] = "Gov"
    df.loc[df["Target Name"] == df["Target Country"],"Target Sectors"] = "Gov"
    
    return df

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def read_cols_filtered(zipfilepath, year):
    """
    Returns a dataframe containing events of the selected year.
    Selected events have the same source and target country. 
    The Source and Target Sectors are classified as Government, Opposition,
    Insurgents or People. 
    Only selects the following columns: "Source Sectors", "Event Text", "CAMEO Code",
    "Intensity", "Target Sectors", "Country", "Year_Month".

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    year : int
        Year of the data we want to load as a dataframe.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe containing all events of the selected year with same source 
        and target country. 
        Unnecessary columns "Story ID", "Sentence Number", "Publisher" are dropped.
        "Year" and "Month" columns are also included. 
        The Source and Target Sectors are classified as Government, Opposition,
        Insurgents or People. 
    """
    columns = ["Source Country",
               "Source Sectors", 
               "CAMEO Code",
               "Intensity", 
               "Target Sectors",  
               "Year_Month"]
    renamer = {"Source Sectors": "Source",
               "CAMEO Code": "CAMEO",
               "Target Sectors": "Target",
               "Source Country": "Country"}
    df = read_filtered_data(zipfilepath, year)
    
    # Selecting columns
    df = df[columns].rename(columns=renamer)
    
    # Different source and target
    df = df[df["Source"]!=df["Target"]].dropna()
    
    return df

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def source_target_interaction(zipfilepath, year):
    """
    Returns a dataframe containing events of the selected year.
    Selected events have the same source and target country. 
    The Source and Target Sectors are classified as Government, Opposition,
    Insurgents or People. 
    Only selects the following columns: "Source Sectors", "Event Text", "CAMEO Code",
    "Intensity", "Target Sectors", "Country", "Year_Month".
    Creates dummy variables for interaction between sectors

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    year : int
        Year of the data we want to load as a dataframe.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe containing all events of the selected year with same source 
        and target country. 
        Unnecessary columns "Story ID", "Sentence Number", "Publisher" are dropped.
        "Year" and "Month" columns are also included. 
        The Source and Target Sectors are classified as Government, Opposition,
        Insurgents or People. 
        Includes dummy variables for interactions between sectors.
    """
    # Reading the dataset
    df = read_cols_filtered(zipfilepath, year)
    
    # Getting dummy variables
    df_2 = pd.get_dummies(df, columns = ["Source", "Target"])
    
    # Generating dummy variables for interaction
    source_cols = ["Source_Gov", "Source_Ins", "Source_Opp", "Source_Peo"]
    target_cols = ["Target_Gov", "Target_Ins", "Target_Opp", "Target_Peo"]
    
    for source in source_cols:
        for target in target_cols: 
            source2 = source[-3:]
            target2 = target[-3:]
            col_name = source2 + "_" + target2
            source_target = df_2[source]*df_2[target]
            df[col_name] = source_target
    
    # Removing redundant variables
    df.drop(["Source", "Target", "Gov_Gov", "Ins_Ins", "Opp_Opp", "Peo_Peo"], 
            axis = 1, inplace = True)
    
    return df

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def iso3country(zipfilepath, year):
    """
    Returns a dataframe containing events of the selected year.
    Selected events have the same source and target country. 
    The Source and Target Sectors are classified as Government, Opposition,
    Insurgents or People. 
    Only selects the following columns: "Source Sectors", "Event Text", "CAMEO Code",
    "Intensity", "Target Sectors", "Country", "Year_Month".
    Creates dummy variables for interaction between sectors.
    Adds the ISO3 country code to each event. 

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    year : int
        Year of the data we want to load as a dataframe.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe containing all events of the selected year with same source 
        and target country. 
        Unnecessary columns "Story ID", "Sentence Number", "Publisher" are dropped.
        "Year_Month" columns are also included. 
        The Source and Target Sectors are classified as Government, Opposition,
        Insurgents or People. 
        Includes dummy variables for interactions between sectors.
        Includes ISO3 country code. 
    """
    # Reading the dataset
    df = source_target_interaction(zipfilepath, year)
    
    # Creating map
    iso_map = {country: country_to_iso3(country) for country in df["Country"].unique()}
    
    # Mapping
    df["ISO3"] = df["Country"].map(iso_map)
    
    # Reordering columns
    cols = list(df.columns)
    cols = cols[0:1]+cols[-1:]+cols[3:4]+cols[1:3]+cols[4:-1]
    
    return df[cols]

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
def country_to_iso3(country):
    """
    Converts an input country to its ISO3 country code. 
    
    Parameters
    ----------
    country : str
        Country name 

    Returns
    -------
    ISO3 : ISO3 country code. 
    """
    converter = CountryConverter()
    if country == "CuraÃ§ao":
        return "CUW"
    elif country == "Bonaire":
        return "BES"
    elif country == "Yugoslavia":
        return "SRB"
    else:
        try:
            ISO3 = converter.convert(country, to='ISO3')
        except Exception:
            return np.nan
        if ISO3 == "not found":
            return np.nan
        else:
            return ISO3
        
        
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def interaction_fraction(zipfilepath, pitffilepath):
    """
    Generates the "Interaction Fraction" model.
    
    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    pitffilepath : str
        File path of the PITF generated excel file.

    Returns
    -------
    model : pandas.core.frame.DataFrame
        DataFrame containing the "Interaction Fraction" model
    """
    print("Generating model...")
    # Reading the data
    df = pd.DataFrame()
    for i in range(1995, 2019):
        newdf = iso3country(zipfilepath, i)
        df = pd.concat([df, newdf])
        
    cols = ["Gov_Ins", "Gov_Opp", "Gov_Peo", "Ins_Gov", "Ins_Opp", "Ins_Peo", 
           "Opp_Gov", "Opp_Ins", "Opp_Peo", "Peo_Gov", "Peo_Ins", "Peo_Opp"]
    
    pd.options.mode.chained_assignment = None
    model1 = df[["ISO3","Year_Month"]]
    model1[cols] = df[cols]
    
    # Aggregating
    modelg = model1.groupby(["ISO3","Year_Month"]).sum().reset_index()
    modelg.loc[:,cols] = modelg.loc[:,cols].apply(lambda x: x/x.sum(), axis=1).fillna(0)
    
    # Cleaning missing years
    print("Cleaning missing years...")
    
    # Unique values
    unique_iso3 = modelg.ISO3.unique()
    unique_date = modelg.Year_Month.unique()
    
    # Dataframe to filter missing Year_Month values
    list_original = []
    for iso3 in unique_iso3:
        for date in unique_date:
            list_original.append([iso3, date])
    year_month = pd.DataFrame(list_original)
    
    # Adding missing values
    final = modelg.merge(year_month, 
                     left_on=["ISO3", "Year_Month"], 
                     right_on=[0,1], how="outer").drop([0,1], axis=1)
    
    # Sorting and substituting missing values
    finals = final.sort_values(["ISO3","Year_Month"]).reset_index(drop=True).fillna(0)
    
    # Adding Civil War columns
    print("Adding civil wars...")
    model = add_cw(finals, pitffilepath)
    
    print("Done!")
    return model        
    
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def mean_intensity(zipfilepath, pitffilepath):
    """
    Generates the "Mean Intensity" model.
    
    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    pitffilepath : str
        File path of the PITF generated excel file.

    Returns
    -------
    model : pandas.core.frame.DataFrame
        DataFrame containing the "Mean Intensity" model
    """
    print("Generating model...")
    # Reading the data
    df = pd.DataFrame()
    for i in range(1995, 2019):
        newdf = iso3country(zipfilepath, i)
        df = pd.concat([df, newdf])
    
    # Looping over this columns
    cols = ["Gov_Ins", "Gov_Opp", "Gov_Peo", "Ins_Gov", "Ins_Opp", "Ins_Peo", 
           "Opp_Gov", "Opp_Ins", "Opp_Peo", "Peo_Gov", "Peo_Ins", "Peo_Opp"]

    # Generating the model
    model1 = df.loc[:,["ISO3", "Year_Month"]]
    for col in cols:
        model1.loc[:,col] = df.loc[:,col]*df.loc[:,"Intensity"]
    
    # Aggregating
    modelg = model1.groupby(["ISO3","Year_Month"]).mean().reset_index()
    
    # Cleaning missing years
    print("Cleaning missing years...")
    
    # Unique values
    unique_iso3 = modelg.ISO3.unique()
    unique_date = modelg.Year_Month.unique()
    
    # Dataframe to filter missing Year_Month values
    list_original = []
    for iso3 in unique_iso3:
        for date in unique_date:
            list_original.append([iso3, date])
    year_month = pd.DataFrame(list_original)
    
    # Adding missing values
    final = modelg.merge(year_month, 
                     left_on=["ISO3", "Year_Month"], 
                     right_on=[0,1], how="outer").drop([0,1], axis=1)
    
    # Sorting and substituting missing values
    finals = final.sort_values(["ISO3","Year_Month"]).reset_index(drop=True).fillna(0)
    
    # Adding Civil War columns
    print("Adding civil wars...")
    model = add_cw(finals, pitffilepath)
    
    print("Done!")
    return model


    
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def cameo_fraction(zipfilepath, pitffilepath):
    """
    Generates the "CAMEO Fraction" model.
    
    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    pitffilepath : str
        File path of the PITF generated excel file.

    Returns
    -------
    model : pandas.core.frame.DataFrame
        DataFrame containing the "CAMEO Fraction" model
    """
    print("Generating model...")
    # Reading the data
    df = pd.DataFrame()
    for i in range(1995, 2019):
        newdf = iso3country(zipfilepath, i)
        df = pd.concat([df, newdf])
    
    # Looping over this columns
    cols = ["Gov_Ins", "Gov_Opp", "Gov_Peo", "Ins_Gov", "Ins_Opp", "Ins_Peo", 
           "Opp_Gov", "Opp_Ins", "Opp_Peo", "Peo_Gov", "Peo_Ins", "Peo_Opp"]

    # Generating the model
        # Selecting two first digits
    df["CAMEO"] = df["CAMEO"].apply(lambda x: x[:2]).astype("str")
    
        # Cameo model:
    for col in cols:
        df[col] = df[col]*df["CAMEO"]
        df[col] = df[col].apply(cleaning_cameo)
        
        # Final model
    model1 = df.drop(["Country","CAMEO","Intensity"], axis=1)
    cameo = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20"]
    for col in cols:
        for code in cameo:
            col_name = col+'_'+code
            model1[col_name] = (df[col] == code).astype("int64")
    
        # Deleting irrelevant columns
    model1 = model1.drop(cols, axis=1)    
    # Aggregating
    modelg = model1.groupby(["ISO3","Year_Month"]).sum().reset_index()
    
    #Fraction
    model_columns = modelg.columns[2:-6]
    modelg.loc[:,model_columns] = modelg.loc[:,model_columns].apply(lambda x: x/x.sum(), axis=1).fillna(0)
    
    # Cleaning missing years
    print("Cleaning missing years...")
    
    # Unique values
    unique_iso3 = modelg.ISO3.unique()
    unique_date = modelg.Year_Month.unique()
    
    # Dataframe to filter missing Year_Month values
    list_original = []
    for iso3 in unique_iso3:
        for date in unique_date:
            list_original.append([iso3, date])
    year_month = pd.DataFrame(list_original)
    
    # Adding missing values
    final = modelg.merge(year_month, 
                     left_on=["ISO3", "Year_Month"], 
                     right_on=[0,1], how="outer").drop([0,1], axis=1)
    
    # Sorting and substituting missing values
    finals = final.sort_values(["ISO3","Year_Month"]).reset_index(drop=True).fillna(0)
    
    # Adding Civil War columns
    print("Adding civil wars...")
    model = add_cw(finals, pitffilepath)
    
    print("Done!")
    return model


# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
def cleaning_cameo(cameo_val):
    """
    Generates the columns for each model to predict Civil Wars.
    
    Parameters
    ----------
    cameo_val : str
        Value of the two-digited CAMEO code 
    Returns
    -------
    cameo_val : str / np.nan
        Fills missing values with nans
    """
    try:
        if len(cameo_val)==0:
            return np.nan
        else:
            return cameo_val
    except:
        return np.nan


# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def add_cw(model, pitffilepath):
    """
    Generates the columns for each model to predict Civil Wars.
    
    Parameters
    ----------
    model : pandas.core.frame.DataFrame
        DataFrame containing the model.
    pitffilepath : str
        File path of the PITF generated excel file.

    Returns
    -------
    model : pandas.core.frame.DataFrame
        DataFrame containing the model with the Civil Wars variables added. 
    """
    
    # Reading PITF
    PITF = read_PITF(pitffilepath)
    
    # Adding columns
    model.loc[:,"CW_s"] = np.nan
    model.loc[:,"CW_f"] = np.nan
    model.loc[:,"CW_o"] = np.nan

    for idx, row in PITF.iterrows():
        # Getting names
        iso3 = row[0]
        start = row[1]
        finish = row[2]

        # Boolean operators
        condition_s = (model["ISO3"] == iso3) & (model["Year_Month"]== start)
        condition_f = (model["ISO3"] == iso3) & (model["Year_Month"]== finish)
        condition_o = (model["ISO3"] == iso3) &  (model["Year_Month"]>start) & \
                            (model["Year_Month"]<finish)

        # Start
        try:
            model.loc[condition_s,"CW_s"] = 1

        except:
            pass

        # Finish
        if finish == max(model["Year_Month"]):
            model.loc[condition_f,"CW_o"] = 1
        else:
            model.loc[condition_f,"CW_f"] = 1

        # Ongoing
        model.loc[condition_o, "CW_o"] = 1
     
    # Fill with 0s
    model = model.fillna(0)
   
    # Adding targets
    countries = model["ISO3"].unique()
    model.loc[:,"CW_s_plus1"] = np.nan
    model.loc[:,"CW_f_plus1"] = np.nan
    model.loc[:,"CW_o_plus1"] = np.nan

    pd.options.mode.chained_assignment = None
    for country in countries:
        country_filter = model["ISO3"]==country
        country_only = model.loc[country_filter,:]
        country_only.loc[:,"CW_s_plus1"] = country_only["CW_s"].shift(-1).fillna(method="ffill")
        country_only.loc[:,"CW_f_plus1"] = country_only["CW_f"].shift(-1).fillna(method="ffill")
        country_only.loc[:,"CW_o_plus1"] = country_only["CW_o"].shift(-1).fillna(method="ffill")
        model.loc[model["ISO3"]==country,:] = country_only.loc[:,:]
    
    # Final targets
    model.iloc[:,-3:] = model.iloc[:,-3:].astype("int64").astype("str").astype("str")
    model.loc[:,"CW_plus1"] = model.iloc[:,-3].astype("str") + model.iloc[:,-2].astype("str") + model.iloc[:,-1].astype("str")
    model.loc[:,"CW_plus1"] = model.loc[:,"CW_plus1"].replace({"110":"100"})
    
    return model

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #

def read_PITF(filepath):
    """
    Reads the generated excel file from https://smallpdf.com/pdf-to-excel for
    the PITF pdf containing 

    Parameters
    ----------
    filepath : str
        File path of the PITF generated excel file.

    Returns
    -------
    PITF: pandas.core.frame.DataFrame
        Returns dataframe containing a list of civil wars as defined by the PITF

    """
    # First table
    # Reading the document
    PITF = pd.read_excel(filepath, sheet_name="Table 1",
                        header = 1)

    # Selecting relevant columns
    PITF = PITF[["Country", "Began", "Unnamed: 8", "Ended"]]

    # Replacing NaNs
    PITF.loc[PITF["Began"].isnull(),"Began"] = PITF.loc[PITF["Began"].isnull(),"Unnamed: 8"]
    PITF = PITF.drop("Unnamed: 8", axis=1).fillna(method="ffill")
    PITF.loc[0,"Ended"]="—"

    # Ongoing conflict as of 31st of December 2018. 
    PITF["Ended"] = PITF["Ended"].replace({"—":"12/2018"})

    # Extracting month and year 
    PITF["Year_Month_S"] = pd.to_datetime(PITF["Began"], format="%m/%Y").dt.to_period('M')
    PITF["Year_Month_F"] = pd.to_datetime(PITF["Ended"], format="%m/%Y").dt.to_period('M')
    PITF = PITF[["Country", "Year_Month_S", "Year_Month_F"]]

    # Second table
    # Reading the document
    PITF2 = pd.read_excel("PITF Consolidated Case List 2018-converted.xlsx", sheet_name="Table 2",
                        header = 1)

    # Selecting relevant columns
    PITF2 = PITF2.loc[0:6,["Country","Began","Ended"]]

    # Replacing NaNs
    PITF2 = PITF2.fillna(method="ffill")

    # Ongoing conflict as of 31st of December 2018. 
    PITF2["Ended"] = PITF2["Ended"].replace({"—":"12/2018"})

    # Extracting month and year 
    PITF2["Year_Month_S"] = pd.to_datetime(PITF2["Began"], format="%m/%Y").dt.to_period('M')
    PITF2["Year_Month_F"] = pd.to_datetime(PITF2["Ended"], format="%m/%Y").dt.to_period('M')
    PITF2 = PITF2[["Country", "Year_Month_S", "Year_Month_F"]]

    # Concatenating both datasets
    final = pd.concat([PITF,PITF2]).reset_index(drop=True)   
    
    
    # Filtering wars from 1995
    final = final[final["Year_Month_F"].dt.year>=1995]
    
    # Country to ISO3
    # Creating map
    iso_map = {country: country_to_iso3(country) for country in final["Country"].unique()}
    
    # Mapping
    final["ISO3"] = final["Country"].map(iso_map)
    
    return final[["ISO3", "Year_Month_S", "Year_Month_F"]]

# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #
# =====*****+++++=====*****+++++=====*****+++++=====*****+++++ #  

def all_events(zipfilepath, startyear=1995, finalyear=2020):
    """

    Parameters
    ----------
    zipfilepath : str
        Path of the .zip file that contains the .zip files for every year.
    startyear : int, optional
        Initial year to start the loop. The default is 1995.
    finalyear : int, optional
        Final year to finish the loop. The default is 2020.

    Returns
    -------
    events : pandas.core.frame.DataFrame
        Dataframe containing the number of events per month for the selected years

    """
    events = pd.DataFrame()
    print("Loading years...")
    for year in range(startyear,finalyear+1):
        events = pd.concat([events,iso3country(zipfilepath,year)])
    print("Done!")
    return events