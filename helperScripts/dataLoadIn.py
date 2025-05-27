import pandas as pd

def dataLoad(url, type):
    """Load in datasets by url
    
    Parameters
    ----------
    url : str
        address where data is located
    type : str
        data type, indicates the file type
    
    Output
    ------
    df : pandas dataframe
        the dataframe retrieved from the link
    """
    # Load the dataset directly from the URL
    try:
        if(type=="rna"):
            # Read the compressed CSV file
            df = pd.read_csv(url, compression='gzip')
        else:
            df = pd.read_excel(url, engine='openpyxl')
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
    return df
