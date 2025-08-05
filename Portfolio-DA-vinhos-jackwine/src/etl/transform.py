def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ","_")
        .str.replace("-","_")
    )
    return df

def remove_nulls(df):
    return df.dropna()
