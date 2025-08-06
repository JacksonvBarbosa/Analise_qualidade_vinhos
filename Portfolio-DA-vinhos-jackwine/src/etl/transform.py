def clean_columns(df):
    '''
        Substitui espaços e hifen das colunas, por underscore
    '''
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ","_")
        .str.replace("-","_")
    )
    return df

def remove_nulls(df):
    '''
        Remover nulos do dataframe
    '''
    return df.dropna()

def rename_colunns(df, new_list):
    '''
        Renomear colunas caso necessário.
        inserir o dataframe.
        Inserir a lista com os novos nomes.
    '''
    renomeia_colunas = {col: new_list[i] for i, col in enumerate(df.columns)}
    df_rename = df.rename(columns = renomeia_colunas)
    return df_rename
