import boto3
import pandas as pd
import io
import s3fs
import os

# Importar dados do S3
def importar_arquivos_s3(
    s3_bucket: str,
    s3_prefix: str,
    arquivos_desejados: list = None,
    sep: str = ","
) -> pd.DataFrame:
    """
    Importa arquivos do S3 de uma camada específica (bronze, silver ou gold).
    Funciona tanto para CSV quanto Parquet.
    
    Params:
        s3_bucket (str): Nome do bucket.
        s3_prefix (str): Prefixo da camada (ex: 'tech_3/bronze/').
        arquivos_desejados (list): Lista com nomes de arquivos desejados. 
                                    Se None, puxa todos.
        sep (str): Separador do arquivo CSV (default: ',').
    
    Returns:
        DataFrame consolidado com todos os arquivos lidos.
    """
    
    s3_client = boto3.client("s3")
    
    # Lista objetos no prefixo
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    lista_dfs = []
    
    for obj in response.get("Contents", []):
        file_key = obj["Key"]
        file_name = file_key.split("/")[-1]
        
        if (arquivos_desejados is None) or (file_name in arquivos_desejados):
            print(f"📥 Lendo s3://{s3_bucket}/{file_key}")
            
            obj_data = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
            file_bytes = io.BytesIO(obj_data["Body"].read())
            
            # Detectar extensão
            if file_name.lower().endswith(".csv"):
                try:
                    df_temp = pd.read_csv(file_bytes, sep=sep)
                except pd.errors.EmptyDataError:
                    print(f"⚠️ Arquivo CSV vazio: {file_name}")
                    continue
            elif file_name.lower().endswith(".parquet"):
                try:
                    df_temp = pd.read_parquet(file_bytes)
                except Exception as e:
                    print(f"❌ Erro ao ler Parquet {file_name}: {e}")
                    continue
            else:
                print(f"⚠️ Formato não suportado: {file_name}, ignorando...")
                continue
            
            lista_dfs.append(df_temp)
    
    if not lista_dfs:
        print("⚠️ Nenhum arquivo encontrado!")
        return pd.DataFrame()
    
    df_consolidado = pd.concat(lista_dfs, ignore_index=True)
    print(f"\n✅ Consolidação concluída: {len(df_consolidado)} linhas")
    
    return df_consolidado


# Salvar dados no S3
def salvar_s3_parquet(df: pd.DataFrame, caminho_s3: str, engine: str = "fastparquet"):
    """
    Salva um DataFrame no S3 em formato Parquet.
    
    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        caminho_s3 (str): Caminho completo no S3 (ex: 's3://bucket/silver/arquivo.parquet').
        engine (str): Motor de escrita parquet, default 'fastparquet'.
    """
    print(f"\n💾 Salvando dados no S3: {caminho_s3}")

    fs = s3fs.S3FileSystem(
        key=os.getenv('AWS_ACCESS_KEY_ID'),
        secret=os.getenv('AWS_SECRET_ACCESS_KEY'),
        token=os.getenv('AWS_SESSION_TOKEN'),
        client_kwargs={"region_name": os.getenv('AWS_REGION')}
    )

    try:
        with fs.open(caminho_s3, "wb") as f:
            df.to_parquet(f, index=False, engine=engine)
        print(f"✅ Sucesso! Dados salvos em: {caminho_s3}")
    except Exception as e:
        print(f"❌ Erro ao salvar dados em {caminho_s3}: {e}")
