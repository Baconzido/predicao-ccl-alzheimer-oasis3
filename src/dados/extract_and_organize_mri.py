#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para descompactar arquivos ZIP do OASIS3 e organizar os arquivos NIfTI.
"""

import os
import sys
import zipfile
import shutil
import glob
from pathlib import Path

# Configura a codificação para UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def extract_subject_id(filename):
    """
    Extrai o ID do sujeito do nome do arquivo.
    Exemplo: OAS30006_MR_d0166.zip -> OAS30006
    """
    basename = os.path.basename(filename)
    # Remove a extensão .zip
    name_without_ext = basename.replace('.zip', '')
    # Pega a parte antes do primeiro underscore
    subject_id = name_without_ext.split('_')[0]
    return subject_id


def find_nifti_file(extracted_dir, scan_name):
    """
    Procura pelo arquivo .nii.gz no caminho especificado.
    Caminho: scans/anat3-T1w/resources/NIFTI/files/*.nii.gz
    Retorna uma lista de todos os arquivos encontrados.
    """
    # Primeiro tenta encontrar especificamente anat3-T1w
    search_path = os.path.join(
        extracted_dir,
        scan_name,
        'scans',
        'anat3-T1w',
        'resources',
        'NIFTI',
        'files',
        '*.nii.gz'
    )

    nifti_files = glob.glob(search_path)

    # Se não encontrar, tenta anat2-T1w
    if not nifti_files:
        search_path = os.path.join(
            extracted_dir,
            scan_name,
            'scans',
            'anat2-T1w',
            'resources',
            'NIFTI',
            'files',
            '*.nii.gz'
        )
        nifti_files = glob.glob(search_path)

    # Se ainda não encontrar, tenta qualquer pasta com padrão anat*T1w
    if not nifti_files:
        search_path = os.path.join(
            extracted_dir,
            scan_name,
            'scans',
            'anat*T1w',
            'resources',
            'NIFTI',
            'files',
            '*.nii.gz'
        )
        nifti_files = glob.glob(search_path)

    return nifti_files


def process_zip_files(output_dir='oasis3_mri_data'):
    """
    Processa todos os arquivos ZIP no diretório atual.
    """
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Encontra todos os arquivos ZIP que correspondem ao padrão OAS
    zip_files = glob.glob('OAS*.zip')

    if not zip_files:
        print("Nenhum arquivo ZIP encontrado no diretório atual.")
        return

    print(f"Encontrados {len(zip_files)} arquivos ZIP para processar.\n")

    temp_extract_dir = 'temp_extract'

    for zip_file in sorted(zip_files):
        print(f"Processando: {zip_file}")

        # Extrai o ID do sujeito
        subject_id = extract_subject_id(zip_file)
        print(f"  ID do sujeito: {subject_id}")

        # Cria pasta de destino para este sujeito
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)

        try:
            # Descompacta o arquivo ZIP em um diretório temporário
            os.makedirs(temp_extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)

            # Nome do scan é o mesmo do arquivo sem .zip
            scan_name = os.path.basename(zip_file).replace('.zip', '')

            # Procura pelos arquivos NIfTI
            nifti_files = find_nifti_file(temp_extract_dir, scan_name)

            if nifti_files:
                # Se houver múltiplos arquivos, pega o primeiro (geralmente anat3 ou run-02)
                for idx, nifti_file in enumerate(nifti_files):
                    # Nome do arquivo de destino
                    if len(nifti_files) > 1:
                        output_filename = f"{subject_id}_run{idx+1:02d}.nii.gz"
                    else:
                        output_filename = f"{subject_id}.nii.gz"

                    output_path = os.path.join(subject_output_dir, output_filename)

                    # Copia o arquivo para o destino
                    shutil.copy2(nifti_file, output_path)
                    print(f"  [OK] Arquivo copiado para: {output_path}")
            else:
                print(f"  [ERRO] Arquivo .nii.gz nao encontrado no caminho esperado")

            # Remove o diretório temporário
            shutil.rmtree(temp_extract_dir)

        except Exception as e:
            print(f"  [ERRO] Erro ao processar {zip_file}: {str(e)}")
            # Tenta limpar o diretório temporário em caso de erro
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)

        print()

    print("Processamento concluído!")
    print(f"Arquivos organizados em: {output_dir}")


if __name__ == "__main__":
    # Obtém o diretório atual
    current_dir = os.getcwd()
    print(f"Diretório de trabalho: {current_dir}\n")

    # Processa os arquivos ZIP
    process_zip_files()
