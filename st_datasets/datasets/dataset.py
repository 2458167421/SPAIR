import scanpy as sc
import numpy as np
import time
import os

from .utils import check_file_location


home_dir = os.environ['HOME']
dataset_url = 'https://huggingface.co/datasets/han-shu/st_datasets/resolve/main'


def datasets():
    print(">>> INFO: find available datasets:")
    for key, value in globals().items(): 
        if (callable(value) and key.startswith('get_') and 'get_data' != key): 
            print(value)


def get_data(dataset_func=None, top_genes=8000, preprocess=True, **args):
    start_time = time.time()
    assert (dataset_func), '>>> ERROR: You must appoint a function!'

    if (dataset_func in globals().values()):
        adata, n_cluster, dataset_details = dataset_func(**args)
    else:
        adata = dataset_func(**args)
        n_cluster = None
        dataset_details = str(dataset_func)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_genes)
    if (preprocess):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if ('cluster' in adata.obs):
        adata.obs['cluster'] = adata.obs['cluster'].astype(str)
    print('>>> INFO: dataset name: {}, size: ({}, {}), cluster: {}.({:.3f}s)'.format(
        dataset_details, adata.X.shape[0], adata.X.shape[1], n_cluster, time.time() - start_time
    ))

    return adata, n_cluster


def get_dlpfc_data(id, path=None):
    section_list = ['151507', '151508', '151509', '151510', '151669', '151670', 
                    '151671', '151672', '151673', '151674', '151675', '151676','151777',
                    '151777RNA','151778RNA','151777ADT','151778ADT','151777RNA1','151778RNA1','151778RNA2','151778ADT1','151778ADT2','151779H3K4','151780H3K4',
                    '151779RNA','151780RNA',
                    'mouse_Thymus1_RNA','mouse_Thymus2_RNA',
                    'mouse_Thymus1_ADT','mouse_Thymus2_ADT',
                    'mouse_Thymus3_RNA','mouse_Thymus4_RNA',
                    'mouse_Thymus3_ADT','mouse_Thymus4_ADT',
                    '151785RNA','151786RNA',#86p22
                    #'151787RNA','151788RNA','151789RNA',#E13E15E18
                    '151790RNA',
                    '151791_7','151791_7O',
                    '151786RNA1',
                    '151786RNA2',
                    '151785ADT','151786ADT',
                    '151792p21_mouse_brain_RNA','151792p22_mouse_brain_RNA',
                    '151785ADTP21atac_pre','151786ADTP22atac_pre',
                    '151507_1',
                    'RNA_xiong_with_clusters','ADT_xiong',
                    '151671_gaussian_noise','151671_gaussian_noise2',
                    'E13_rna_spatial','E15_rna_spatial','E18_rna_spatial','E13_ATAC_spatial','E15_ATAC_spatial','E18_ATAC_spatial',
                    'simulation1_moni_RNA','simulation1_moni_ADT','simulation2_moni_RNA','simulation2_moni_ADT','simulation3_moni_RNA','simulation3_moni_ADT','simulation4_moni_RNA','simulation4_moni_ADT','simulation5_moni_RNA','simulation5_moni_ADT','simulation_moni_ADT','simulation_moni_RNA','simulation_moni_ATAC',
                    'simulation_moni_ADT_noise1','simulation_moni_RNA_noise1','simulation_moni_ATAC_noise1','simulation_moni_ADT_noise2','simulation_moni_RNA_noise2','simulation_moni_ATAC_noise2','simulation_moni_ADT_noise3','simulation_moni_RNA_noise3','simulation_moni_ATAC_noise3','simulation_moni_ADT_noise4','simulation_moni_RNA_noise4','simulation_moni_ATAC_noise4',
                    'mousep22_atac',
                    'mouse_spleen1_RNA','mouse_spleen2_RNA','mouse_spleen1_ADT','mouse_spleen2_ADT',
                    'E15_5-S1_rna_spatial','E15_5-S2_rna_spatial','E15_5-S1_atac_spatial','E15_5-S2_atac_spatial',]#151777是第七张切片截取最中间的一部分,

    if (isinstance(id, int) and id in range(80)):
        section_id = section_list[id]
    elif (isinstance(id, str) and id in section_list):
        section_id = id

    if (not path):
        path = os.path.join(home_dir, 'data', 'DLPFC', f'DLPFC_{section_id}.h5ad')
    url = f'{dataset_url}/DLPFC/DLPFC_{section_id}.h5ad'

    adata = sc.read_h5ad(check_file_location(path=path, url=url))
    adata.var_names_make_unique()
    cluster_num = 6 #len(set(adata.obs['cluster'])) - 1

    return adata, cluster_num ,f'dorsolateral prefrontal cortex (DLPFC), slice: {section_id}'


