import sys
sys.path.append('/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/code/model')
from model import Multiomics_plus, QWENModel, PromptEmbedder
from demo_generator import demo_input
import torch
import torch.nn as nn
import torch.optim as optim
import argparse



# Example usage:
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # # generate demo minibatch
    # generator = demo_input( n_batch=4,
    #                         gene_len=5,
    #                         atac_len=10,
    #                         gene_total_len=10,
    #                         atac_total_len=15,
    #                         gExpr_mask_frac=0.15,
    #                         aPos_mask_frac=0,
    #                         aExpr_mask_frac=0.15,
    #                         max_gene_num=1000,
    #                         max_expr=10,
    #                         max_pos=1000000)
    # RNA_module, ATAC_module = generator.generate_data()
    # torch.save(RNA_module, 'RNA_module.pt')
    # torch.save(ATAC_module, 'ATAC_module.pt')
    
    # 临时文件 for test
    RNA_module = torch.load('RNA_module.pt')
    ATAC_module = torch.load('ATAC_module.pt')
    for k in RNA_module:
        RNA_module[k] = RNA_module[k].to(device)
    for k in ATAC_module:
        ATAC_module[k] = ATAC_module[k].to(device)


    
    # 指定模型路径
    model_path = '/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/qwen_model/qwen'

    prompt = (
        "The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. "
        # "Each token represents information about either a gene or an ATAC peak..."
    )

    prompt_embedder = PromptEmbedder(model_path).to(device)
    prompt_embeddings = prompt_embedder.encode_text(prompt, device=device)

    print(prompt_embeddings.shape)

    # init moodel
    Mymodel=Multiomics_plus(gene_vocab_size= 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                             pad_id=0,
                             mask_id=1,
                             cls_id=2,
                             RNAstart_id=19068,
                             RNAend_id=19069,
                             ATACstart_id=19070,
                             ATACend_id=19071,
                             ebdim_total=896,
                             pad_value=0,
                             mask_value=-1,
                             nhead = 16,
                             num_layers = 2,
                             dropout = 0.1).to(device)

    # test forward
    # out_RNA, out_ATAC = Mymodel(RNA_module, ATAC_module)


    # test forward before LLM
    LMM_input = Mymodel.forward_beforeLLM(RNA_module, ATAC_module, prompt_embeddings)
    
    # qwen_model = QWENModel(model_path)
    # # print(qwen_model)

    # outputs = qwen_model(LMM_input)
    # print(LMM_input.size)
    # print(outputs.size)
    

    # print(outputs)



# def main():
#     parser = argparse.ArgumentParser(description="Multiomic Model Test")
#     parser.add_argument("--config", 
#                         default="/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/code/config/config_temp.json", 
#                         help="Path to model config")
#     parser.add_argument("--checkpoint_dir", 
#                         default="/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/code/checkpoints", 
#                         help="Checkpoint directory")
#     args = parser.parse_args()
    
#     # training_sys = TrainingSystem(
#     #     config_path=args.config,
#     #     checkpoint_dir=args.checkpoint_dir,
#     #     resume_from=None       #start here
#     # )
    
# """
#     try:
#         training_sys.train(total_epochs=1)
#     finally:
#         if training_sys.world_size > 1:
#             torch.distributed.destroy_process_group()
# """

