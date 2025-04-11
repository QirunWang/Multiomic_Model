dsub -n trainingWQR\
     -N 3\
     -A root.project.P24Z28400N0259_tmp\
     -R "cpu=4;gpu=4;mem=100000"\
     -oo /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/logs/test_task.%J.out\
     -eo /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/logs/test_task.%J.err\
     -s /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/dsub.sh\

